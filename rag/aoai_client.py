# One Azure OpenAI call, shared by the batch generator (rag/generate_rag_aoai.py)
# and the deployed service (service/pipeline.py), so a served answer is produced
# by exactly the call that produced the G1 numbers: same message split, same
# decoding parameters, same clean().
#
# Auth: an API key when one is configured (local runs, azure.env); otherwise
# Microsoft Entra ID via DefaultAzureCredential -- the container's managed
# identity in Azure, `az login` on a laptop. The identity needs the
# "Cognitive Services OpenAI User" role on the resource.

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "fine_tuning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from generation_utils import clean  # noqa: E402  canonical post-processing, MLX-free

# The G1 decoding. Changing any of these makes a different arm.
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 300
_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


def load_aoai_settings() -> dict:
    """AZURE_OPENAI_* from the environment first, then the gitignored azure.env."""
    values = {}
    env_file = REPO_ROOT / "azure.env"
    if env_file.exists():
        from atlas_env import load_env_file  # noqa: PLC0415

        values = load_env_file(env_file)
    get = lambda k: os.environ.get(k) or values.get(k)  # noqa: E731
    cfg = {
        "endpoint": get("AZURE_OPENAI_ENDPOINT"),
        "deployment": get("AZURE_OPENAI_DEPLOYMENT"),
        "api_key": get("AZURE_OPENAI_API_KEY"),  # optional: managed identity when absent
    }
    missing = [k for k in ("endpoint", "deployment") if not cfg[k]]
    if missing:
        raise RuntimeError(f"Azure OpenAI settings missing: {missing}")
    return cfg


class AoaiClient:
    """Thin wrapper that keeps an Entra token fresh when no key is set."""

    def __init__(self, settings: dict, max_retries: int = 6, timeout: float = 60):
        from openai import OpenAI  # noqa: PLC0415

        self._OpenAI = OpenAI
        self.settings = settings
        self.base_url = settings["endpoint"].rstrip("/") + "/openai/v1/"
        self.max_retries, self.timeout = max_retries, timeout
        self._credential = None
        self._token = None
        self._client = None
        if settings.get("api_key"):
            self._client = OpenAI(base_url=self.base_url, api_key=settings["api_key"],
                                  max_retries=max_retries, timeout=timeout)
        else:
            from azure.identity import DefaultAzureCredential  # noqa: PLC0415

            self._credential = DefaultAzureCredential()

    @property
    def auth_mode(self) -> str:
        return "api_key" if self._credential is None else "entra_id"

    def client(self):
        if self._credential is None:
            return self._client
        # Refresh five minutes before expiry; a token lasts about an hour.
        if self._token is None or self._token.expires_on - time.time() < 300:
            self._token = self._credential.get_token(_COGNITIVE_SCOPE)
            self._client = self._OpenAI(base_url=self.base_url, api_key=self._token.token,
                                        max_retries=self.max_retries, timeout=self.timeout)
        return self._client


def complete(aoai: AoaiClient, system_prompt: str, question: str,
             temperature: float = TEMPERATURE, seed: int = SEED, max_tokens: int = MAX_TOKENS) -> dict:
    """The G1 call. Returns the cleaned query plus what the manifests record."""
    r = aoai.client().chat.completions.create(
        model=aoai.settings["deployment"],
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": question}],
        temperature=temperature, max_tokens=max_tokens, seed=seed,
    )
    raw = r.choices[0].message.content or ""
    return {
        "generated_query": clean(raw),
        "raw_output": raw,
        "finish_reason": r.choices[0].finish_reason,
        "model": r.model,
        "system_fingerprint": r.system_fingerprint,
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
    }
