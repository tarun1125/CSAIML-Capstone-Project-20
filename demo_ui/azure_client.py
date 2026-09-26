"""Client for the DEPLOYED RAG service (service/app.py on Azure Container Apps).

Used by the "RAG on Azure" tab of demo_ui/app.py. The service URL and API key
come from the environment (AZURE_SERVICE_URL, SERVICE_API_KEY) or, failing
that, from Azure itself via the az CLI and the resource names in the gitignored
azure.env -- the key is read from Key Vault into memory and never written or
displayed.

The deployed service is paused between sessions (infra/pause.sh disables
ingress), so every failure mode here returns a message that says what to do,
rather than raising into the UI.
"""

import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("demo_ui.azure_client")

_TARGET: dict = {}


class AzureServiceUnavailable(RuntimeError):
    """The deployed service can't be reached or isn't configured; the message says why."""


def _az(*args) -> str:
    try:
        out = subprocess.run(["az", *args, "-o", "tsv"], capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise AzureServiceUnavailable("Azure CLI (az) not found -- set AZURE_SERVICE_URL and SERVICE_API_KEY") from exc
    if out.returncode != 0:
        raise AzureServiceUnavailable(f"az {' '.join(args[:3])} failed -- are you logged in (az login)?")
    return out.stdout.strip()


def target() -> dict:
    """{'url', 'key'}; resolved once per process."""
    if _TARGET:
        return _TARGET
    url, key = os.environ.get("AZURE_SERVICE_URL"), os.environ.get("SERVICE_API_KEY")
    if not (url and key):
        env_file = ROOT / "azure.env"
        if not env_file.exists():
            raise AzureServiceUnavailable("azure.env not found -- set AZURE_SERVICE_URL and SERVICE_API_KEY")
        from atlas_env import load_env_file  # noqa: PLC0415

        env = load_env_file(env_file)
        if not url:
            fqdn = _az("containerapp", "show", "-g", env["RG"], "-n", env["APP"],
                       "--query", "properties.configuration.ingress.fqdn")
            if not fqdn:
                raise AzureServiceUnavailable("the app has no public URL -- ingress is off; run ./infra/resume.sh")
            url = "https://" + fqdn
        if not key:
            key = _az("keyvault", "secret", "show", "--vault-name", env["KV"], "-n", "service-api-key",
                      "--query", "value")
    _TARGET.update(url=url.rstrip("/"), key=key)
    logger.info("Azure service target resolved: %s", _TARGET["url"])
    return _TARGET


def health(timeout: float = 10) -> dict:
    t = target()
    try:
        with urllib.request.urlopen(t["url"] + "/healthz", timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise AzureServiceUnavailable(
                "403 from Azure -- your current IP isn't allowed. Update MYIP in azure.env, then run ./infra/resume.sh"
            ) from exc
        raise AzureServiceUnavailable(f"health check returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AzureServiceUnavailable(
            f"can't reach {t['url']} ({exc}) -- the service is probably paused; run ./infra/resume.sh"
        ) from exc


def ask(question: str, execute: bool = True, timeout: float = 120) -> dict:
    """POST /query. The first call after scale-to-zero can take ~30 s (measured cold start)."""
    t = target()
    req = urllib.request.Request(
        t["url"] + "/query", method="POST",
        data=json.dumps({"question": question, "execute": execute}).encode(),
        headers={"x-api-key": t["key"], "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        hints = {401: "API key rejected", 429: "rate limit (10/min) -- wait a minute", 400: "question rejected",
                 403: "your IP isn't allowed -- update MYIP and run ./infra/resume.sh"}
        raise AzureServiceUnavailable(f"HTTP {exc.code}: {hints.get(exc.code, detail)}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AzureServiceUnavailable(f"request failed ({exc}) -- is the service resumed?") from exc
