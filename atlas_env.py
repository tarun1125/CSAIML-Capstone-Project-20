"""Shared Atlas env/connection helpers -- single source of truth.

Extracted this session to close a real DRY gap flagged in the project's own
code-smell audit: `load_env_file()` was duplicated verbatim in
`evaluation/execute_queries.py` and `atlas_verify_and_load.py`, and their two
`connect()` implementations had quietly diverged -- `atlas_verify_and_load.py`
had a connection timeout and password-masked logging, `execute_queries.py`
had neither (a bad/unreachable URI would hang on the driver's ~30s default
instead of failing fast, and any accidental print of a raw client/URI object
would leak the password in a log). Every script now gets the safer behavior
by importing from here instead of by which file it happened to be copied
into.

`socketTimeoutMS` (not just `serverSelectionTimeoutMS`) is also new here: the
audit separately flagged that no executed query anywhere had a time cap, so a
"safe-per-the-allowlist" but pathological query (uncapped cross-collection
$lookup, aggregate() with no $limit) could hang an unattended batch-scoring
run indefinitely. A per-query `maxTimeMS` would be the precise fix, but this
codebase `eval()`s an arbitrary model-generated PyMongo expression (see
`evaluation/execute_queries.py`'s `safe_eval_query`) -- there's no single
call site to attach `.max_time_ms()` to before the cursor is already
materialized. `socketTimeoutMS` at the connection level is the coarser but
correct-here alternative: it caps any single socket operation, including a
hanging query, without needing to intercept every possible eval'd query
shape. Documented as a deliberate tradeoff, not an oversight.
"""

from pathlib import Path

from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Generous enough that a normal aggregation with a real $lookup/$group never
# trips it, but bounded so an unattended run can't hang forever on one
# pathological query.
DEFAULT_SOCKET_TIMEOUT_MS = 60_000
DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 8_000


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mask_uri(uri: str) -> str:
    """mongodb+srv://user:password@host/... -> mongodb+srv://user:***@host/...
    so connection details stay traceable in logs without leaking the secret."""
    if "@" not in uri:
        return uri
    creds, rest = uri.split("@", 1)
    if ":" in creds:
        user, _password = creds.rsplit(":", 1)
        return f"{user}:***@{rest}"
    return f"{creds}@{rest}"


def find_env_file(start: Path) -> Path:
    """atlas-credentials.env always lives at the repo root. Callers pass
    their own file's location; this walks up to find it so both a
    top-level script and one nested under evaluation/ or rag/ resolve the
    same file without hardcoding a parents[N] depth that breaks if either
    script ever moves."""
    for candidate_root in [start, *start.parents]:
        candidate = candidate_root / "atlas-credentials.env"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"atlas-credentials.env not found by walking up from {start} -- "
        "this must run somewhere inside the project."
    )


def connect(
    *,
    socket_timeout_ms: int = DEFAULT_SOCKET_TIMEOUT_MS,
    server_selection_timeout_ms: int = DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    verbose: bool = True,
) -> MongoClient:
    env_file = find_env_file(Path(__file__).resolve().parent)
    env_values = load_env_file(env_file)
    uri = env_values.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(f"MONGODB_URI not found in {env_file}")

    if verbose:
        print(f"[connect] Connecting to {mask_uri(uri)}")

    client = MongoClient(
        uri,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=server_selection_timeout_ms,
        socketTimeoutMS=socket_timeout_ms,
    )
    client.admin.command("ping")
    if verbose:
        print("[connect] Pinged Atlas, connection OK")
    return client
