"""
Optional LIVE-inference wrapper for the demo UI.

This is the "reach" feature of the demo -- everything in curated_examples.py
needs zero external dependencies and always works; this module actually
loads Qwen2.5-Coder-1.5B-Instruct via MLX and generates a fresh PyMongo
query for a question typed in at demo time. It exists to let Tarun show the
professor a genuinely new question, not just replay canned results.

Design choice (explicitly not over-engineered): this module does NOT
reimplement generation, prompt assembly, or scoring -- it imports and calls
the exact functions the repo's own scripts already use (rag/build_prompts.py,
rag/embed_utils.py, evaluation/execute_queries.py, fine_tuning/spot_check.py),
via sys.path insertion, so a live demo answer is produced by the SAME code
path that generated the reported benchmark numbers. Nothing here re-derives
prompt text or generation parameters independently.

Reliability note: model loading (~seconds, cold start) and Atlas execution
are both optional and independently guarded -- a failure in one (e.g. Atlas
unreachable from the demo laptop's network) does not block the other, and
every failure is logged and surfaced in the UI rather than crashing the app.

Restricted scope on purpose: live 23-db-adapter fine-tuned generation only
supports the 23 databases the adapter was actually trained on (see
KNOWN_23DB_DATABASES below) -- for an unknown database there is no
"reconstruct the training prompt" fallback, matching the same hard
requirement fine_tuning/generate_predictions_23db.py enforces (byte-identical
prompt to training, or refuse). Baseline and RAG have no such restriction --
they can run on any database with a local schema dump.
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("demo_ui.live_inference")

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"

ADAPTER_PATHS = {
    "baseline": None,
    "rag": None,
    "finetuned_6db": ROOT / "fine_tuning" / "adapters",
    "finetuned_23db_200iter": ROOT / "fine_tuning" / "adapters_23db",
    "finetuned_23db_1000iter": ROOT / "fine_tuning" / "adapters_23db_1000iter",
}
DEFAULT_FT_VARIANT = "finetuned_23db_1000iter"  # the epoch-parity-fixed adapter -- current best

for _p in (ROOT, ROOT / "rag", ROOT / "fine_tuning", ROOT / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_23db_manifest():
    manifest_path = ROOT / "fine_tuning" / "data_23db" / "split_manifest.json"
    if not manifest_path.exists():
        logger.error("23db split manifest missing at %s -- fine-tuned live inference unavailable", manifest_path)
        return None, set()
    manifest = json.loads(manifest_path.read_text())
    db_to_batch = manifest["database_to_batch_index"]
    batches = manifest["batches"]
    db_to_prompt = {db: batches[idx]["system_prompt"] for db, idx in db_to_batch.items()}
    logger.info("Loaded 23db split manifest: %d known databases", len(db_to_prompt))
    return db_to_prompt, set(db_to_prompt.keys())


DB_TO_23DB_SYSTEM_PROMPT, KNOWN_23DB_DATABASES = _load_23db_manifest()


class ModelUnavailable(RuntimeError):
    """Raised when mlx_lm or a required model/adapter path isn't available on this machine."""


_MODEL_CACHE = {}  # keyed by adapter_path string (or "" for none) -> (model, tokenizer)


def _get_model(adapter_path: Path | None):
    """Load (and cache) MLX model + tokenizer, optionally with a LoRA adapter attached.

    Caching is per-adapter because mlx_lm.load() re-reads and re-attaches
    adapter weights on every call -- without this cache, switching tabs in
    the Streamlit app would reload the whole 1.5B model from disk every time.
    """
    key = str(adapter_path) if adapter_path else ""
    if key in _MODEL_CACHE:
        logger.info("Model cache hit for adapter=%s", key or "(none, base model)")
        return _MODEL_CACHE[key]

    try:
        from mlx_lm import load
    except ImportError as exc:
        logger.error("mlx_lm not importable: %s", exc)
        raise ModelUnavailable(
            "mlx_lm is not installed in this Python environment. Live inference requires running "
            "Streamlit from the project's own .venv (source .venv/bin/activate) on the Mac -- it will "
            "not work in a generic system Python."
        ) from exc

    if adapter_path is not None and not Path(adapter_path).exists():
        logger.error("Adapter path does not exist: %s", adapter_path)
        raise ModelUnavailable(f"Adapter directory not found: {adapter_path}")

    logger.info("Loading model=%s adapter=%s (cold start, may take several seconds)...", MODEL_ID, key or "(none)")
    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)
    logger.info("Model load complete for adapter=%s", key or "(none, base model)")
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _generate(model, tokenizer, system_prompt: str, question: str, max_tokens: int = 300) -> str:
    from mlx_lm import generate
    from spot_check import clean  # fine_tuning/spot_check.py -- canonical post-processing, reused verbatim

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    logger.info("Generating (max_tokens=%d, question=%r)", max_tokens, question[:80])
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    cleaned = clean(raw)
    logger.info("Generation complete, %d chars raw -> %d chars cleaned", len(raw), len(cleaned))
    return cleaned


_BASELINE_PROMPT_CACHE = {}  # {"prompt": str, "databases": set, "coverage": dict} -- built once, reused


def _get_baseline_system_prompt() -> str:
    """The REAL baseline system prompt, reproduced exactly as generate_baseline_mlx.py builds it --
    NOT restricted to one database. The actual benchmark shows the model ONE prompt containing the
    schemas of every database in the whole test slice (rag/data/rag_test.json), and the model has to
    figure out which database/collection a question is about purely from the question's wording --
    same as a student handed a textbook with 20+ chapters and asked a question that only makes sense
    against one of them, with no chapter number given.

    Earlier version of this function narrowed the prompt to just the caller's chosen `database` --
    that was an unintentional simplification that made live-demo baseline easier than the real
    benchmark condition it's supposed to represent. Fixed 2026-08-29 per explicit instruction to
    match the benchmark exactly, even though it means a much larger prompt (rebuilt once and cached,
    since re-reading local schema dumps for 20+ databases on every call would be wasteful).
    """
    if "prompt" in _BASELINE_PROMPT_CACHE:
        logger.info("Using cached baseline mega-prompt (%d chars, %d databases)",
                    len(_BASELINE_PROMPT_CACHE["prompt"]), len(_BASELINE_PROMPT_CACHE["databases"]))
        return _BASELINE_PROMPT_CACHE["prompt"]

    from generate_baseline_mlx import build_full_schema_system_prompt  # repo root script

    test_path = ROOT / "rag" / "data" / "rag_test.json"
    if not test_path.exists():
        raise ModelUnavailable(f"{test_path} missing -- cannot reconstruct the real baseline test-slice "
                                f"database set.")
    cases = json.loads(test_path.read_text(encoding="utf-8"))
    test_databases = {c["database"] for c in cases}
    logger.info("Building real baseline mega-prompt over %d databases from %d test-slice cases "
                "(this matches generate_baseline_mlx.py's actual eval condition, not a per-question "
                "single-db shortcut)...", len(test_databases), len(cases))

    system_prompt, coverage, n_colls = build_full_schema_system_prompt(
        ROOT / "rag" / "data" / "rag_prompts.json", test_databases,
        label="live-demo baseline, reproducing the real eval condition",
    )
    logger.info("Baseline mega-prompt built: %d chars, %d databases covered, %d collections, "
                "%d databases missing a real schema dump: %s",
                len(system_prompt), len(coverage["covered"]), n_colls,
                len(coverage.get("missing", [])), sorted(coverage.get("missing", [])))
    _BASELINE_PROMPT_CACHE["prompt"] = system_prompt
    _BASELINE_PROMPT_CACHE["databases"] = test_databases
    _BASELINE_PROMPT_CACHE["coverage"] = coverage
    return system_prompt


def run_baseline(question: str, database: str, max_tokens: int = 300) -> dict:
    """Zero-shot baseline: base model, the REAL multi-database schema prompt (see
    _get_baseline_system_prompt), no retrieval, no adapter.

    `database` is NOT used to narrow the prompt -- the real baseline eval never tells the model
    which database a question is about either. It's accepted here only so the UI can show you,
    for reference, which database you *intended* to ask about, alongside whatever the model
    actually decided to answer against (inspect the generated query's collection name to see).
    """
    logger.info("run_baseline(intended_database=%s) -- note: baseline sees ALL test-slice databases, "
                "not just this one; the model must infer the right one from the question text alone",
                database)
    model, tokenizer = _get_model(None)
    system_prompt = _get_baseline_system_prompt()
    query = _generate(model, tokenizer, system_prompt, question, max_tokens)
    return {"arm": "baseline", "intended_database": database, "question": question, "generated_query": query}


def run_rag(question: str, database: str, top_k: int = 10, max_tokens: int = 300) -> dict:
    """RAG: retrieve top_k few-shot examples for `question`, build a schema+examples prompt, no adapter."""
    import faiss
    from embed_utils import embed
    from build_prompts import (
        PROMPT_HEADER, PROMPT_RULES, render_examples_block, render_schema_block,
        render_numeric_string_note, majority_vote_database,
    )
    from schema_cards import build_cards

    logger.info("run_rag(database=%s, top_k=%d)", database, top_k)
    index_path = ROOT / "rag" / "data" / "fewshot.index"
    meta_path = ROOT / "rag" / "data" / "fewshot_metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise ModelUnavailable(f"RAG index/metadata missing ({index_path} / {meta_path}) -- run "
                                f"rag/build_retrieval_index.py first.")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text())

    qvec = embed([question])
    _scores, idxs = index.search(qvec, top_k)
    neighbors = [metadata[i] for i in idxs[0] if 0 <= i < len(metadata)]
    predicted_database = majority_vote_database([n["database"] for n in neighbors]) if neighbors else database
    logger.info("RAG retrieval: %d neighbors, predicted_database=%s (caller-supplied=%s)",
                len(neighbors), predicted_database, database)

    cards_by_db: dict = {}
    for card in build_cards():
        cards_by_db.setdefault(card["database"], []).append(card)

    system_prompt = (
        PROMPT_HEADER
        + render_examples_block(neighbors)
        + "\n"
        + render_schema_block(predicted_database, cards_by_db, include_fk=True)
        + render_numeric_string_note(predicted_database)
        + "\n"
        + PROMPT_RULES
    )
    model, tokenizer = _get_model(None)
    query = _generate(model, tokenizer, system_prompt, question, max_tokens)
    return {
        "arm": "rag", "database": database, "predicted_database": predicted_database,
        "question": question, "generated_query": query,
        "retrieved_neighbor_ids": [n.get("id") for n in neighbors],
    }


def run_finetuned(question: str, database: str, variant: str = DEFAULT_FT_VARIANT, max_tokens: int = 300) -> dict:
    """Fine-tuned: base model + LoRA adapter. 23-db variants require `database` to be one of the
    23 databases the adapter was actually trained on (see module docstring for why)."""
    logger.info("run_finetuned(database=%s, variant=%s)", database, variant)
    if variant not in ADAPTER_PATHS or ADAPTER_PATHS[variant] is None:
        raise ModelUnavailable(f"Unknown fine-tuned variant: {variant!r}")

    if "23db" in variant:
        if not KNOWN_23DB_DATABASES:
            raise ModelUnavailable("23db split manifest failed to load -- cannot build a training-identical "
                                    "prompt. Live fine-tuned inference unavailable for this variant.")
        if database not in KNOWN_23DB_DATABASES:
            raise ModelUnavailable(
                f"'{database}' was not one of the 23 databases this adapter was trained on. "
                f"Known databases: {sorted(KNOWN_23DB_DATABASES)}"
            )
        system_prompt = DB_TO_23DB_SYSTEM_PROMPT[database]
    else:
        from prepare_data import SYSTEM_PROMPT  # fine_tuning/prepare_data.py -- fixed 6-db prompt
        system_prompt = SYSTEM_PROMPT

    model, tokenizer = _get_model(ADAPTER_PATHS[variant])
    query = _generate(model, tokenizer, system_prompt, question, max_tokens)
    return {"arm": "finetuned", "variant": variant, "database": database, "question": question,
            "generated_query": query}


def execute_against_atlas(query: str, database: str) -> dict:
    """Optional: actually run a generated query against live Atlas and return {status, result, error}.
    Off by default in the UI -- opt-in, since it needs atlas-credentials.env and a live connection.

    IMPORTANT: raw PyMongo results can contain BSON types (ObjectId, datetime) that Python's stock
    json module -- and therefore Streamlit's st.json() -- cannot serialize. Every other script in this
    repo runs results through evaluation/execute_queries.py's to_json_safe() before displaying or
    storing them (see run_model / execute_gold.py); this function was missing that step, which is
    exactly what produced the "src property must be a valid json object" error in the UI when a
    result happened to carry a raw ObjectId. Fixed 2026-08-29 -- reusing to_json_safe(), not
    reimplementing BSON conversion.
    """
    from execute_queries import safe_eval_query, to_json_safe  # evaluation/execute_queries.py -- same AST-gated eval
    import atlas_env

    logger.info("execute_against_atlas(database=%s, query=%r)", database, query[:120])
    try:
        client = atlas_env.connect()
        db = client[database]
        raw_result = safe_eval_query(query, db)
        raw_type = type(raw_result).__name__

        # Mirror evaluation/execute_queries.py's run_model() EXACTLY: a query like
        # db.collection.find(...) returns a live, un-materialized pymongo Cursor, not a list --
        # iterating it here is required before it can be converted to JSON or displayed at all.
        # (This step was missing in the previous fix, which only added to_json_safe() -- that
        # alone doesn't help if what's handed to it is still a raw Cursor object; to_json_safe()
        # only knows how to recurse into dict/list, so an un-materialized Cursor passed straight
        # through it, got stringified into something like "<pymongo.cursor.Cursor object at ...>"
        # by Streamlit's JSON encoder, and a bare string at the JSON root is exactly what makes
        # the frontend's JSON viewer throw "src property must be a valid json object".)
        if isinstance(raw_result, (int, float, str, bool)):
            pass
        elif not isinstance(raw_result, list):
            raw_result = list(raw_result)

        result = to_json_safe(raw_result)
        logger.info("Atlas execution OK, raw type=%s -> materialized+JSON-safe result of type=%s",
                    raw_type, type(result).__name__)
        return {"status": "PASS", "result": result, "error": None}
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: surface ANY failure to the UI, never crash it
        logger.exception("Atlas execution failed")
        return {"status": "FAIL", "result": None, "error": str(exc)}
