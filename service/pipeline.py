# The served RAG pipeline (docs/AZURE-PLAN.md, Phase 5).
#
#   question -> embed -> retrieve top-K -> pick database -> build prompt
#            -> generate -> normalize -> AST guard -> execute (read-only) -> rows
#
# IMPORT, DON'T COPY. Every step is the benchmark's own code: embed_utils.embed,
# rag/retrievers, build_prompts.majority_vote_database / build_system_prompt,
# rag/aoai_client.complete (the G1 call), normalize.normalize, and
# evaluation/execute_queries' guard. The configuration served by default is the
# best one measured: gpt-4o + rank-1 database policy (212/304,
# docs/FINDING-azure-generator.md §3.1).
#
# Everything is built ONCE (build_pipeline_from_env, at startup) and every
# external dependency is injected, so tests run it with fakes and no network.
#
# SAFETY, in the order it applies (the endpoint eval()s model output):
#   1. The database is the one retrieval predicted, and must be in the
#      allowlist of this project's 23 databases. An empty allowlist refuses
#      startup (it used to mean "allow everything" in the demo).
#   2. normalize() then check_query_is_safe() -- the AST allowlist, which also
#      rejects .client/.database back-references and * ** % << arithmetic.
#   3. Execution against ONE database handle, with a short socket timeout, and
#      at most max_rows rows pulled from any cursor (never materialized whole).
#   4. The Atlas user in production is READ-ONLY (docs/AZURE-PLAN.md, Security).

import contextlib
import io
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "rag", REPO_ROOT / "evaluation", REPO_ROOT / "fine_tuning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# build_prompts pulls in embed_utils (torch) first -- the import-order rule of
# rag/build_prompts.py still applies to anything that later loads faiss.
from build_prompts import build_system_prompt, majority_vote_database  # noqa: E402
from execute_queries import check_query_is_safe, to_json_safe  # noqa: E402
from normalize import normalize  # noqa: E402

DB_POLICIES = ("rank1", "vote")


@dataclass
class Answer:
    database: str
    db_policy: str
    retrieved_ids: list
    query: str = ""
    safe: bool = False
    rejected_reason: str | None = None
    rows: list | int | float | str | bool | None = None
    row_count: int | None = None
    truncated: bool = False
    error: str | None = None
    model: str | None = None
    tokens: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)


def take_rows(result, max_rows: int):
    """At most max_rows rows from whatever the query returned, without
    materializing a whole cursor. Returns (rows, truncated). Mirrors
    execute_queries.materialize_result's shape rules (find_one's dict -> [dict],
    None -> []), but never calls list() on an unbounded iterable."""
    if isinstance(result, (int, float, str, bool)):
        return result, False
    if result is None:
        return [], False
    if isinstance(result, dict):
        return [result], False
    head = list(itertools.islice(iter(result), max_rows + 1))
    return head[:max_rows], len(head) > max_rows


class _Timer:
    def __init__(self, latency: dict, tracer):
        self.latency, self.tracer = latency, tracer

    @contextlib.contextmanager
    def stage(self, name: str):
        span_cm = self.tracer.start_as_current_span(name) if self.tracer else contextlib.nullcontext()
        t0 = time.perf_counter()
        with span_cm:
            try:
                yield
            finally:
                self.latency[name] = round((time.perf_counter() - t0) * 1000, 1)


class RagPipeline:
    def __init__(self, *, embed: Callable, retriever, metadata: list[dict], cards_by_db: dict,
                 generate: Callable, mongo_client, db_policy: str = "rank1", top_k: int = 10,
                 max_rows: int = 50, tracer=None):
        if db_policy not in DB_POLICIES:
            raise ValueError(f"db_policy must be one of {DB_POLICIES}")
        self.allowed_databases = frozenset(cards_by_db)
        if not self.allowed_databases:
            # Fail closed: without an allowlist there is nothing to check the
            # predicted database against, so nothing may be executed.
            raise RuntimeError("no schema cards loaded -- refusing to start without a database allowlist")
        self.embed, self.retriever, self.metadata = embed, retriever, metadata
        self.cards_by_db, self.generate, self.mongo = cards_by_db, generate, mongo_client
        self.db_policy, self.top_k, self.max_rows, self.tracer = db_policy, top_k, max_rows, tracer

    def answer(self, question: str, execute: bool = True) -> Answer:
        latency: dict = {}
        timer = _Timer(latency, self.tracer)
        t_total = time.perf_counter()

        with timer.stage("embed"):
            qvec = self.embed([question])
        with timer.stage("retrieve"):
            hits = self.retriever.search(question, qvec, self.top_k)
        neighbors = [self.metadata[r] for r, _ in hits]
        dbs = [n["database"] for n in neighbors]
        database = dbs[0] if self.db_policy == "rank1" else majority_vote_database(dbs)
        ans = Answer(database=database, db_policy=self.db_policy,
                     retrieved_ids=[n["id"] for n in neighbors], latency_ms=latency)

        system_prompt = build_system_prompt(neighbors, database, self.cards_by_db, include_fk=True)
        with timer.stage("generate"):
            gen = self.generate(system_prompt, question)
        ans.model = gen.get("model")
        ans.tokens = {k: gen.get(k) for k in ("prompt_tokens", "completion_tokens")}

        with timer.stage("guard"):
            with contextlib.redirect_stdout(io.StringIO()):  # normalize() prints the query; keep it out of logs
                ans.query = normalize(gen["generated_query"]) if gen["generated_query"] else ""
            ok, reason = check_query_is_safe(ans.query) if ans.query else (False, "empty generation")
            if ok and database not in self.allowed_databases:
                ok, reason = False, f"database {database!r} not in allowlist"
            ans.safe, ans.rejected_reason = ok, (None if ok else reason)

        if ans.safe and execute:
            with timer.stage("execute"):
                try:
                    db = self.mongo[database]  # the ONLY name eval() can reach besides builtins it may call
                    result = eval(ans.query, {"__builtins__": {}}, {  # noqa: S307  guarded above
                        "db": db, "None": None, "True": True, "False": False,
                        "len": len, "sorted": sorted, "list": list, "dict": dict,
                    })
                    rows, ans.truncated = take_rows(result, self.max_rows)
                    ans.rows = to_json_safe(rows)
                    ans.row_count = len(rows) if isinstance(rows, list) else None
                except Exception as exc:  # noqa: BLE001 -- surface any failure as data, not a 500
                    ans.error = f"{type(exc).__name__}: {str(exc)[:300]}"

        latency["total"] = round((time.perf_counter() - t_total) * 1000, 1)
        return ans


def build_pipeline_from_env(env: dict, tracer=None) -> RagPipeline:
    """Production wiring. Heavy imports stay here so tests never pay for them."""
    from atlas_env import connect  # noqa: PLC0415
    from aoai_client import AoaiClient, complete, load_aoai_settings  # noqa: PLC0415
    from embed_utils import embed, get_embedder  # noqa: PLC0415
    from retrievers import AzureSearchRetriever, AZURE_MODES, FaissRetriever, load_azure_settings  # noqa: PLC0415
    from schema_cards import build_cards  # noqa: PLC0415

    get_embedder()  # load MiniLM now, not on the first request
    metadata = json.loads((REPO_ROOT / "rag" / "data" / "fewshot_metadata.json").read_text(encoding="utf-8"))
    cards_by_db: dict = {}
    for card in build_cards():
        cards_by_db.setdefault(card["database"], []).append(card)

    retriever_name = env.get("RETRIEVER", "azure-vector")
    if retriever_name == "faiss":
        retriever = FaissRetriever()
    elif retriever_name in AZURE_MODES:
        retriever = AzureSearchRetriever(AZURE_MODES[retriever_name][0],
                                         settings=load_azure_settings(allow_identity=True))
    else:
        raise ValueError(f"unknown RETRIEVER {retriever_name!r}")

    aoai = AoaiClient(load_aoai_settings())
    mongo = connect(socket_timeout_ms=int(env.get("MONGO_SOCKET_TIMEOUT_MS", "15000")), verbose=False)

    return RagPipeline(
        embed=embed, retriever=retriever, metadata=metadata, cards_by_db=cards_by_db,
        generate=lambda system_prompt, question: complete(aoai, system_prompt, question),
        mongo_client=mongo, db_policy=env.get("DB_POLICY", "rank1"),
        top_k=int(env.get("TOP_K", "10")), max_rows=int(env.get("MAX_ROWS", "50")), tracer=tracer,
    )
