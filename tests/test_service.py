# Tests for the served pipeline (service/) and the shared Azure OpenAI call
# (rag/aoai_client.py). No network, no model, no Atlas: every external piece
# is a fake.
#
# The one that matters most is test_served_prompt_is_the_benchmark_prompt: fed
# the exemplars recorded in rag/data/rag_prompts_rank1_k10.json, the service
# must build a byte-identical system prompt. That is the "import, don't copy"
# rule of docs/AZURE-PLAN.md, checked rather than asserted.
#
#   python -m pytest tests/ -q

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag"))

from fastapi.testclient import TestClient  # noqa: E402

from service.app import FixedWindowLimiter, create_app  # noqa: E402
from service.pipeline import RagPipeline, take_rows  # noqa: E402

RAG_DATA = REPO_ROOT / "rag" / "data"
METADATA = json.loads((RAG_DATA / "fewshot_metadata.json").read_text(encoding="utf-8"))
ROW_OF_ID = {m["id"]: i for i, m in enumerate(METADATA)}


def cards_by_db():
    from schema_cards import build_cards
    out = {}
    for c in build_cards():
        out.setdefault(c["database"], []).append(c)
    return out


CARDS = cards_by_db()


class FakeRetriever:
    name = "fake"

    def __init__(self, rows):
        self.rows = rows

    def search(self, question, qvec, k):
        return [(r, 1.0) for r in self.rows[:k]]


class FakeCursor:
    def __init__(self, docs):
        self.docs, self.pulled = docs, 0

    def __iter__(self):
        for d in self.docs:
            self.pulled += 1
            yield d

    def sort(self, *a, **k):
        return self

    def limit(self, n):
        return FakeCursor(self.docs[:n])


class FakeDB:
    def __init__(self, docs):
        self.docs, self.cursors = docs, []

    def __getattr__(self, name):
        db = self

        class Coll:
            def find(self, *a, **k):
                c = FakeCursor(db.docs)
                db.cursors.append(c)
                return c

            def count_documents(self, *a, **k):
                return len(db.docs)
        return Coll()

    __getitem__ = __getattr__


class FakeMongo(dict):
    def __missing__(self, name):
        self[name] = FakeDB([{"n": i} for i in range(200)])
        return self[name]


def make_pipeline(rows, query, db_policy="rank1", mongo=None, capture=None):
    def generate(system_prompt, question):
        if capture is not None:
            capture.append(system_prompt)
        return {"generated_query": query, "model": "fake", "prompt_tokens": 1, "completion_tokens": 1}
    return RagPipeline(embed=lambda texts: None, retriever=FakeRetriever(rows), metadata=METADATA,
                       cards_by_db=CARDS, generate=generate,
                       mongo_client=mongo if mongo is not None else FakeMongo(),
                       db_policy=db_policy)


# --- the benchmark contract -------------------------------------------------

def test_served_prompt_is_the_benchmark_prompt():
    cases = json.loads((RAG_DATA / "rag_prompts_rank1_k10.json").read_text(encoding="utf-8"))[:40]
    for case in cases:
        captured = []
        p = make_pipeline([ROW_OF_ID[i] for i in case["retrieved_ids"]], "db.x.count_documents({})",
                          capture=captured)
        ans = p.answer(case["question"], execute=False)
        assert ans.database == case["predicted_database"], case["id"]
        assert captured[0] == case["system_prompt"], f"prompt drift on {case['id']}"


def test_vote_policy_matches_the_vote_prompts_file():
    case = json.loads((RAG_DATA / "rag_prompts.json").read_text(encoding="utf-8"))[0]
    p = make_pipeline([ROW_OF_ID[i] for i in case["retrieved_ids"]], "db.x.count_documents({})",
                      db_policy="vote")
    assert p.answer(case["question"], execute=False).database == case["predicted_database"]


# --- safety -----------------------------------------------------------------

def test_empty_allowlist_refuses_to_start():
    with pytest.raises(RuntimeError, match="allowlist"):
        RagPipeline(embed=None, retriever=None, metadata=[], cards_by_db={}, generate=None,
                    mongo_client=None)


@pytest.mark.parametrize("query", [
    'db.client["other"].x.find({})',          # back-reference to another database
    '"a" * 10**12',                             # pure-Python blowup
    '__import__("os").system("id")',
    'db.x.aggregate([{"$out": "pwned"}])',
    "",                                         # empty generation
])
def test_unsafe_generation_is_never_executed(query):
    mongo = FakeMongo()
    ans = make_pipeline([0], query, mongo=mongo).answer("q")
    assert not ans.safe and ans.rejected_reason
    assert ans.rows is None and not mongo, "nothing may touch the database"


def test_rows_are_capped_without_draining_the_cursor():
    mongo = FakeMongo()
    ans = make_pipeline([0], "db.singer.find({})", mongo=mongo).answer("q")
    assert ans.safe and ans.row_count == 50 and ans.truncated
    (db,) = mongo.values()
    assert db.cursors[0].pulled == 51, "reads max_rows + 1 to detect truncation, no more"


def test_execution_errors_are_data_not_500s():
    ans = make_pipeline([0], "db.x.find({}).nonexistent_but_allowed()", mongo=FakeMongo()).answer("q")
    assert not ans.safe  # rejected by the method allowlist first
    ans = make_pipeline([0], "db.x.find({}).sort({'a': 1}).limit(3)", mongo=FakeMongo()).answer("q")
    assert ans.safe and ans.row_count == 3 and ans.error is None


def test_take_rows_shapes():
    assert take_rows(None, 5) == ([], False)
    assert take_rows({"a": 1}, 5) == ([{"a": 1}], False)
    assert take_rows(7, 5) == (7, False)
    assert take_rows(iter(range(10)), 3) == ([0, 1, 2], True)


def test_latency_is_reported_per_stage():
    ans = make_pipeline([0], "db.x.count_documents({})").answer("q")
    assert set(ans.latency_ms) >= {"embed", "retrieve", "generate", "guard", "execute", "total"}
    assert ans.rows == 200


# --- HTTP layer -------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app(pipeline=make_pipeline([0], "db.x.count_documents({})"),
                     env={"SERVICE_API_KEY": "test-key", "RATE_LIMIT_PER_MIN": "3"})
    with TestClient(app) as c:
        yield c


def test_refuses_to_start_without_an_api_key():
    with pytest.raises(RuntimeError, match="SERVICE_API_KEY"):
        create_app(pipeline=object(), env={})


def test_healthz_needs_no_key(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["n_databases"] == 23


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}])
def test_query_requires_the_key(client, headers):
    assert client.post("/query", json={"question": "q"}, headers=headers).status_code == 401


def test_long_questions_rejected(client):
    r = client.post("/query", json={"question": "x" * 501}, headers={"x-api-key": "test-key"})
    assert r.status_code == 400


def test_rate_limit(client):
    h = {"x-api-key": "test-key"}
    codes = [client.post("/query", json={"question": "q"}, headers=h).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_response_shape_and_logs_never_hold_the_question(client, caplog):
    secret = "How many singers are there in the secret-question-text?"
    with caplog.at_level(logging.INFO, logger="service"):
        r = client.post("/query", json={"question": secret}, headers={"x-api-key": "test-key"})
    body = r.json()
    assert r.status_code == 200 and body["safe"] and body["database"] and body["question_sha256"]
    assert "secret-question-text" not in caplog.text


def test_limiter_window_resets():
    t = [0.0]
    lim = FixedWindowLimiter(2, clock=lambda: t[0])
    assert [lim.allow("a"), lim.allow("a"), lim.allow("a")] == [True, True, False]
    assert lim.allow("b")
    t[0] = 60.0
    assert lim.allow("a")


# --- the shared Azure OpenAI call ---------------------------------------------

def test_complete_sends_exactly_the_g1_request():
    from aoai_client import complete
    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            msg = SimpleNamespace(content="```python\ndb.x.find({})\n```<|im_end|>junk")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                   model="m", system_fingerprint="fp",
                                   usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))

    fake = SimpleNamespace(settings={"deployment": "gpt-4o"},
                           client=lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    out = complete(fake, "SYSTEM", "QUESTION")
    assert sent == {"model": "gpt-4o",
                    "messages": [{"role": "system", "content": "SYSTEM"},
                                 {"role": "user", "content": "QUESTION"}],
                    "temperature": 0.0, "max_tokens": 300, "seed": 42}
    assert out["generated_query"] == "db.x.find({})"  # clean(): fences and stop marker removed
