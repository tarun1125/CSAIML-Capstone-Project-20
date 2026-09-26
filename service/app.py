# FastAPI front end for the served RAG pipeline (docs/AZURE-PLAN.md, Phase 5).
#
#   GET  /healthz                         liveness + configuration, no auth, no external calls
#   POST /query  {"question", "execute"}  header x-api-key; returns database, query, rows, latency
#
#   uvicorn service.app:app --host 0.0.0.0 --port 8000
#
# Configuration is environment only (Container Apps sets it; locally, export it
# or rely on azure.env / atlas-credentials.env for the Azure/Atlas parts):
#   SERVICE_API_KEY        required; requests without it get 401
#   RETRIEVER              azure-vector (default) | azure-hybrid | faiss
#   DB_POLICY              rank1 (default, best measured) | vote
#   TOP_K / MAX_ROWS       10 / 50
#   MAX_QUESTION_CHARS     500
#   RATE_LIMIT_PER_MIN     10, per client IP
#   APPLICATIONINSIGHTS_CONNECTION_STRING   optional; enables per-stage traces
#
# Logging convention of this repo: never the raw question or prompt -- a
# sha256 prefix identifies a request across logs without storing its text.

import hashlib
import hmac
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger("service")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    execute: bool = True


class FixedWindowLimiter:
    """N requests per client per 60-second window.

    PLACEHOLDER. docs/AZURE-PLAN.md reserves the rate limiter for Tarun's own
    token bucket (dsa-design-drill, exercise 05). This fixed window is here
    only so the endpoint is never deployed unlimited; swap it for the token
    bucket, keeping the allow(key) -> bool interface."""

    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._counts: dict = defaultdict(lambda: [0.0, 0])
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            start, n = self._counts[key]
            if now - start >= self.window_s:
                self._counts[key] = [now, 1]
                return True
            if n < self.limit:
                self._counts[key][1] = n + 1
                return True
            return False


def _tracer():
    """App Insights tracing when a connection string is present; otherwise none."""
    if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        return None
    from azure.monitor.opentelemetry import configure_azure_monitor  # noqa: PLC0415
    from opentelemetry import trace  # noqa: PLC0415

    configure_azure_monitor()
    return trace.get_tracer("capstone-rag")


def client_ip(request: Request) -> str:
    # Container Apps ingress sets X-Forwarded-For; the first hop is the caller.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def create_app(pipeline=None, env: dict | None = None) -> FastAPI:
    env = dict(os.environ if env is None else env)
    api_key = env.get("SERVICE_API_KEY")
    if not api_key:
        raise RuntimeError("SERVICE_API_KEY is not set -- refusing to start an unauthenticated endpoint")
    max_chars = int(env.get("MAX_QUESTION_CHARS", "500"))
    limiter = FixedWindowLimiter(int(env.get("RATE_LIMIT_PER_MIN", "10")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            from service.pipeline import build_pipeline_from_env  # noqa: PLC0415

            t0 = time.perf_counter()
            app.state.pipeline = build_pipeline_from_env(env, tracer=_tracer())
            log.info("pipeline ready in %.1fs", time.perf_counter() - t0)
        yield

    app = FastAPI(title="Capstone NL-to-PyMongo RAG", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/healthz")
    def healthz():
        p = app.state.pipeline
        return {
            "status": "ok" if p is not None else "starting",
            "db_policy": getattr(p, "db_policy", None),
            "top_k": getattr(p, "top_k", None),
            "retriever": getattr(getattr(p, "retriever", None), "name", None),
            "n_databases": len(getattr(p, "allowed_databases", ())),
        }

    @app.post("/query")
    def query(body: QueryRequest, request: Request, x_api_key: str | None = Header(default=None)):
        # A sync def endpoint runs in FastAPI's threadpool, so a slow model or
        # Atlas call never blocks the event loop.
        if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="missing or invalid x-api-key")
        if len(body.question) > max_chars:
            raise HTTPException(status_code=400, detail=f"question longer than {max_chars} characters")
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

        request_id = uuid.uuid4().hex[:12]
        qhash = hashlib.sha256(body.question.encode("utf-8")).hexdigest()[:16]
        ans = app.state.pipeline.answer(body.question, execute=body.execute)
        log.info("request=%s q_sha256=%s db=%s safe=%s rows=%s error=%s latency_ms=%s",
                 request_id, qhash, ans.database, ans.safe, ans.row_count,
                 bool(ans.error), ans.latency_ms)
        return {"request_id": request_id, "question_sha256": qhash, **ans.__dict__}

    return app


def _app_from_environment():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s")
    return create_app()


# `uvicorn service.app:app` -- built lazily so importing this module in tests
# (which call create_app directly) doesn't require SERVICE_API_KEY.
def __getattr__(name):
    if name == "app":
        globals()["app"] = _app_from_environment()
        return globals()["app"]
    raise AttributeError(name)
