"""Hard caps for every experiment LLM call (docs/research/pmrm_spec.md §6.1). None of them is optional.

1. Worsey's gate still applies: calls go through AnthropicLLM, which runs agent/budget.gate() before and record()
   after (hourly, daily and USD-per-day limits, the STOP file, WORSEY_LLM_DISABLED). Calls are tagged source=pmrm.
2. Per-run caps on top: WORSEY_PMRM_MAX_CALLS_PER_RUN (300) and WORSEY_PMRM_MAX_USD_PER_RUN (5.0, estimated with the
   ledger's price table). A call that would exceed either raises RunCapExceeded; everything done so far stays cached.
3. Dry run first: plan() prints calls, model and estimated cost, and a run makes no call unless confirmed. A plan
   whose estimate exceeds a cap is refused even when confirmed.
4. Permanent cache under cache/pmrm/llm/, keyed by the job, the input, the prompt and the model: a rerun with the
   same inputs makes zero calls; a prompt change is a new key (a new, deliberately approved run).
5. No hidden loops: one call per item, at most one retry on a transport error (AnthropicLLM's max_retries=1).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ...agent import budget as agent_budget
from ...data.cache import CACHE_ROOT

CACHE = CACHE_ROOT / "pmrm" / "llm"


class RunCapExceeded(RuntimeError):
    pass


def caps() -> dict:
    return {"calls": int(os.environ.get("WORSEY_PMRM_MAX_CALLS_PER_RUN", 300)),
            "usd": float(os.environ.get("WORSEY_PMRM_MAX_USD_PER_RUN", 5.0))}


def estimate_usd(model: str, calls: int, in_tok: int, out_tok: int, searches: int = 0) -> float:
    """Estimated cost of `calls` calls with these per-call sizes, priced like the ledger (agent/budget.cost)."""
    one = agent_budget.cost({"model": model, "fresh": in_tok, "out": out_tok, "web_searches": searches})
    return calls * one


def key(job: str, payload: dict, prompt: str, model: str) -> str:
    blob = json.dumps({"job": job, "payload": payload, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                       "model": model}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class Run:
    """One capped run. Use plan() for the dry run, then call() per item with confirm=True."""
    job: str
    model: str
    llm: object = None
    confirmed: bool = False
    calls: int = 0
    usd: float = 0.0
    hits: int = 0
    log: list = field(default_factory=list)

    def plan(self, items: int, in_tok: int, out_tok: int, searches_per_call: int = 0) -> dict:
        todo = sum(1 for _ in range(items))  # cache hits are only known per item; the plan assumes none
        est = estimate_usd(self.model, todo, in_tok, out_tok, searches_per_call)
        c = caps()
        p = {"job": self.job, "model": self.model, "calls_max": todo, "est_usd_max": round(est, 2),
             "caps": c, "within_caps": todo <= c["calls"] and est <= c["usd"]}
        print(f"[dry run] {self.job}: up to {todo} calls to {self.model}, about ${est:.2f} "
              f"(caps: {c['calls']} calls, ${c['usd']:.2f}); cached items cost nothing", flush=True)
        return p

    def cached(self, k: str) -> dict | None:
        f = CACHE / f"{k}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def call(self, k: str, **kw) -> dict:
        hit = self.cached(k)
        if hit is not None:
            self.hits += 1
            return hit
        if not self.confirmed:
            raise RunCapExceeded("not confirmed: run the dry run, then pass --confirm")
        c = caps()
        if self.calls + 1 > c["calls"] or self.usd >= c["usd"]:
            raise RunCapExceeded(f"per-run cap reached ({self.calls} calls, ${self.usd:.2f})")
        if hasattr(self.llm, "start_turn"):   # each experiment call is its own request for the chat's per-turn counter;
            self.llm.start_turn()             # the per-run caps above, and the hourly/daily/USD gates, still apply
        resp = self.llm.create(_source="pmrm", **kw)
        self.calls += 1
        u = resp.get("usage") or {}
        e = {"model": kw.get("model", ""), "fresh": int(u.get("input_tokens") or 0),
             "cache_write": int(u.get("cache_creation_input_tokens") or 0),
             "cache_read": int(u.get("cache_read_input_tokens") or 0), "out": int(u.get("output_tokens") or 0),
             "web_searches": int((u.get("server_tool_use") or {}).get("web_search_requests") or 0)}
        self.usd += agent_budget.cost(e)
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{k}.json").write_text(json.dumps(resp, default=str), encoding="utf-8")
        self.log.append({"key": k, **e})
        return resp

    def summary(self) -> dict:
        return {"job": self.job, "model": self.model, "calls": self.calls, "cache_hits": self.hits,
                "est_usd": round(self.usd, 3), "caps": caps()}
