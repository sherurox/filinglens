r"""The analyst crew on LlamaIndex Workflows: planner, retriever, writer, numbers checker (T7).

Four roles, the spec's:
- planner (LLM, request class short_qa -> the FL-M4 student on vLLM, per the FL-M5 router): which
  companies, fiscal years, metrics and searches the question needs, as JSON constrained by a schema.
  A plan that doesn't validate (unknown ticker, bad year, malformed) is replaced by a rule-based
  plan read off the question, and the switch is recorded: the plan-validity rate is measured on dev.
- retriever (code): search_filings per planned search, get_financials per company (a tool error,
  e.g. a year before the data starts, becomes an "unavailable" line for the writer, not a crash),
  then the token-budgeted context builder (agents.context).
- writer (LLM, request class long_synthesis -> Gemini): numbered excerpts [n] and a numbered list
  of figures [Fn], each tied to its own 10-K's URL; every sentence must cite one of them, and it may
  state no number except the figures given.
- numbers checker (code): agents.numbers.check_answer (figure_matches, hand-written by the user)
  plus a citation check. A failed draft goes back to the writer once with the problems listed; a
  second failure blocks the answer. That gate is an event branch: Draft -> Failed -> Draft -> Stop.

Backends come from the FL-M5 router's LP shares (bench/router_results.json, scenario S1_gpu_on/main)
and are wrapped in the cost meter, so every model call is a row in serve.requests and an
OpenTelemetry span. Gemini calls are capped per run by a hard counter (--max-gemini-calls), so a
run cannot spend more of the 500-a-day free tier than its stated plan.

    uv run --group api --group agents python -m filinglens.agents.crew run --split dev \
        --budget 4096 --max-gemini-calls 20 --tag dev1
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from filinglens.agents.context import build_context
from filinglens.agents.numbers import Allowed, allowed_values, check_answer
from filinglens.agents.tools import FIRST_YEAR, LAST_YEAR, ToolError

ROUTER_RESULTS = Path("bench/router_results.json")
SCENARIO = "S1_gpu_on/main"
ROLES = {"planner": "short_qa", "writer": "long_synthesis", "extract": "extraction"}
MAX_ATTEMPTS = 2  # the first draft plus one revision
MAX_SEARCHES = 4
SEARCH_K = 8
PLAN_MAX_TOKENS = 400
WRITE_MAX_TOKENS = 700
WORKFLOW_TIMEOUT_S = 900.0  # above two 100 s Gemini deadlines plus retrieval
METRICS = (
    "revenue",
    "operating_income",
    "net_income",
    "operating_margin",
    "net_margin",
    "revenue_growth",
    "operating_margin_change",
    "net_margin_change",
)
TICKER_IN_PARENS = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
CITATION = re.compile(r"\[(F?\d+)\]")

Chat = Callable[..., Awaitable[Any]]  # MeteredBackend.chat(conv, schema, max_tokens, route, user)


# ---------------------------------------------------------------- routing


def route_table(results: dict, scenario: str = SCENARIO) -> dict[str, str]:
    """Request class -> backend with the largest LP share (S1: extraction and short_qa -> vllm,
    long_synthesis -> gemini). Ties go to the first listed."""
    shares = results["results"][scenario]["lp"]["shares"]
    best: dict[str, tuple[float, str]] = {}
    for key, share in shares.items():
        cls, backend = key.split("/")
        if cls not in best or share > best[cls][0]:
            best[cls] = (share, backend)
    return {cls: backend for cls, (_, backend) in best.items()}


class CallCap:
    """A hard per-run cap on calls to one backend: past it, calls fail without being sent."""

    def __init__(self, chat: Chat, limit: int | None, name: str) -> None:
        self.chat, self.limit, self.name, self.used = chat, limit, name, 0

    async def __call__(self, *args, **kwargs):
        if self.limit is not None and self.used >= self.limit:
            from filinglens.serve.backends import Reply

            return Reply(
                self.name,
                "-",
                "",
                0,
                0,
                0,
                0.0,
                None,
                f"cap: {self.limit} {self.name} calls planned for this run, all used",
            )
        self.used += 1
        return await self.chat(*args, **kwargs)


# ---------------------------------------------------------------- planner


def plan_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "companies": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "fiscal_years": {"type": "array", "items": {"type": "integer"}, "maxItems": 6},
            "metrics": {"type": "array", "items": {"enum": list(METRICS)}, "maxItems": 8},
            "searches": {
                "type": "array",
                "maxItems": MAX_SEARCHES,
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "fiscal_year": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                    "required": ["company", "fiscal_year", "query"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["companies", "fiscal_years", "metrics", "searches"],
        "additionalProperties": False,
    }


PLANNER_SYSTEM = (
    "You plan how to answer an analyst question about US public companies' 10-K filings. Reply "
    "with JSON only.\n"
    "- companies: the ticker of every company the question names (the symbol in parentheses).\n"
    "- fiscal_years: every fiscal year the question needs, as four-digit years (a growth or "
    "change question needs both years). 'The fiscal year that ended in 2023' is 2023.\n"
    f"- metrics: the financial figures it asks for, from {list(METRICS)}.\n"
    "- searches: one per company and fiscal year whose 10-K risk factors the question asks "
    f"about (at most {MAX_SEARCHES}); query = a short search phrase for that risk, such as "
    "'cybersecurity breach risk' or 'fuel price volatility'."
)


def planner_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]


def metrics_in(question: str) -> list[str]:
    q = question.lower()
    out = []
    if "revenue" in q:
        out.append("revenue")
    if "operating income" in q:
        out.append("operating_income")
    if "operating margin" in q:
        out += ["operating_margin"] + (["operating_margin_change"] if "change" in q else [])
    if "net margin" in q:
        out += ["net_margin", "net_income", "revenue"]
    if "grow" in q:
        out.append("revenue_growth")
    return list(dict.fromkeys(out))


def fallback_plan(question: str) -> dict:
    """The plan read off the question's own words: tickers in parentheses, four-digit years."""
    companies = list(dict.fromkeys(TICKER_IN_PARENS.findall(question)))[:3]
    years = sorted({int(y) for y in YEAR.findall(question)})[:6]
    searches = [
        {"company": c, "fiscal_year": y, "query": question} for c in companies for y in years
    ][:MAX_SEARCHES]
    return {
        "companies": companies,
        "fiscal_years": years,
        "metrics": metrics_in(question),
        "searches": searches,
    }


def validate_plan(raw: Any, known: Callable[[str], bool]) -> tuple[dict | None, list[str]]:
    """The plan if it is usable, else None and why. `known(ticker)` says if a company exists."""
    problems: list[str] = []
    if not isinstance(raw, dict) or set(raw) != {
        "companies",
        "fiscal_years",
        "metrics",
        "searches",
    }:
        return None, ["not an object with exactly companies, fiscal_years, metrics, searches"]
    companies = raw["companies"]
    if not companies or not all(isinstance(c, str) and known(c) for c in companies):
        problems.append(f"unknown or missing company in {companies!r}")
    years = raw["fiscal_years"]
    if not years or not all(isinstance(y, int) and 1990 <= y <= 2030 for y in years):
        problems.append(f"fiscal years not plausible: {years!r}")
    if not all(m in METRICS for m in raw["metrics"]):
        problems.append("unknown metric")
    for s in raw["searches"][:MAX_SEARCHES]:
        if s.get("company") not in companies or not str(s.get("query", "")).strip():
            problems.append(f"search not for a planned company or empty: {s!r}")
    if problems:
        return None, problems
    plan = {
        "companies": [c.upper() for c in companies],
        "fiscal_years": sorted(set(years)),
        "metrics": list(dict.fromkeys(raw["metrics"])),
        "searches": [s | {"company": s["company"].upper()} for s in raw["searches"][:MAX_SEARCHES]],
    }
    return plan, []


# ---------------------------------------------------------------- writer


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def money(x: float) -> str:
    return f"${x:,.0f}"


def figure_lines(financials: list[dict], unavailable: list[str]) -> tuple[list[str], dict]:
    """[Fn] lines for the writer and Fn -> {fiscal year: filing URL} for the citation check."""
    lines, cites = [], {}
    names = {
        "revenue": "revenue",
        "operating_income": "operating income",
        "net_income": "net income",
    }
    for res in financials:
        t = res["company"]["ticker"]
        years = res["years"]
        for e in years:
            n = f"F{len(lines) + 1}"
            fy = e["fiscal_year"]
            if not e.get("available"):
                lines.append(
                    f"[{n}] {t}, fiscal year ended in {fy}: no figures available"
                    f" ({e.get('reason') or 'not in the data'})."
                )
                cites[n] = {fy: e.get("filing_url")} if e.get("filing_url") else {}
                continue
            parts = [f"{names[m]} {money(f['value'])}" for m, f in e["figures"].items()]
            parts += [f"{k.replace('_', ' ')} {pct(d['value'])}" for k, d in e["derived"].items()]
            restated = [
                f"{names[m]} first reported as {money(f['first_reported_value'])}"
                for m, f in e["figures"].items()
                if f["was_restated"]
            ]
            missing = [names[m] for m in e["missing"]]
            text = f"[{n}] {t}, fiscal year ended in {fy} (10-K report date {e['report_date']}): "
            text += "; ".join(parts + restated) or "no figures"
            if missing:
                text += f"; not available: {', '.join(missing)}"
            lines.append(text + ".")
            cites[n] = {fy: e["filing_url"]}
        for g in res.get("growth", []):
            n = f"F{len(lines) + 1}"
            lines.append(
                f"[{n}] {t} revenue growth from the fiscal year ended in {g['from']} to"
                f" the one ended in {g['to']}: {pct(g['revenue_growth'])}."
            )
            cites[n] = {g["to"]: _url(years, g["to"]), g["from"]: _url(years, g["from"])}
        by_year = {e["fiscal_year"]: e for e in years if e.get("derived")}
        ys = sorted(by_year)
        for i, y1 in enumerate(ys):
            for y2 in ys[i + 1 :]:
                for k in set(by_year[y1]["derived"]) & set(by_year[y2]["derived"]):
                    change = by_year[y2]["derived"][k]["value"] - by_year[y1]["derived"][k]["value"]
                    n = f"F{len(lines) + 1}"
                    lines.append(
                        f"[{n}] {t} {k.replace('_', ' ')} change from {y1} to {y2}:"
                        f" {change * 100:+.2f} percentage points."
                    )
                    cites[n] = {y1: _url(years, y1), y2: _url(years, y2)}
    for note in unavailable:
        n = f"F{len(lines) + 1}"
        lines.append(f"[{n}] {note}")
        cites[n] = {}
    return lines, cites


def _url(years: list[dict], fy: int) -> str | None:
    return next((e.get("filing_url") for e in years if e["fiscal_year"] == fy), None)


WRITER_SYSTEM = (
    "You answer an analyst's question about US companies' 10-K filings, using ONLY the numbered "
    "excerpts [1], [2], ... (from the companies' Item 1A risk factors) and the numbered figures "
    "[F1], [F2], ... (from the SEC financial statement data). Rules:\n"
    "1. Every sentence ends with at least one citation, like [2] or [F1][3]: the excerpts or "
    "figures it is based on.\n"
    "2. State a figure only if it is in the figures list. You may round it (for example "
    "$86,377,000,000 as $86.4 billion, 12.29% as 12.3%) but never change it. Write no other "
    "number at all: no counts, no dates, no amounts from the excerpts.\n"
    "3. If a figure is listed as not available, say it is not available. Never estimate one.\n"
    "4. For a change between years, compare that company's excerpts for those years; an excerpt "
    "marked 'near-identical text in each' means the language did not change.\n"
    "5. No outside knowledge, advice or forecasts. At most 200 words, plain text, no headings."
)


def writer_messages(
    question: str, figures: list[str], excerpts: str, feedback: list[str] | None = None
) -> list[dict]:
    user = (
        f"Question: {question}\n\nFigures:\n"
        + ("\n".join(figures) or "(none)")
        + "\n\nExcerpts:\n"
        + (excerpts or "(none)")
    )
    conv = [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}]
    if feedback:
        conv.append(
            {
                "role": "user",
                "content": "Your previous answer was rejected:\n- "
                + "\n- ".join(feedback)
                + "\nWrite the whole answer again, following the "
                "rules.",
            }
        )
    return conv


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", " ".join(text.split()))
    return [p for p in parts if re.search(r"[A-Za-z]", p)]


def check_citations(text: str, excerpt_ns: set[int], figure_ns: set[str]) -> list[str]:
    problems = []
    for s in sentences(text):
        tokens = CITATION.findall(s)
        if not tokens:
            problems.append(f"a sentence has no citation: {s[:80]!r}")
        for t in tokens:
            ok = t in figure_ns if t.startswith("F") else int(t) in excerpt_ns
            if not ok:
                problems.append(f"citation [{t}] doesn't exist")
    return problems


# ---------------------------------------------------------------- the workflow


class Planned(Event):
    plan: dict


class Gathered(Event):
    pass


class Drafted(Event):
    text: str


class Rejected(Event):
    problems: list[str]


class CrewState(BaseModel):
    question: str = ""
    plan: dict = {}
    plan_source: str = ""
    plan_problems: list[str] = []
    figures: list[str] = []
    figure_cites: dict = {}
    excerpts: str = ""
    excerpt_cites: dict = {}
    context_stats: dict = {}
    allowed: list[dict] = []
    attempts: int = 0
    trace: list[dict] = []
    tool_errors: list[str] = []


@dataclass
class Crew:
    """What the workflow needs: the tools, the role backends, the budget and the token counter."""

    tools: Any
    planner: Chat
    writer: Chat
    budget: int
    count: Callable
    user: str = "fl-m6"
    calls: dict = field(default_factory=dict)


class AnalystWorkflow(Workflow):
    def __init__(self, crew: Crew, **kwargs) -> None:
        super().__init__(timeout=kwargs.pop("timeout", WORKFLOW_TIMEOUT_S), **kwargs)
        self.crew = crew

    async def _trace(self, ctx: Context[CrewState], step_name: str, started: float, **info):
        async with ctx.store.edit_state() as s:
            s.trace.append({"step": step_name, "s": round(time.perf_counter() - started, 3)} | info)

    @step
    async def plan(self, ctx: Context[CrewState], ev: StartEvent) -> Planned:
        started = time.perf_counter()
        question = ev.question
        reply = await self.crew.planner(
            planner_messages(question),
            plan_schema(),
            PLAN_MAX_TOKENS,
            route="crew:planner",
            user=self.crew.user,
        )

        def known(ticker: str) -> bool:
            try:
                self.crew.tools.company(ticker)
                return True
            except ToolError:
                return False

        plan, problems = None, []
        if reply.error:
            problems = [f"planner backend: {reply.error}"]
        else:
            try:
                plan, problems = validate_plan(json.loads(reply.text), known)
            except json.JSONDecodeError as error:
                problems = [f"planner JSON: {error}"]
        source = "model" if plan else "fallback"
        if plan is None:
            plan = fallback_plan(question)
        async with ctx.store.edit_state() as s:
            s.question, s.plan, s.plan_source, s.plan_problems = question, plan, source, problems
        await self._trace(ctx, "plan", started, source=source, backend=reply.backend)
        return Planned(plan=plan)

    @step
    async def retrieve(self, ctx: Context[CrewState], ev: Planned) -> Gathered:
        started = time.perf_counter()
        tools, plan = self.crew.tools, ev.plan
        hits, financials, errors = [], [], []
        for s in plan["searches"]:
            try:
                res = await tools.search_filings(
                    s["company"], s["query"][:500], [s["fiscal_year"]], SEARCH_K
                )
                hits += [h | {"ticker": res["company"]["ticker"]} for h in res["hits"]]
            except ToolError as error:
                errors.append(f"search {s['company']} FY{s['fiscal_year']}: {error}")
        years_ok = [y for y in plan["fiscal_years"] if FIRST_YEAR <= y <= LAST_YEAR]
        unavailable = [
            f"Fiscal year ended in {y}: outside the data ({FIRST_YEAR}-{LAST_YEAR});"
            " no figures or 10-K available."
            for y in plan["fiscal_years"]
            if y not in years_ok
        ]
        needs_figures = bool(plan["metrics"])
        for c in plan["companies"] if needs_figures and years_ok else []:
            try:
                financials.append(tools.get_financials(c, years_ok))
            except ToolError as error:
                errors.append(f"financials {c}: {error}")
        ctx_built = build_context(hits, self.crew.budget, self.crew.count)
        figures, figure_cites = figure_lines(financials, unavailable)
        async with ctx.store.edit_state() as st:
            st.figures, st.figure_cites = (
                figures,
                {k: {str(y): u for y, u in v.items()} for k, v in figure_cites.items()},
            )
            st.excerpts = ctx_built.text
            st.excerpt_cites = {
                str(n): {str(y): u for y, u in c.items()}
                for n, c in ctx_built.citation_map().items()
            }
            st.context_stats = ctx_built.stats
            st.allowed = [asdict(a) for a in allowed_values(financials)]
            st.tool_errors = errors
        await self._trace(
            ctx,
            "retrieve",
            started,
            hits=len(hits),
            financials=len(financials),
            excerpts=len(ctx_built.excerpts),
            tokens=ctx_built.tokens,
        )
        return Gathered()

    @step
    async def write(self, ctx: Context[CrewState], ev: Gathered | Rejected) -> Drafted | StopEvent:
        started = time.perf_counter()
        s = await ctx.store.get_state()
        feedback = ev.problems if isinstance(ev, Rejected) else None
        conv = writer_messages(s.question, s.figures, s.excerpts, feedback)
        reply = await self.crew.writer(
            conv, None, WRITE_MAX_TOKENS, route="crew:writer", user=self.crew.user
        )
        async with ctx.store.edit_state() as st:
            st.attempts += 1
        await self._trace(
            ctx,
            "write",
            started,
            backend=reply.backend,
            prompt_tokens=reply.prompt_tokens,
            error=reply.error,
        )
        if reply.error:
            return StopEvent(
                result=await self._result(ctx, "error", "", [f"writer: {reply.error}"])
            )
        return Drafted(text=reply.text.strip())

    @step
    async def check(self, ctx: Context[CrewState], ev: Drafted) -> Rejected | StopEvent:
        started = time.perf_counter()
        s = await ctx.store.get_state()
        allowed = [Allowed(**a) for a in s.allowed]
        numbers = check_answer(ev.text, allowed)
        cites = check_citations(ev.text, {int(n) for n in s.excerpt_cites}, set(s.figure_cites))
        problems = numbers.problems + cites
        await self._trace(
            ctx,
            "check",
            started,
            ok=not problems,
            problems=len(problems),
            figures=len(numbers.figures),
        )
        if not problems:
            return StopEvent(
                result=await self._result(ctx, "answered", ev.text, [], numbers.figures)
            )
        if s.attempts < MAX_ATTEMPTS:
            return Rejected(problems=problems)
        return StopEvent(
            result=await self._result(ctx, "blocked", ev.text, problems, numbers.figures)
        )

    async def _result(
        self, ctx, status: str, text: str, problems: list[str], figures: list[dict] | None = None
    ) -> dict:
        s = await ctx.store.get_state()
        used = {t for t in CITATION.findall(text)}
        return {
            "status": status,
            "answer": text if status == "answered" else None,
            "last_draft": text or None,
            "problems": problems,
            "citations": {
                t: (s.figure_cites if t.startswith("F") else s.excerpt_cites).get(t, {})
                for t in sorted(used)
            },
            "figures_checked": figures or [],
            "plan": s.plan,
            "plan_source": s.plan_source,
            "plan_problems": s.plan_problems,
            "context": s.context_stats,
            "figure_lines": s.figures,
            "excerpts": s.excerpts,
            "tool_errors": s.tool_errors,
            "attempts": s.attempts,
            "trace": s.trace,
        }


async def answer(crew: Crew, question: str) -> dict:
    """One question through the crew (call inside a running event loop)."""
    started = time.perf_counter()
    result = await AnalystWorkflow(crew).run(question=question)
    return result | {"seconds": round(time.perf_counter() - started, 2)}


# ---------------------------------------------------------------- running a split

GEMINI_MODEL = "gemini-3.1-flash-lite"  # FL-M5's hosted backend (AI Studio free tier)
GEMINI_RPM, GEMINI_RPD = 15, 500  # its free-tier limits (AI Studio rate-limit page, 26 Nov)
VLLM_URL, VLLM_MODEL = "http://127.0.0.1:58000", "gptq-w4a16"
GGUF_URL = "http://127.0.0.1:58081"
RERANK_URL = "http://127.0.0.1:58090"
ROUTER_INPUTS = Path("bench/router_inputs.json")
OUT_DIR = Path("data/interim/agents")
RULE = Path("bench/m6_crew_rule.json")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


def summarize_run(rows: list[dict], gemini_used: int, gemini_cap: int | None) -> dict:
    from collections import Counter

    secs = [r["seconds"] for r in rows]
    return {
        "questions": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "plan_source": dict(Counter(r["plan_source"] for r in rows)),
        "plan_valid_rate": round(sum(r["plan_source"] == "model" for r in rows) / len(rows), 4)
        if rows
        else None,
        "attempts": dict(Counter(str(r["attempts"]) for r in rows)),
        "gemini_calls": {"used": gemini_used, "cap": gemini_cap},
        "seconds_per_question": {"p50": percentile(secs, 0.5), "p95": percentile(secs, 0.95)},
        "context_tokens_mean": round(
            sum(r["context"].get("tokens_used", 0) for r in rows) / len(rows), 1
        )
        if rows
        else None,
        "tool_errors": sum(len(r["tool_errors"]) for r in rows),
    }


def build_crew(budget: int, max_gemini_calls: int | None, user: str) -> tuple[Crew, CallCap]:
    from filinglens.agents.context import student_counter
    from filinglens.agents.tools import local_tools
    from filinglens.serve.backends import GeminiBackend, OpenAICompatBackend
    from filinglens.serve.cost_meter import MeteredBackend, PostgresSink, Prices

    routes = route_table(json.loads(ROUTER_RESULTS.read_text()))
    prices = Prices.from_inputs(json.loads(ROUTER_INPUTS.read_text()))
    sink = PostgresSink()
    inner = {
        "vllm": OpenAICompatBackend("vllm", VLLM_URL, VLLM_MODEL),
        "gguf": OpenAICompatBackend("gguf", GGUF_URL, "gguf"),
        "gemini": GeminiBackend.from_env(GEMINI_MODEL, GEMINI_RPM, GEMINI_RPD),
    }
    metered = {name: MeteredBackend(b, prices, [sink]) for name, b in inner.items()}
    cap = CallCap(metered["gemini"].chat, max_gemini_calls, "gemini")
    chats = {"vllm": metered["vllm"].chat, "gguf": metered["gguf"].chat, "gemini": cap}
    crew = Crew(
        tools=local_tools(RERANK_URL),
        planner=chats[routes[ROLES["planner"]]],
        writer=chats[routes[ROLES["writer"]]],
        budget=budget,
        count=student_counter(),
        user=user,
        calls={role: routes[cls] for role, cls in ROLES.items()},
    )
    return crew, cap


async def run_split(args) -> dict:
    from filinglens.agents.questions import load_questions

    if args.split == "test":
        if not args.allow_test:
            raise SystemExit("the test questions are scored once: --allow-test")
        if not RULE.exists():
            raise SystemExit(f"{RULE} must be committed before the test run")
    questions = load_questions(args.split, allow_test=args.allow_test)
    if args.qid:
        questions = [q for q in questions if q["qid"] in set(args.qid)]
    if args.limit:
        questions = questions[: args.limit]
    crew, cap = build_crew(args.budget, args.max_gemini_calls, f"fl-m6-{args.split}-{args.tag}")
    rows = []
    for q in questions:
        result = await answer(crew, q["question"])
        rows.append({"qid": q["qid"], "template": q["template"], **result})
        print(
            f"{q['qid']} {result['status']} attempts={result['attempts']}"
            f" plan={result['plan_source']} {result['seconds']} s"
            f" gemini={cap.used}",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"answers_{args.split}_{args.tag}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {
        "routes": crew.calls,
        "budget": args.budget,
        "answers": str(out),
        **summarize_run(rows, cap.used, args.max_gemini_calls),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import asyncio
    import sys
    from datetime import UTC, datetime

    parser = argparse.ArgumentParser(description="FL-M6 analyst crew")
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test"], required=True)
    r.add_argument("--allow-test", action="store_true")
    r.add_argument("--budget", type=int, required=True, help="context tokens (student tokenizer)")
    r.add_argument("--max-gemini-calls", type=int, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--qid", nargs="*", default=None)
    r.add_argument("--limit", type=int, default=None)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    record = {
        "command": " ".join(
            ["uv run --group api --group agents python -m filinglens.agents.crew", *argv]
        ),
        "started_utc": started,
        **asyncio.run(run_split(args)),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = Path(f"bench/agent_run_{args.split}_{args.tag}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
