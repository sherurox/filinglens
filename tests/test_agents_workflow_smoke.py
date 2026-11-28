"""Pins the LlamaIndex Workflows API the FL-M6 crew relies on (T1 smoke test; no model, no network).

The shape is the crew's gate in miniature: a writer step drafts, a checker step either stops the run
or sends one `Failed` event back; after one revision a still-bad draft stops the run as blocked.
It proves the pinned version supports what T7 needs: custom events, a step that accepts two event
types, a step that returns one of two, typed run state in `ctx.store`, and an explicit run timeout.
`Workflow.run()` schedules its task at call time, so it must be called inside a running event loop
(found in T1: calling it outside one raises "no running event loop"); `finish()` does that.
"""

import asyncio

import pytest

pytest.importorskip("workflows")  # the `agents` group: uv run --group agents pytest

from pydantic import BaseModel  # noqa: E402
from workflows import Context, Workflow, step  # noqa: E402
from workflows.errors import WorkflowTimeoutError  # noqa: E402
from workflows.events import Event, StartEvent, StopEvent  # noqa: E402

MAX_ATTEMPTS = 2  # the first draft plus one revision, as in the crew


class Draft(Event):
    text: str


class Failed(Event):
    reason: str


class GateState(BaseModel):
    drafts: list[str] = []
    attempts: int = 0


def cited(text: str) -> bool:
    return "[1]" in text


class Gate(Workflow):
    @step
    async def write(self, ctx: Context[GateState], ev: StartEvent | Failed) -> Draft:
        async with ctx.store.edit_state() as state:
            if isinstance(ev, StartEvent):
                state.drafts = list(ev.drafts)
            state.attempts += 1
            text = state.drafts[state.attempts - 1]
        return Draft(text=text)

    @step
    async def check(self, ctx: Context[GateState], ev: Draft) -> StopEvent | Failed:
        attempts = (await ctx.store.get_state()).attempts
        if cited(ev.text):
            return StopEvent(result={"status": "ok", "text": ev.text, "attempts": attempts})
        if attempts < MAX_ATTEMPTS:
            return Failed(reason="no citation")
        return StopEvent(result={"status": "blocked", "text": None, "attempts": attempts})


class Slow(Workflow):
    @step
    async def wait(self, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(1.0)
        return StopEvent(result="late")


def finish(wf: Workflow, **start):
    """Run a workflow to its StopEvent from synchronous test code."""

    async def go():
        return await wf.run(**start)

    return asyncio.run(go())


def run_gate(drafts: list[str]) -> dict:
    return finish(Gate(timeout=10), drafts=drafts)


def test_first_draft_passes():
    assert run_gate(["fuel is volatile [1]"]) == {
        "status": "ok",
        "text": "fuel is volatile [1]",
        "attempts": 1,
    }


def test_one_revision_then_passes():
    out = run_gate(["no source", "fuel is volatile [1]"])
    assert out["status"] == "ok" and out["attempts"] == 2


def test_blocked_after_one_revision():
    out = run_gate(["no source", "still none", "never reached [1]"])
    assert out == {"status": "blocked", "text": None, "attempts": 2}


def test_each_run_starts_with_fresh_state():
    wf = Gate(timeout=10)
    first = finish(wf, drafts=["a [1]"])
    second = finish(wf, drafts=["b [1]"])
    assert first["attempts"] == second["attempts"] == 1


def test_explicit_timeout_raises():
    with pytest.raises(WorkflowTimeoutError):
        finish(Slow(timeout=0.05))
