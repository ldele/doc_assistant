"""Chainlit web UI — thin renderer over ``doc_assistant.chat_controller``.

All turn orchestration lives in ``src/doc_assistant/chat_controller.py`` (PR-M0). This
file only maps the controller's ``TurnEvent`` stream + ``TurnResult`` value object onto
Chainlit widgets:

1. Chainlit lifecycle hooks (``@cl.on_chat_start``, ``@cl.on_message``).
2. Streaming token delivery + step status.
3. Final message assembly (answer + pre-rendered blocks), source/figure elements,
   per-claim + export action buttons.

No business logic here — no provenance/claim/citation computation. One ``Session`` is
held in ``cl.user_session``; one ``ChatController`` is built at module load.
"""

import chainlit as cl

from doc_assistant.chat_controller import (
    ChatController,
    Result,
    Session,
    Step,
    Token,
    TurnResult,
)
from doc_assistant.config import LLM_MODEL, LLM_PROVIDER
from doc_assistant.embeddings import get_active_model_name

controller = ChatController()


# ============================================================
# Lifecycle
# ============================================================


def _session() -> Session:
    s = cl.user_session.get("session")
    if not isinstance(s, Session):
        s = Session()
        cl.user_session.set("session", s)
    return s


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("session", Session())
    await cl.Message(
        content=(
            f"📚 **Document assistant ready.** {controller.chunk_count()} chunks indexed.  \n"
            f"🤖 Generation model: `{LLM_PROVIDER}/{LLM_MODEL}` · "
            f"🧬 Embeddings: `{get_active_model_name()}`  \n"
            "_Tip: `/export` downloads this conversation; `/export-debug` writes a dev "
            "bundle (sources + scores + figures + log) to `data/exports/`._"
        ),
    ).send()


# ============================================================
# Message handler — consume the TurnEvent stream
# ============================================================


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session = _session()
    msg: cl.Message | None = None  # the streaming message, lazily created on first Token
    streamed = False
    for event in controller.handle_message(session, message.content):
        if isinstance(event, Token):
            if msg is None:
                msg = cl.Message(content="")
                await msg.send()
            streamed = True
            await msg.stream_token(event.text)
        elif isinstance(event, Step):
            async with cl.Step(name=event.name, type="tool") as step:
                step.output = event.status
        elif isinstance(event, Result):
            await _render_result(event.result, msg, streamed=streamed)


async def _render_result(result: TurnResult, msg: cl.Message | None, *, streamed: bool) -> None:
    """Map a finished TurnResult onto a Chainlit message (+ source/figure elements +
    action buttons). The export action rides only on real answers (a streamed AI turn
    or a human-mode turn) — command/library/edit responses stay plain, as before."""
    elements: list[cl.Text | cl.Image | cl.File] = []
    for s in result.sources:
        elements.append(
            cl.Text(name=f"[{s.n}]", content=f"**{s.citation}**\n\n{s.excerpt}", display="side")
        )
        if s.figure_path:
            elements.append(
                cl.Image(
                    name=f"[{s.n}] figure", path=s.figure_path, display="inline", size="medium"
                )
            )
    if result.download_path is not None:
        elements.append(
            cl.File(
                name=result.download_path.name, path=str(result.download_path), display="inline"
            )
        )

    actions: list[cl.Action] = []
    for c in result.flagged_claims:
        actions.append(
            cl.Action(name="claim_accept", payload={"id": c.claim_id, "n": c.n}, label=f"✓ #{c.n}")
        )
        actions.append(
            cl.Action(name="claim_reject", payload={"id": c.claim_id, "n": c.n}, label=f"✗ #{c.n}")
        )
        actions.append(
            cl.Action(name="claim_edit", payload={"id": c.claim_id, "n": c.n}, label=f"✎ #{c.n}")
        )
    if streamed or result.mode == "human":
        actions.append(_export_action())

    content = (
        result.answer
        + result.sources_md
        + result.usage_md
        + result.provenance_card_md
        + result.claim_review_md
        + result.citation_note_md
    )

    if msg is None:
        await cl.Message(content=content, elements=elements, actions=actions).send()
    else:
        msg.content = content
        msg.elements = elements
        msg.actions = actions
        await msg.update()


# ============================================================
# Action callbacks — map to controller methods
# ============================================================


async def _resolve_claim(action: cl.Action, decision: str) -> None:
    p = action.payload
    try:
        controller.adjudicate(str(p["id"]), decision)
        mark = "✓ accepted" if decision == "accepted" else "✗ rejected"
        await cl.Message(content=f"Claim #{p['n']} {mark}.").send()
    except Exception as e:
        await cl.Message(content=f"⚠ Could not record claim #{p['n']}: {e}").send()


@cl.action_callback("claim_accept")
async def _on_claim_accept(action: cl.Action) -> None:
    await _resolve_claim(action, "accepted")


@cl.action_callback("claim_reject")
async def _on_claim_reject(action: cl.Action) -> None:
    await _resolve_claim(action, "rejected")


@cl.action_callback("claim_edit")
async def _on_claim_edit(action: cl.Action) -> None:
    p = action.payload
    _session().awaiting_edit = {"id": str(p["id"]), "n": p["n"]}
    await cl.Message(content=f"✏️ Send the corrected text for claim #{p['n']}.").send()


def _export_action() -> cl.Action:
    return cl.Action(
        name="export_conversation",
        payload={"dev": False},
        label="⬇ Export chat",
        tooltip="Download this conversation as markdown",
    )


@cl.action_callback("export_conversation")
async def _on_export(action: cl.Action) -> None:
    dev = bool(action.payload.get("dev", False))
    msg, path = controller.export_conversation(_session(), dev=dev)
    elements = (
        [cl.File(name=path.name, path=str(path), display="inline")] if path is not None else []
    )
    await cl.Message(content=msg, elements=elements).send()
