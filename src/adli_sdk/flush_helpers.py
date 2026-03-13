"""Shared helpers for assembling and sending LearnRequest to /learn."""
from __future__ import annotations

from adli_sdk.models import AgentTrace, LearnRequest, Message, MessagePart


def count_trace_stats(messages: list[Message]) -> tuple[int, int]:
    """Return ``(steps_count, tool_calls_count)`` derived from *messages*."""
    steps = sum(1 for m in messages if m.kind == "response")
    tools = sum(
        1 for m in messages for p in m.parts if p.part_kind == "tool-call"
    )
    return steps, tools


def build_learn_request(
    *,
    agent_name: str,
    project_id: int,
    framework: str,
    adli_trace_id: str,
    user_message: str,
    outcome: str,
    trace: AgentTrace,
) -> LearnRequest:
    """Build a :class:`LearnRequest` with computed stats from *trace*."""
    steps, tools = count_trace_stats(trace.messages)
    trace.usage.tool_calls = tools
    return LearnRequest(
        agent_name=agent_name,
        project_id=project_id,
        framework=framework,
        adli_trace_id=adli_trace_id,
        user_message=user_message,
        injected=bool(adli_trace_id),
        outcome=outcome,
        steps_count=steps,
        cost_usd=None,
        trace=trace,
    )


def interleave_tool_pairs(messages: list[Message]) -> list[Message]:
    """Re-order messages so each tool-call is immediately followed by its return.

    The frontend links call/return positionally (first TOOL-CALL pairs with
    next TOOL-RETURN).  When multiple tool-calls are batched in a single
    response message, this breaks.

    This function splits batch messages into individual call/return pairs
    matched by ``tool_call_id``, producing: TC1, TR1, TC2, TR2, ...

    Handles both formats:
    - PydanticAI: returns batched in one request message with N tool-return parts
    - LangChain: N separate request messages each with 1 tool-return part
    """
    result: list[Message] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        call_parts = [p for p in msg.parts if p.part_kind == "tool-call"]
        other_parts = [p for p in msg.parts if p.part_kind != "tool-call"]

        if len(call_parts) <= 1:
            result.append(msg)
            i += 1
            continue

        # Emit any non-tool-call parts (e.g. text) first
        if other_parts:
            result.append(
                Message(kind=msg.kind, parts=other_parts, timestamp=msg.timestamp)
            )

        # Collect tool-return parts from subsequent messages
        return_parts: list[MessagePart] = []
        j = i + 1
        while j < len(messages):
            candidate = messages[j]
            ret = [p for p in candidate.parts if p.part_kind == "tool-return"]
            non_ret = [p for p in candidate.parts if p.part_kind != "tool-return"]
            if not ret:
                break
            return_parts.extend(ret)
            # Preserve non-return parts from return messages (rare but safe)
            if non_ret:
                result.append(
                    Message(
                        kind=candidate.kind,
                        parts=non_ret,
                        timestamp=candidate.timestamp,
                    )
                )
            j += 1

        # Build return lookup by tool_call_id
        return_by_id: dict[str, MessagePart] = {}
        positional_returns: list[MessagePart] = []
        for rp in return_parts:
            tcid = getattr(rp, "tool_call_id", None)
            if tcid:
                return_by_id[tcid] = rp
            else:
                positional_returns.append(rp)

        # Emit interleaved pairs
        for cp in call_parts:
            result.append(
                Message(kind=msg.kind, parts=[cp], timestamp=msg.timestamp)
            )
            tcid = getattr(cp, "tool_call_id", None)
            matched: MessagePart | None = None
            if tcid and tcid in return_by_id:
                matched = return_by_id.pop(tcid)
            elif positional_returns:
                matched = positional_returns.pop(0)
            if matched:
                result.append(Message(kind="request", parts=[matched]))

        # Emit any unmatched returns
        leftover = list(return_by_id.values()) + positional_returns
        for lp in leftover:
            result.append(Message(kind="request", parts=[lp]))

        i = j  # skip past consumed messages

    return result
