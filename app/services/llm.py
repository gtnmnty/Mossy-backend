# The one place that talks to Claude. Both routes in chat.py call into this file.
from app.core.config import settings
from app.services.history import trim_by_tokens
from pathlib import Path
import json
import logging
from app.tools import TOOLS, REGISTRY

# Feature 2: System prompt and persona
# Mossy's personality and rules live in prompts/system.md, so you can edit them without touching code.
# The file is read once when the server starts, not on every request.
SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")

log = logging.getLogger("mossy")
MAX_STEPS = 6   # Feature 5: stops a runaway tool loop after 6 model turns


# Feature 7: "Think harder" toggle (extended thinking)
# Current models (claude-sonnet-5-5 and newer) use ADAPTIVE thinking: you ask for an
# effort level ("high"), not a fixed token budget. The old {"type": "enabled",
# "budget_tokens": N} form is for older models and returns a 400 error on these.
# display="summarized" is set on purpose: newer models default to "omitted", which gives
# back empty thinking blocks, so the UI would have nothing to show.
def thinking_params(think: bool) -> dict:
    if not think:
        return {"max_tokens": settings.max_tokens}
    return {
        "max_tokens": settings.think_max_tokens,     # thinking tokens count against this
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    }


# Feature 1 + Feature 4 + Feature 7: Multi-turn history, Prompt caching, Thinking
# Builds the settings every Claude call shares.
# Flow: trim the chat history to fit the token budget -> add the model and system prompt ->
# add the thinking settings (or just max_tokens, if thinking is off) -> switch on caching
# so repeated parts of the prompt are cheaper and faster.
async def build_request(client, messages, system=SYSTEM_PROMPT, think: bool = False) -> dict:
    return {
        "model": settings.model,
        "system": system,
        "messages": await trim_by_tokens(client, messages, system),
        # Automatic caching: the API puts the cache marker on the last block it can cache
        # and moves it forward as the chat grows. Earlier turns are then read from cache.
        # Note: turning thinking on or off changes this request's shape, so the very next
        # call after flipping the toggle is a cache miss. That is expected, not a bug.
        "cache_control": {"type": "ephemeral"},
        **thinking_params(think),
    }


# Non-streaming call used by the /chat fallback route. Waits for the whole answer at once.
async def reply(client, messages, system=SYSTEM_PROMPT, think: bool = False):
    return await client.messages.create(**await build_request(client, messages, system, think))


# Feature 5: Tools
# Runs one tool that Claude asked for and wraps the output as a "tool_result".
# Flow: find the Python function by tool name -> call it with Claude's inputs ->
# return the result. Errors are sent back to Claude (is_error) instead of crashing,
# so Mossy can explain what went wrong.
async def run_tool(block) -> dict:
    fn = REGISTRY.get(block.name)

    if fn is None:
        return {"type": "tool_result", "tool_use_id": block.id,
                "content": f"Unknown tool: {block.name}", "is_error": True}

    try:
        out = await fn(**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": out}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": block.id,
                "content": f"Error: {e}", "is_error": True}


# Feature 3 + Feature 5 + Feature 7: Streaming with a tool loop and thinking
# Streams Claude's reply and runs tools until Claude has a final answer.
# Flow for each step:
#   1. Stream Claude's thinking (if "think" is on) as "thinking" events, then its answer
#      text as "delta" events.
#   2. If Claude stopped normally, send "done" with token usage and finish.
#   3. If Claude stopped to use tools, run each one, send a "tool" event so the UI can
#      show it, add the results to the chat, and ask Claude again.
# Every tool_use block must get a matching tool_result in the next message, or the API errors.
# When thinking is on, Claude's turn (final.content) carries its thinking blocks too; they
# are appended to convo unmodified below, because the API requires them to come back exactly
# as given whenever a tool result follows a thinking turn.
async def run_with_tools(client, messages, think: bool = False):
    kwargs = await build_request(client, messages, think=think)   # trims history once, up front
    kwargs["tools"] = TOOLS
    convo = kwargs["messages"]                       # same list object: appending updates kwargs

    for _ in range(MAX_STEPS):
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue
                if event.delta.type == "thinking_delta":
                    yield {"event": "thinking", "data": json.dumps({"text": event.delta.thinking})}
                elif event.delta.type == "text_delta":
                    yield {"event": "delta", "data": json.dumps({"text": event.delta.text})}
            final = await stream.get_final_message()

        # Feature 4: these numbers show whether caching works (cache_read > 0 on later turns).
        u = final.usage
        log.info("stop=%s in=%s out=%s cache_write=%s cache_read=%s", final.stop_reason,
                 u.input_tokens, u.output_tokens,
                 u.cache_creation_input_tokens, u.cache_read_input_tokens)

        if final.stop_reason != "tool_use":
            yield {"event": "done", "data": json.dumps({"usage": u.model_dump()})}
            return

        # Pass Claude's turn back exactly as received, then answer every tool_use block.
        convo.append({"role": "assistant", "content": final.content})
        results = []
        for block in final.content:
            if block.type == "tool_use":
                log.info("tool call: %s %s", block.name, block.input)
                yield {"event": "tool", "data": json.dumps({"name": block.name})}
                results.append(await run_tool(block))
        convo.append({"role": "user", "content": results})

    yield {"event": "error", "data": json.dumps({"message": "Too many tool steps"})}
