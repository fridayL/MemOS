"""
ToolAgent MemReader - ReAct Tool Calling Memory Extraction

Design:
- Inherits MultiModalStructMemReader, only overrides _process_string_fine
- Chat-type string_fine extraction is handled by an external 4B model via HTTP tool calling
- All other logic (doc processing, skill/tool trajectory/preference extraction, embedding)
  is delegated to the parent MultiModalStructMemReader
- Per-user buffer is maintained in-memory within the MemReader (across conversation turns)
- search_memory invokes MemOS's own searcher
"""

import contextlib
import json
import threading
import time

from typing import TYPE_CHECKING, Any

import requests

from memos import log
from memos.mem_reader.multi_modal_struct import MultiModalStructMemReader
from memos.memories.textual.item import TextualMemoryItem


if TYPE_CHECKING:
    from memos.configs.mem_reader import ToolAgentMemReaderConfig


logger = log.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool schema definitions, sent to the 4B model
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search existing long-term memories for relevant context before deciding "
                "what to store. Use this when you need to check if similar information "
                "already exists or to get background context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_memory",
            "description": (
                "Extract and store important facts, preferences, or events from the conversation. "
                "Call this when the conversation contains clear, valuable information worth remembering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_list": {
                        "type": "array",
                        "description": "List of memory items to store",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "Short title or keyword for the memory",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The actual memory content (fact, preference, event)",
                                },
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["LongTermMemory", "UserMemory"],
                                    "description": (
                                        "LongTermMemory for facts/events/knowledge, "
                                        "UserMemory for personal attributes/preferences"
                                    ),
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Relevant topic tags",
                                },
                            },
                            "required": ["key", "value", "memory_type"],
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of this conversation for background context",
                    },
                },
                "required": ["memory_list", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buffer_memory",
            "description": (
                "Buffer this conversation turn for future processing. Use when the conversation "
                "is still ongoing and more context is needed before making extraction decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why buffering instead of extracting now",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ignore_memory",
            "description": (
                "Ignore this conversation — nothing worth storing. Use for small talk, "
                "greetings, or content already captured in memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why ignoring",
                    }
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _build_system_prompt(buffer_summary: str, session_time: str) -> str:
    parts = [
        "You are a memory extraction agent. Your job is to analyze conversations and decide "
        "what information is worth storing in long-term memory.",
        "",
        "Available actions (call exactly one per turn):",
        "- search_memory: Search existing memories for context (use before add_memory if relevant)",
        "- add_memory: Extract and store valuable facts, preferences, or events",
        "- buffer_memory: Accumulate this turn, wait for more context",
        "- ignore_memory: Nothing worth storing (small talk, already known, repetitive)",
        "",
        "Guidelines:",
        "- Store specific, verifiable facts (names, preferences, events, decisions)",
        "- Do NOT store generic greetings, chitchat, or vague statements",
        "- UserMemory: personal attributes or preferences about the user",
        "- LongTermMemory: facts, events, shared knowledge from the conversation",
        "- If unsure whether info already exists, call search_memory first",
    ]
    if buffer_summary:
        parts += ["", f"Previously buffered context:\n{buffer_summary}"]
    if session_time:
        parts += ["", f"Session time: {session_time}"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ToolAgentMemReader
# ---------------------------------------------------------------------------


class ToolAgentMemReader(MultiModalStructMemReader):
    """
    Inherits MultiModalStructMemReader, only overrides _process_string_fine.

    Data flow (chat fine mode):
      MultiModalStructMemReader._process_multi_modal_data()
        ├── _process_string_fine()           ← overridden here: 4B ReAct extraction
        ├── _process_tool_trajectory_fine()   ← inherited, unchanged
        ├── process_skill_memory_fine()       ← inherited, unchanged
        └── process_preference_fine()         ← inherited, unchanged
      Embeddings are computed via _make_memory_item(need_embed=True) inside _process_string_fine.
    """

    def __init__(self, config: "ToolAgentMemReaderConfig"):
        super().__init__(config)
        # Per-user conversation buffer: (user_id, user_name) -> list[dict]
        self._buffer: dict[tuple[str, str], list[dict]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Override _process_string_fine: replace LLM string extraction with 4B ReAct
    # ------------------------------------------------------------------

    def _process_string_fine(
        self,
        fast_memory_items: list[TextualMemoryItem],
        info: dict[str, Any],
        custom_tags: list[str] | None = None,
        **kwargs,
    ) -> list[TextualMemoryItem]:
        """
        Override parent method: use 4B ToolAgent ReAct loop for chat memory extraction.

        Reconstructs original conversation messages from fast_memory_items sources,
        invokes _react_loop, and converts decisions into TextualMemoryItems (with embeddings).
        """
        if not fast_memory_items:
            logger.info("[ToolAgentMemReader] _process_string_fine: empty fast_memory_items, skip")
            return []

        user_id = info.get("user_id", "")
        session_id = info.get("session_id", "")
        session_time = info.get("session_time", "")
        user_name = kwargs.get("user_name")
        logger.info(
            f"[ToolAgentMemReader] _process_string_fine called: "
            f"user={user_id} session={session_id} user_name={user_name!r} "
            f"fast_items={len(fast_memory_items)}"
        )

        # Reconstruct original conversation messages from fast_memory_items sources (deduplicated)
        seen: set[tuple] = set()
        all_msgs: list[dict] = []
        for item in fast_memory_items:
            for src in item.metadata.sources or []:
                if isinstance(src, dict):
                    role = str(src.get("role") or "user")
                    content = str(src.get("content") or "")
                elif hasattr(src, "role"):
                    role = str(src.role or "user")
                    content = str(src.content or "")
                else:
                    continue
                key = (role, content)
                if content and key not in seen:
                    seen.add(key)
                    all_msgs.append({"role": role, "content": content})

        logger.info(f"[ToolAgentMemReader] reconstructed {len(all_msgs)} messages from sources")

        if not all_msgs:
            logger.info(
                "[ToolAgentMemReader] No messages reconstructed from sources, "
                "falling back to parent _process_string_fine."
            )
            return super()._process_string_fine(fast_memory_items, info, custom_tags, **kwargs)

        # Invoke the ReAct loop, returns a list of raw memory dicts
        logger.info(
            f"[ToolAgentMemReader] entering _react_loop: user={user_id} "
            f"session={session_id} msgs_count={len(all_msgs)}"
        )
        raw_memories = self._react_loop(
            user_id=user_id,
            session_id=session_id,
            session_time=session_time,
            turn_msgs=all_msgs,
            info=info,
            user_name=user_name,
        )

        if raw_memories is None:
            logger.warning(
                f"[ToolAgentMemReader] _react_loop exceeded max rounds for user={user_id}, "
                f"falling back to parent _process_string_fine."
            )
            return super()._process_string_fine(fast_memory_items, info, custom_tags, **kwargs)

        logger.info(f"[ToolAgentMemReader] _react_loop returned {len(raw_memories)} raw memories")

        if not raw_memories:
            return []

        # Build sources list for provenance tracking
        sources = [
            {"type": "chat", "role": m["role"], "content": m["content"][:200]} for m in all_msgs
        ]

        # Convert to TextualMemoryItem; embeddings are computed by _make_memory_item
        result: list[TextualMemoryItem] = []
        for mem in raw_memories:
            value = (mem.get("value") or "").strip()
            if not value:
                continue
            try:
                logger.info(
                    f"[ToolAgentMemReader] _make_memory_item: key={mem.get('key')!r} "
                    f"type={mem.get('memory_type')!r} value_len={len(value)}"
                )
                node = self._make_memory_item(
                    value=value,
                    info=info,
                    memory_type=mem.get("memory_type") or "LongTermMemory",
                    tags=mem.get("tags") or [],
                    key=mem.get("key") or "",
                    sources=sources,
                    background=mem.get("background") or "",
                    need_embed=True,
                )
                result.append(node)
                logger.info(f"[ToolAgentMemReader] _make_memory_item done: key={mem.get('key')!r}")
            except Exception as e:
                logger.info(f"[ToolAgentMemReader] Error making memory item: {e}")

        logger.info(
            f"[ToolAgentMemReader] _process_string_fine: {len(result)} items for user={user_id}"
        )
        return result

    # ------------------------------------------------------------------
    # Buffer helpers (thread-safe, isolated by user_id + user_name)
    # ------------------------------------------------------------------

    @staticmethod
    def _buf_key(user_id: str, user_name: str | None) -> tuple[str, str]:
        return (user_id, user_name or "")

    def _get_buffer(self, user_id: str, user_name: str | None = None) -> list[dict]:
        with self._lock:
            return list(self._buffer.get(self._buf_key(user_id, user_name), []))

    def _set_buffer(self, user_id: str, turns: list[dict], user_name: str | None = None) -> None:
        with self._lock:
            self._buffer[self._buf_key(user_id, user_name)] = turns

    def _clear_buffer(self, user_id: str, user_name: str | None = None) -> None:
        with self._lock:
            self._buffer.pop(self._buf_key(user_id, user_name), None)

    # ------------------------------------------------------------------
    # ReAct main loop
    # ------------------------------------------------------------------

    def _react_loop(
        self,
        user_id: str,
        session_id: str,
        session_time: str,
        turn_msgs: list[dict],
        info: dict[str, Any] | None = None,
        user_name: str | None = None,
    ) -> list[dict] | None:
        """
        Execute the ReAct tool-calling loop for a single batch of messages.

        Returns:
            list of raw memory dicts: [{value, memory_type, tags, key, background}, ...]
            buffer_memory / ignore_memory -> returns empty list
            None -> max rounds exceeded, caller should fallback to parent extraction
        """
        logger.info(
            f"[ToolAgentMemReader] _react_loop start: user={user_id} "
            f"session={session_id} user_name={user_name!r} "
            f"turn_msgs={len(turn_msgs)} max_rounds={self.config.max_rounds}"
        )
        buf = self._get_buffer(user_id, user_name)
        logger.info(
            f"[ToolAgentMemReader] buffer size for user={user_id} "
            f"user_name={user_name!r}: {len(buf)}"
        )

        buffer_summary = ""
        if buf:
            buffer_summary = "\n".join(
                f"[{m.get('role', '')}]: {str(m.get('content', ''))[:120]}" for m in buf[-6:]
            )

        system_prompt = _build_system_prompt(buffer_summary, session_time)

        convo_text = "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in turn_msgs
        )
        user_content = (
            "Please analyze the following conversation and decide what to store:\n\n" + convo_text
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        for _round in range(self.config.max_rounds):
            logger.info(
                f"[ToolAgentMemReader] round={_round} calling API: "
                f"model={self.config.model} url={self.config.api_url} "
                f"messages_count={len(messages)}"
            )
            response = self._call_api(messages)
            choices = response.get("choices", [])
            if not choices:
                logger.info("[ToolAgentMemReader] Empty choices from API, ignoring.")
                return []

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            logger.info(
                f"[ToolAgentMemReader] round={_round} API response: "
                f"finish_reason={choice.get('finish_reason')} "
                f"tool_calls_count={len(tool_calls)} content_len={len(content)}"
            )

            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                logger.info(
                    f"[ToolAgentMemReader] No tool call "
                    f"(finish_reason={choice.get('finish_reason')}), ignoring."
                )
                return []

            # Only process the first tool_call per round, discard the rest
            first_call = tool_calls[0]
            if len(tool_calls) > 1:
                logger.info(
                    f"[ToolAgentMemReader] round={_round} got {len(tool_calls)} "
                    f"tool_calls, only processing the first one, discarding rest"
                )
            tool_id = first_call.get("id", "")
            function = first_call.get("function", {})
            tool_name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            logger.info(
                f"[ToolAgentMemReader] round={_round} user={user_id} "
                f"tool={tool_name} args={json.dumps(arguments, ensure_ascii=False)[:500]}"
            )

            # --- search_memory: retrieve results, then proceed to next round ---
            if tool_name == "search_memory":
                query = arguments.get("query", "")
                search_results = self._search_by_searcher(
                    user_id, query, info=info, user_name=user_name
                )
                result_text = (
                    "\n".join(search_results) if search_results else "No relevant memories found."
                )
                logger.info(
                    f"[ToolAgentMemReader] search_memory query={query!r} "
                    f"hits={len(search_results)} results={result_text[:300]}"
                )
                # Keep only the first tool_call in the assistant message
                assistant_msg["tool_calls"] = [first_call]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_text,
                    }
                )
                logger.info(
                    f"[ToolAgentMemReader] round={_round} search completed, "
                    f"entering next round for model to review results"
                )
                continue  # proceed to next ReAct round

            # --- add_memory: return extracted results directly ---
            elif tool_name == "add_memory":
                memory_list = arguments.get("memory_list", [])
                summary = arguments.get("summary", "")
                logger.info(
                    f"[ToolAgentMemReader] add_memory: user={user_id} "
                    f"memory_count={len(memory_list)} summary={summary[:200]!r}"
                )
                self._clear_buffer(user_id, user_name)
                return [
                    {
                        "value": (m.get("value") or "").strip(),
                        "memory_type": m.get("memory_type") or "LongTermMemory",
                        "tags": m.get("tags") or [],
                        "key": m.get("key") or "",
                        "background": summary,
                    }
                    for m in memory_list
                    if (m.get("value") or "").strip()
                ]

            # --- buffer_memory: buffer current turn, attempt extraction from accumulated buffer ---
            elif tool_name == "buffer_memory":
                reason = arguments.get("reason", "")
                new_buf = buf + turn_msgs
                self._set_buffer(user_id, new_buf, user_name)
                logger.info(
                    f"[ToolAgentMemReader] buffer_memory for user={user_id}, "
                    f"reason={reason!r} buffer_size={len(new_buf)}"
                )
                # If buffer has accumulated enough messages, attempt model-based extraction
                if len(new_buf) >= 4:
                    logger.info(
                        f"[ToolAgentMemReader] buffer large enough ({len(new_buf)} msgs), "
                        f"attempting extraction from buffered content"
                    )
                    buf_convo = "\n".join(
                        f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in new_buf
                    )
                    extract_messages: list[dict] = [
                        {"role": "system", "content": _build_system_prompt("", session_time)},
                        {
                            "role": "user",
                            "content": (
                                "The following conversation has been buffered across multiple turns. "
                                "Please review and extract any valuable information now:\n\n"
                                + buf_convo
                            ),
                        },
                    ]
                    try:
                        extract_resp = self._call_api(extract_messages)
                        extract_choices = extract_resp.get("choices", [])
                        if extract_choices:
                            extract_tc = (
                                extract_choices[0].get("message", {}).get("tool_calls") or []
                            )
                            if extract_tc:
                                tc = extract_tc[0]
                                tc_name = tc.get("function", {}).get("name", "")
                                try:
                                    tc_args = json.loads(
                                        tc.get("function", {}).get("arguments", "{}")
                                    )
                                except json.JSONDecodeError:
                                    tc_args = {}
                                if tc_name == "add_memory":
                                    mem_list = tc_args.get("memory_list", [])
                                    summary = tc_args.get("summary", "")
                                    logger.info(
                                        f"[ToolAgentMemReader] buffer extraction succeeded: "
                                        f"extracted {len(mem_list)} memories from buffer"
                                    )
                                    self._clear_buffer(user_id, user_name)
                                    return [
                                        {
                                            "value": (m.get("value") or "").strip(),
                                            "memory_type": m.get("memory_type") or "LongTermMemory",
                                            "tags": m.get("tags") or [],
                                            "key": m.get("key") or "",
                                            "background": summary,
                                        }
                                        for m in mem_list
                                        if (m.get("value") or "").strip()
                                    ]
                                else:
                                    logger.info(
                                        f"[ToolAgentMemReader] buffer extraction: "
                                        f"model chose {tc_name}, no extraction"
                                    )
                    except Exception as e:
                        logger.info(f"[ToolAgentMemReader] buffer extraction failed: {e}")
                return []

            # --- ignore_memory: return empty directly ---
            elif tool_name == "ignore_memory":
                reason = arguments.get("reason", "")
                logger.info(
                    f"[ToolAgentMemReader] ignore_memory for user={user_id} reason={reason!r}"
                )
                return []

            else:
                logger.info(f"[ToolAgentMemReader] Unknown tool: {tool_name}, ignoring.")
                return []

        logger.warning(
            f"[ToolAgentMemReader] Max rounds ({self.config.max_rounds}) exceeded "
            f"for user={user_id}, will fallback to parent extraction."
        )
        return None

    # ------------------------------------------------------------------
    # API call (with retries)
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict]) -> dict:
        """Call the external 4B tool calling service with exponential backoff, up to 3 retries."""
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "tools": _TOOLS,
            "tool_choice": "required",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.enable_thinking:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"[ToolAgentMemReader] _call_api: POST {self.config.api_url} "
            f"model={self.config.model} messages={len(messages)} "
            f"temperature={self.config.temperature} max_tokens={self.config.max_tokens} "
            f"enable_thinking={self.config.enable_thinking}"
        )

        last_exc: Exception | None = None
        delay = 1.0
        for attempt in range(3):
            try:
                logger.info(
                    f"[ToolAgentMemReader] _call_api attempt {attempt + 1}/3 "
                    f"sending request to {self.config.api_url}"
                )
                t0 = time.time()
                resp = requests.post(
                    self.config.api_url,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                elapsed = time.time() - t0
                resp.raise_for_status()
                logger.info(
                    f"[ToolAgentMemReader] _call_api success: "
                    f"status={resp.status_code} elapsed={elapsed:.2f}s"
                )
                return resp.json()
            except Exception as e:
                last_exc = e
                elapsed = time.time() - t0
                if attempt < 2:
                    err_body = ""
                    if hasattr(e, "response") and e.response is not None:
                        with contextlib.suppress(Exception):
                            err_body = e.response.text[:300]
                    logger.info(
                        f"[ToolAgentMemReader] API attempt {attempt + 1}/3 failed "
                        f"after {elapsed:.2f}s, retrying in {delay}s: {e} | body: {err_body}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.info(
                        f"[ToolAgentMemReader] API attempt {attempt + 1}/3 failed "
                        f"after {elapsed:.2f}s, no more retries: {e}"
                    )

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Search proxy
    # ------------------------------------------------------------------

    def _search_by_searcher(
        self,
        user_id: str,
        query: str,
        info: dict[str, Any] | None = None,
        user_name: str | None = None,
    ) -> list[str]:
        """Search via MemOS searcher, returns a list of text results."""
        logger.info(
            f"[ToolAgentMemReader] _search_by_searcher: user={user_id} "
            f"user_name={user_name!r} query={query!r} "
            f"top_k={self.config.search_top_k}"
        )
        if self.searcher is None:
            logger.info("[ToolAgentMemReader] No searcher configured, returning empty.")
            return []
        try:
            search_info = info if info else {"user_id": user_id, "session_id": ""}
            logger.info(
                f"[ToolAgentMemReader] _search_by_searcher calling searcher.search: "
                f"user_name={user_name!r} "
                f"info.user_id={search_info.get('user_id')!r} "
                f"info.session_id={search_info.get('session_id')!r}"
            )
            t0 = time.time()
            results = self.searcher.search(
                query=query,
                info=search_info,
                top_k=self.config.search_top_k,
                user_name=user_name,
            )
            elapsed = time.time() - t0
            texts: list[str] = []
            for r in results or []:
                if hasattr(r, "memory"):
                    texts.append(r.memory)
                elif isinstance(r, dict):
                    texts.append(r.get("memory", str(r)))
                else:
                    texts.append(str(r))
            logger.info(
                f"[ToolAgentMemReader] _search_by_searcher done: "
                f"results={len(texts)} elapsed={elapsed:.2f}s"
            )
            return texts
        except Exception as e:
            logger.info(f"[ToolAgentMemReader] Search failed: {e}")
            return []
