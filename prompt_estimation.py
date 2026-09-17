"""Conservative prompt reservation estimates for the Gemini request payload.

The repository does not have sentencepiece installed, so the Google GenAI
LocalTokenizer cannot be used. This fallback counts the UTF-8 serialized
payload and applies a safety multiplier to cover provider tokenization and
message/tool structure overhead. It is a reservation estimate, not provider
usage accounting.
"""

import json
import math
from typing import Any, Sequence


FALLBACK_BYTES_PER_TOKEN = 4
FALLBACK_SAFETY_MULTIPLIER = 1.5


def estimate_complete_prompt_tokens(
    *,
    system_text: str,
    contents: Sequence[Any],
    retrieved_context: Sequence[Any] = (),
    tool_definitions: Sequence[Any] = (),
) -> int:
    payload = {
        "system_instruction": system_text,
        "contents": list(contents),
        "retrieved_context": list(retrieved_context),
        "tools": list(tool_definitions),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    raw_estimate = max(1, math.ceil(len(serialized.encode("utf-8")) / FALLBACK_BYTES_PER_TOKEN))
    return max(1, math.ceil(raw_estimate * FALLBACK_SAFETY_MULTIPLIER))


def estimate_reservation_tokens(
    *,
    system_text: str,
    contents: Sequence[Any],
    max_output_tokens: int,
    retrieved_context: Sequence[Any] = (),
    tool_definitions: Sequence[Any] = (),
) -> int:
    if isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    return estimate_complete_prompt_tokens(
        system_text=system_text,
        contents=contents,
        retrieved_context=retrieved_context,
        tool_definitions=tool_definitions,
    ) + max_output_tokens
