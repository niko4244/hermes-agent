"""Opt-in bounded, reproducible *local* Ollama inference observations.

This probes a fixed, public arithmetic prompt. It is not a general model
benchmark or a tool-use certification. It never sends credentials or a user's
conversation to the model, and it does not download or start model services.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from .discovery import safe_endpoint


_PROMPT = "What is 2 + 2? Reply with the single digit only."
_EXPECTED = "4"
_BODY_LIMIT = 64 * 1024
_STREAM_LINE_LIMIT = 8 * 1024
_OUTPUT_LIMIT = 2048


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _read_json(opener: Any, url: str, timeout: float) -> dict | None:
    try:
        with opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as reply:
            if reply.status != 200:
                return None
            data = reply.read(128 * 1024 + 1)
            if len(data) > 128 * 1024:
                return None
            payload = json.loads(data)
            return payload if isinstance(payload, dict) else None
    except (OSError, UnicodeError, ValueError, urllib.error.HTTPError):
        return None


def _valid_origin(row: dict) -> str | None:
    if row.get("source") != "ollama-live" or row.get("availability") != "reachable" or row.get("provider") != "custom":
        return None
    endpoint = row.get("endpoint")
    if not isinstance(endpoint, str):
        return None
    try:
        parsed = urllib.parse.urlsplit(endpoint)
    except ValueError:
        return None
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        return None
    origin = safe_endpoint(endpoint)
    if endpoint != f"{origin}/v1" or parsed.username or parsed.password:
        return None
    return origin


def _exists(payload: dict | None, model: str) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return False
    return any(isinstance(entry, dict) and (entry.get("name") == model or entry.get("model") == model)
               for entry in payload["models"])


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _stream_chat(opener: Any, origin: str, model: str, max_tokens: int, timeout_seconds: float) -> dict:
    """Stream with per-read remaining-deadline timeout, size and token limits."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": _PROMPT}],
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{origin}/api/chat", data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    started = time.monotonic()
    deadline = started + timeout_seconds
    response_text = ""
    first_text_at: float | None = None
    first_visible_at: float | None = None
    last: dict = {}
    used_bytes = 0
    seen_done = False
    with opener.open(request, timeout=timeout_seconds) as reply:
        if reply.status != 200:
            raise ValueError("unexpected_status")
        for _ in range(512):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("deadline_exceeded")
            # urllib HTTPResponse exposes its underlying socket. Updating the
            # timeout on each read makes the *total* deadline enforceable even
            # when a server trickles a stream one event at a time.
            sock = getattr(getattr(getattr(reply, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(remaining)
            line = reply.readline(_STREAM_LINE_LIMIT + 1)
            if not line:
                break
            used_bytes += len(line)
            if len(line) > _STREAM_LINE_LIMIT or used_bytes > _BODY_LIMIT:
                raise ValueError("response_limit_exceeded")
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("malformed_stream")
            if event.get("model") not in (None, model):
                raise ValueError("model_identity_mismatch")
            content = event.get("message", {})
            part = content.get("content") if isinstance(content, dict) else None
            thought = content.get("thinking") if isinstance(content, dict) else None
            if first_text_at is None and ((isinstance(thought, str) and thought) or (isinstance(part, str) and part)):
                first_text_at = time.monotonic()
            if isinstance(part, str) and part:
                if first_visible_at is None:
                    first_visible_at = time.monotonic()
                response_text += part
                if len(response_text) > _OUTPUT_LIMIT:
                    raise ValueError("output_limit_exceeded")
            if event.get("done") is True:
                seen_done = True
                last = event
                break
    elapsed = time.monotonic() - started
    if not seen_done:
        raise ValueError("incomplete_stream")
    token_count = _positive_int(last.get("eval_count"))
    if token_count is not None and token_count > max_tokens:
        raise ValueError("server_exceeded_max_tokens")
    eval_duration_ns = _positive_int(last.get("eval_duration"))
    tps = (
        round(token_count * 1_000_000_000 / eval_duration_ns, 3)
        if token_count is not None and eval_duration_ns is not None else None
    )
    if tps is not None and not math.isfinite(tps):
        tps = None
    return {
        "ttft_ms": round((first_text_at - started) * 1000, 3) if first_text_at is not None else None,
        "ttft_source": "first_nonempty_ollama_streamed_text_chunk",
        "first_visible_content_ms": round((first_visible_at - started) * 1000, 3) if first_visible_at is not None else None,
        "elapsed_ms": round(elapsed * 1000, 3),
        "tokens_per_second": tps,
        "tokens_per_second_source": "ollama_reported_eval_count_and_duration" if tps is not None else None,
        "completion_tokens_reported": token_count,
        "eval_duration_ns_reported": eval_duration_ns,
        "quality": {"probe": "fixed_arithmetic_exact_match_v1", "method": "exact_match", "passed": response_text.strip() == _EXPECTED},
        "provenance": {
            "protocol": "ollama_native_streaming_chat", "backend": "native_ollama_listing",
            "prompt_sha256": hashlib.sha256(_PROMPT.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
            "model_reported": last.get("model") if isinstance(last.get("model"), str) else None,
            "done": True,
        },
    }


def qualify_native_ollama(
    row: dict,
    *,
    timeout_seconds: float = 15.0,
    max_tokens: int = 16,
    require_loaded: bool = True,
) -> dict:
    """Bounded live inference, only after *caller* explicitly opted in.

    No fallback, arbitrary prompt, cloud proxy, credentials, model pulling or
    subprocess invocation. The default requires /api/ps to show the model is
    *already loaded* before generation, so this check cannot evict another
    locally loaded model through a new load.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int)) or not 1 <= timeout_seconds <= 30:
        raise ValueError("timeout_seconds must be between 1 and 30")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 64:
        raise ValueError("max_tokens must be between 1 and 64")
    if not isinstance(require_loaded, bool):
        raise ValueError("require_loaded must be a boolean")
    model_id = row.get("id")
    result: dict[str, Any] = {
        "model_id": model_id,
        "available": False,
        "level": "bounded-local-inference",
        "inference_tested": False,
        "status": "refused",
        "timestamp": datetime.now(UTC).isoformat(),
        "settings": {"timeout_seconds": float(timeout_seconds), "max_tokens": max_tokens,
                     "require_loaded": require_loaded, "temperature": 0, "prompt_id": "fixed_arithmetic_exact_match_v1"},
    }
    origin = _valid_origin(row)
    if origin is None:
        result["reason"] = "not_verified_native_ollama_loopback"
        return result
    model = row.get("model")
    if not isinstance(model, str) or not model or any(ord(c) < 32 for c in model):
        result["reason"] = "invalid_model_identity"
        return result
    opener = _opener()
    began = time.monotonic()
    deadline = began + timeout_seconds

    def remaining() -> float:
        budget = deadline - time.monotonic()
        if budget <= 0:
            return 0.00001
        return min(budget, 3.0)

    # Fresh verification prevents a 120-second old catalog row from being
    # treated as proof that a particular Ollama model is still available.
    if not _exists(_read_json(opener, f"{origin}/api/tags", remaining()), model):
        result["reason"] = "model_not_listed_now"
        return result
    if require_loaded and not _exists(_read_json(opener, f"{origin}/api/ps", remaining()), model):
        result["reason"] = "model_not_already_loaded"
        return result
    result["available"] = True
    try:
        left = deadline - time.monotonic()
        if left <= 0:
            raise ValueError("deadline_exceeded")
        measured = _stream_chat(opener, origin, model, max_tokens, left)
    except (OSError, ValueError, TimeoutError, UnicodeError, urllib.error.HTTPError):
        # Never serialize raw HTTP errors: headers and body might contain keys.
        result["status"] = "failed"
        result["reason"] = "stream_failed_or_incomplete"
        return result
    result.update(measured)
    result["inference_tested"] = True
    result["status"] = "passed" if measured["quality"]["passed"] else "failed"
    result["reason"] = "exact_match" if measured["quality"]["passed"] else "exact_match_failed"
    result["provenance"].update({"endpoint": f"{origin}/v1", "model": model})
    return result
