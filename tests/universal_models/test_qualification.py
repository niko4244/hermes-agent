"""Native HTTP contracts and falsifiable provenance for local qualification."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli.universal_models import ModelService


@pytest.fixture
def local_ollama():
    state = {"loaded": True, "requests": [], "response": "4", "mode": "success", "eval_count": 2}

    class NativeOllama(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["requests"].append(("GET", self.path))
            if self.path == "/api/tags":
                self._json({"models": [{"name": "test-native:3b"}]})
            elif self.path == "/api/ps":
                models = [{"name": "test-native:3b", "context_length": 131072}] if state["loaded"] else []
                self._json({"models": models})
            else:
                self.send_error(404)

        def do_POST(self):
            state["requests"].append(("POST", self.path))
            assert self.path == "/api/chat"
            assert "Authorization" not in self.headers
            assert self.headers.get("Content-Type") == "application/json"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["model"] == "test-native:3b"
            assert payload["options"]["num_predict"] <= 64
            assert payload["options"]["temperature"] == 0
            assert payload["stream"] is True
            assert payload["messages"] == [{"role": "user", "content": "What is 2 + 2? Reply with the single digit only."}]
            if state["mode"] == "redirect":
                self.send_response(307)
                self.send_header("Location", "https://cloud.example.invalid/chat")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            model = "not-selected" if state["mode"] == "mismatch" else "test-native:3b"
            first = {"model": model, "message": {"role": "assistant", "content": state["response"]}}
            self.wfile.write(json.dumps(first).encode() + b"\n")
            if state["mode"] != "incomplete":
                final = {"model": model, "done": True, "message": {"role": "assistant", "content": ""},
                         "eval_count": state["eval_count"], "eval_duration": 1_000_000_000}
                self.wfile.write(json.dumps(final).encode() + b"\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), NativeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _service(home, url):
    svc = ModelService(home=home, config={}, local_endpoints=[url])
    model = svc.refresh()["models"][0]
    return svc, model


def test_bounded_real_http_stream_collects_provenance_and_persists(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    assert model["loaded"] is True
    assert model["context_window"] == 131072
    assert model["context_source"] == "ollama_api_ps_loaded_context"
    # The earlier listing-only API sends no chat request.
    listing = service.qualify(model["id"])
    assert listing["inference_tested"] is False
    assert ("POST", "/api/chat") not in state["requests"]

    result = service.qualify(model["id"], run_inference=True, timeout_seconds=5, max_tokens=12)
    assert result["status"] == "passed"
    assert result["available"] and result["inference_tested"]
    assert result["quality"] == {"probe": "fixed_arithmetic_exact_match_v1", "method": "exact_match", "passed": True}
    assert result["ttft_ms"] >= 0
    assert result["ttft_source"] == "first_nonempty_ollama_streamed_text_chunk"
    assert result["first_visible_content_ms"] >= result["ttft_ms"]
    assert result["elapsed_ms"] >= result["ttft_ms"]
    assert result["completion_tokens_reported"] == 2
    assert result["tokens_per_second"] == 2.0
    assert result["tokens_per_second_source"] == "ollama_reported_eval_count_and_duration"
    assert result["provenance"]["model_reported"] == "test-native:3b"
    assert result["provenance"]["protocol"] == "ollama_native_streaming_chat"
    assert result["settings"]["max_tokens"] == 12
    assert "+00:00" in result["timestamp"]
    assert state["requests"].count(("POST", "/api/chat")) == 1
    second = ModelService(home=tmp_path, config={}, local_endpoints=[url])
    assert second.status()["models"][0]["last_qualification"] == result
    assert second.qualification(model["id"]) == result
    with sqlite3.connect(tmp_path / "universal_models.sqlite3") as con:
        stored = con.execute("SELECT payload FROM qualifications WHERE model_id=?", (model["id"],)).fetchone()[0]
    assert json.loads(stored) == result
    assert "What is 2 + 2" not in stored  # provenance is hashed
    assert '"content":' not in stored


def test_unloaded_model_refuses_before_post_and_does_not_trigger_load(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    state["loaded"] = False
    result = service.qualify(model["id"], run_inference=True)
    assert result["status"] == "refused"
    assert result["reason"] == "model_not_already_loaded"
    assert result["inference_tested"] is False
    assert not any(method == "POST" for method, _ in state["requests"])


def test_explicit_override_may_run_unloaded_native_model(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    state["loaded"] = False
    outcome = service.qualify(model["id"], run_inference=True, require_loaded=False, max_tokens=4)
    assert outcome["status"] == "passed"
    assert outcome["settings"]["require_loaded"] is False
    assert state["requests"].count(("POST", "/api/chat")) == 1


@pytest.mark.parametrize("mode,expected", [("redirect", "stream_failed_or_incomplete"),
                                            ("mismatch", "stream_failed_or_incomplete"),
                                            ("incomplete", "stream_failed_or_incomplete")])
def test_redirect_model_swap_and_truncated_stream_are_fail_closed(tmp_path, local_ollama, mode, expected):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    state["mode"] = mode
    outcome = service.qualify(model["id"], run_inference=True, timeout_seconds=3)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == expected
    assert outcome["inference_tested"] is False


def test_quality_failure_is_measured_not_an_inference_failure(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    state["response"] = "five"
    outcome = service.qualify(model["id"], run_inference=True)
    assert outcome["inference_tested"] is True
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "exact_match_failed"
    assert outcome["quality"]["passed"] is False


def test_server_ignoring_token_ceiling_invalidates_qualification(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    state["eval_count"] = 100
    result = service.qualify(model["id"], run_inference=True, max_tokens=8)
    assert result["inference_tested"] is False
    assert result["status"] == "failed"
    assert result["reason"] == "stream_failed_or_incomplete"


def test_opt_in_rejects_remote_unlisted_and_bad_limits_without_generating(tmp_path, local_ollama):
    url, state = local_ollama
    service, model = _service(tmp_path, url)
    with pytest.raises(ValueError, match="recent explicit local discovery"):
        service.qualify("openrouter::anthropic/claude-test", run_inference=True)
    with pytest.raises(ValueError, match="max_tokens"):
        service.qualify(model["id"], run_inference=True, max_tokens=65)
    with pytest.raises(ValueError, match="timeout_seconds"):
        service.qualify(model["id"], run_inference=True, timeout_seconds=120)
    with pytest.raises(ValueError, match="timeout_seconds"):
        service.qualify(model["id"], run_inference=True, timeout_seconds=True)
    assert not any(method == "POST" for method, _ in state["requests"])


def test_auto_routing_is_deterministic_local_and_pin_never_falls_through(tmp_path, local_ollama):
    url, _ = local_ollama
    config = {"model": {"provider": "openrouter", "default": "billable/model"}}
    service = ModelService(home=tmp_path, config=config, local_endpoints=[url])
    assert service.resolve("delegation.leaf") is None
    local = next(row for row in service.refresh()["models"] if row["source"] == "ollama-live")
    assert service.resolve("delegation.leaf")["id"] == local["id"]
    service.assign("delegation.leaf", "pinned", "openrouter::billable/model", [local["id"]])
    service.profile("save", "default", {"roles": service.profile("export")["roles"], "policy": {"local_only": True}})
    assert service.resolve("delegation.leaf") is None
    service.assign("delegation.leaf", "auto")
    assert service.resolve("delegation.leaf")["id"] == local["id"]


def test_future_dated_observations_do_not_claim_live(tmp_path, local_ollama, monkeypatch):
    url, _ = local_ollama
    svc, model = _service(tmp_path, url)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1.0)
    assert svc.status()["models"] == []
    assert svc.resolve("delegation.leaf") is None
