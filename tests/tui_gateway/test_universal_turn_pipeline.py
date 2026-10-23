"""Run saved local routing through Hermes' real prompt turn and settlement path."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.universal_models.service import ModelService
from tui_gateway import server


@pytest.fixture
def local_ollama_wire(monkeypatch):
    requests = []
    model = "turn-pipeline-stub"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/api/tags":
                payload = {"models": [{"name": model}]}
            elif self.path == "/api/ps":
                payload = {"models": [{"name": model, "context_length": 4096}]}
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            reply = {
                "id": "chatcmpl-turn", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "TURN_PIPELINE_OK"},
                             "finish_reason": "stop"}],
            }
            body = f"data: {json.dumps(reply)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests, model
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def _rpc(method, params):
    return server.handle_request({"jsonrpc": "2.0", "id": "turn-test",
                                  "method": method, "params": params})


@pytest.fixture
def pipeline(tmp_path, monkeypatch, local_ollama_wire):
    endpoint, requests, model = local_ollama_wire
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = ModelService(home=tmp_path, config={"model": {"base_url": endpoint}},
                            local_endpoints=[], litellm_paths=[], model_paths=[])
    observed = registry.refresh()["models"]
    row = next(item for item in observed if item["model"] == model and item["source"] == "ollama-live")
    assert row["loaded"]
    assert _rpc("orchestration.policy", {"action": "set", "policy": {"local_only": True}})["result"]
    assert _rpc("orchestration.assign", {"role": "main", "mode": "pinned",
                                         "model_id": row["id"], "fallback_ids": []})["result"]

    # The import-time update recovery is unrelated to a gateway turn and can
    # traverse the user's installed Git common directory from a worktree.
    with patch("hermes_cli._early_recovery.restore_interrupted_pull", return_value=False):
        from run_agent import AIAgent

    events = []
    monkeypatch.setattr(server, "_emit", lambda name, sid, payload=None: events.append((name, payload)))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {
        "model": {"provider": "openrouter", "default": "cloud-never-called"},
        "agent": {"max_turns": 1},
    })
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_: [])
    monkeypatch.setattr(server, "_load_disabled_toolsets", list)
    monkeypatch.setattr(server, "_agent_cbs", lambda *_: {})
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_: True)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_: None)
    monkeypatch.setattr(server, "_adopt_out_of_band_turns", lambda *_: None)
    monkeypatch.setattr(server, "_stage_first_contact_onboarding_note", lambda *_: None)
    monkeypatch.setattr(server, "_stage_onboarding_turn_note", lambda *_: None)
    monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *_: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_: None)
    monkeypatch.setattr(server, "_session_live_title", lambda *_: "")
    monkeypatch.setattr(server, "_get_usage", lambda *_: {})
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_: None)
    monkeypatch.setattr(server, "_run_post_turn_followups", lambda *_: None)
    monkeypatch.setattr(server, "_after_complete_turn", lambda *_: None)
    monkeypatch.setattr(server, "_publish_session_control_snapshot", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_sessions_quiescent", lambda **_kw: False)
    monkeypatch.setattr(server, "_CRASH_LOG", str(tmp_path / "turn-crash.log"))
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *_: (
        threading.Event(), SimpleNamespace(join=lambda: None)))

    def start_inline(fn, **_kw):
        fn()
        return object()

    monkeypatch.setattr(server, "_start_session_work", start_inline)
    session = {
        "agent": None, "session_key": "turn-pipeline-session", "running": False,
        "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "attached_images": [], "cols": 80, "inflight_turn": None,
    }
    with patch.dict(server._sessions, {"turn-sid": session}), \
            patch("agent.shell_hooks.register_from_config"), \
            patch.object(server, "_resolve_runtime_with_fallback",
                         side_effect=AssertionError("cloud provider resolver called")):
        agent = server._make_agent("turn-sid", session["session_key"])
        assert isinstance(agent, AIAgent)
        session["agent"] = agent
        try:
            yield registry, session, events, requests, row
        finally:
            agent.close()


def test_gateway_turn_emits_stream_and_completes_from_saved_local_role(pipeline):
    _registry, session, events, requests, row = pipeline
    assert session["model_override"]["_universal_model_id"] == row["id"]
    assert server._run_prompt_submit("turn-request", "turn-sid", session, "Say one word") is True
    complete = [payload for kind, payload in events if kind == "message.complete"]
    deltas = [payload for kind, payload in events if kind == "message.delta"]
    assert len(complete) == 1
    assert complete[0]["status"] == "complete"
    assert "TURN_PIPELINE_OK" in complete[0]["text"]
    assert any("TURN_PIPELINE_OK" in delta["text"] for delta in deltas)
    assert session["running"] is False
    assert server._inflight_snapshot(session) is None
    assert len(requests) == 1
    assert requests[0]["model"] == row["model"]
    assert requests[0]["stream"] is True
    assert session["agent"]._fallback_chain == []


def test_changed_policy_rejects_before_chat_and_settles_turn(pipeline):
    registry, session, events, requests, _row = pipeline
    registry.profile("save", "default", {
        "name": "default", "roles": registry.profile("export")["roles"],
        "policy": {"local_only": True, "max_output_cost_per_million": 0},
    })
    assert server._run_prompt_submit("blocked-request", "turn-sid", session, "Must refuse") is True
    assert requests == []
    assert session["running"] is False
    snapshot = server._inflight_snapshot(session)
    assert snapshot["status"] == "error"
    complete = [payload for name, payload in events if name == "message.complete"]
    assert len(complete) == 1
    assert complete[0]["status"] == "error"
    assert complete[0]["recoverable"] is True
