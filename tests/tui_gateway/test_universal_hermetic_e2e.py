"""Registry RPC to real AIAgent streaming over an isolated local OpenAI wire."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from hermes_cli.universal_models.service import ModelService
from tui_gateway import server
from tui_gateway.universal_model_runtime import main_model_override


@pytest.fixture
def ollama_wire(monkeypatch):
    requests = []
    model = "hermetic-stub"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/api/tags":
                data = {"models": [{"name": model}]}
            elif self.path == "/api/ps":
                data = {"models": [{"name": model, "context_length": 4096}]}
            else:
                self.send_error(404)
                return
            body = json.dumps(data).encode()
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
            if payload["model"] != model:
                self.send_error(400)
                return
            reply = {
                "id": "chatcmpl-hermetic", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "HERMETIC_REPLY"},
                             "finish_reason": "stop"}],
            }
            data = f"data: {json.dumps(reply)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    try:
        yield f"http://127.0.0.1:{http.server_port}", requests, model
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def _rpc(method, params=None):
    return server.handle_request({"jsonrpc": "2.0", "id": "test", "method": method,
                                  "params": params or {}})


def test_registry_rpc_to_real_agent_reply_with_no_cloud_fallback(tmp_path, monkeypatch, ollama_wire):
    endpoint, sent, expected_model = ollama_wire
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service = ModelService(home=tmp_path, config={"model": {"base_url": endpoint}},
                           local_endpoints=[], litellm_paths=[], model_paths=[])
    observed = service.refresh()["models"]
    local = next(row for row in observed if row["model"] == expected_model
                 and row["source"] == "ollama-live")
    assert local["loaded"] is True

    policy = _rpc("orchestration.policy", {"action": "set", "policy": {"local_only": True}})
    assert policy["result"]["policy"]["local_only"] is True
    assigned = _rpc("orchestration.assign", {"role": "main", "mode": "pinned",
                                              "model_id": local["id"], "fallback_ids": []})
    assert assigned["result"]["deferred"] is True
    selected = _rpc("orchestration.profile", {"action": "export"})
    assert selected["result"]["roles"][0]["model_id"] == local["id"]

    # The checkout's recovery hook resolves Git's common directory beneath
    # the user's installed Hermes tree; it is irrelevant to test-side agent
    # imports and forbidden by the isolated-home I/O guard.
    with patch("hermes_cli._early_recovery.restore_interrupted_pull", return_value=False):
        from run_agent import AIAgent

    assert AIAgent.__name__ == "AIAgent"

    cfg = {"model": {"provider": "openrouter", "default": "should-never-run"},
           "agent": {"max_turns": 1}}
    with patch.object(server, "_load_cfg", return_value=cfg), \
            patch.object(server, "_get_db", return_value=None), \
            patch.object(server, "_load_enabled_toolsets", return_value=[]), \
            patch.object(server, "_load_disabled_toolsets", return_value=[]), \
            patch.object(server, "_agent_cbs", return_value={}), \
            patch("agent.shell_hooks.register_from_config"), \
            patch.object(server, "_resolve_runtime_with_fallback",
                         side_effect=AssertionError("cloud provider resolver invoked")):
        agent = server._make_agent("hermetic-sid", "hermetic-key",
                                   model_override=main_model_override(local))
        try:
            reply = agent.chat("Reply with one word.")
        finally:
            agent.close()

    assert "HERMETIC_REPLY" in reply
    assert len(sent) == 1
    assert sent[0]["model"] == expected_model
    assert sent[0]["stream"] is True
    assert agent.provider == "custom"
    assert agent._fallback_chain == []


def test_stale_session_refuses_registry_mutations_without_network(tmp_path, monkeypatch, ollama_wire):
    _endpoint, requests, _model = ollama_wire
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch.dict(server._sessions, {}, clear=True):
        for method, params in (
            ("orchestration.status", {}),
            ("orchestration.profile", {"action": "export"}),
            ("orchestration.policy", {"action": "set", "policy": {"local_only": False}}),
            ("orchestration.assign", {"role": "main", "mode": "auto"}),
            ("orchestration.qualify", {"model_id": "missing"}),
        ):
            response = _rpc(method, {**params, "session_id": "unowned-session"})
            assert response["error"]["code"] == 4001, (method, response)
    assert not requests
    assert not (tmp_path / "universal_models.sqlite3").exists()
