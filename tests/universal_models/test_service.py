"""Behavior tests for model discovery, profile persistence, and safe routing."""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli.universal_models import ModelService
from hermes_cli.universal_models.discovery import safe_endpoint


@pytest.fixture
def settings():
    return {
        "model": {"provider": "openrouter", "default": "anthropic/claude-test"},
        "providers": {
            "litellm": {
                "base_url": "http://127.0.0.1:4000/v1?key=hidden",
                "models": {"gateway-code": {"context_length": 100000}},
                "api_key": "NEVER_SAVE_SUPER_SECRET",
            },
        },
        "model_aliases": {"fast": {"provider": "openrouter", "model": "anthropic/claude-test", "api_key": "ALIAS_KEY"}},
        "auxiliary": {"compression": {"provider": "openrouter", "model": "google/gemini-test"}},
    }


def test_discovery_no_credentials_and_no_network_by_default(tmp_path, settings, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("status(False) cannot probe remote endpoints")

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", forbidden)
    svc = ModelService(home=tmp_path, config=settings)
    data = svc.status()
    entries = {(model["provider"], model["model"]) for model in data["models"]}
    assert ("openrouter", "anthropic/claude-test") in entries
    assert ("custom:litellm", "gateway-code") in entries
    assert data["aliases"] == [{"alias": "fast", "provider": "openrouter", "model": "anthropic/claude-test"}]
    assert data["hardware"]["gpu_discovery"] == "not_measured"
    assert data["local_only"] is False
    combined = json.dumps(data) + (tmp_path / "universal_models.sqlite3").read_bytes().decode("latin1")
    assert "NEVER_SAVE_SUPER_SECRET" not in combined
    assert "ALIAS_KEY" not in combined
    assert "?key=" not in combined


def test_alias_target_is_selectable_even_if_not_in_current_model(tmp_path):
    cfg = {"model_aliases": {"fast": {"provider": "deepseek", "model": "deepseek-chat"}}}
    svc = ModelService(home=tmp_path, config=cfg)
    assert "deepseek::deepseek-chat" in {m["id"] for m in svc.status()["models"]}
    assert svc.assign("delegation.leaf", "pinned", "deepseek::deepseek-chat")["mode"] == "pinned"


def test_assign_persists_across_instances_and_explicit_fallback(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    missing = svc.assign("delegation.leaf", "manual", "openrouter::anthropic/claude-test", ["openrouter::google/gemini-test"])
    assert missing["role"] == "delegation.leaf"
    again = ModelService(home=tmp_path, config=settings)
    assert missing in again.status()["roles"]
    assert {"main", "delegation.leaf", "delegation.orchestrator", "auxiliary.compression"}.issubset({r["role"] for r in again.status()["roles"]})
    assert again.resolve("delegation.leaf")["model"] == "anthropic/claude-test"
    assert again.resolve("delegation.orchestrator") is None


def test_gateway_pinned_mode_is_accepted(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    assigned = svc.assign("delegation.orchestrator", "pinned", "openrouter::anthropic/claude-test")
    assert assigned["mode"] == "pinned"
    assert svc.resolve("delegation.orchestrator")["provider"] == "openrouter"


def test_import_export_profiles_and_isolation(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    svc.assign("main", "manual", "openrouter::anthropic/claude-test")
    exported = svc.profile("export")
    imported = svc.profile("import", "team", exported)
    assert imported["roles"] == exported["roles"]
    svc.profile("activate", "team")
    svc.assign("delegation.leaf", "auto")
    assert svc.profile("export", "team")["roles"] != svc.profile("export", "default")["roles"]
    assert ModelService(home=tmp_path, config=settings).status()["active_profile"] == "team"
    with pytest.raises(ValueError):
        svc.profile("delete", "team")
    svc.profile("activate", "default")
    assert svc.profile("delete", "team") == {"deleted": "team"}


def test_fail_closed_on_credentials_and_invalid_profile(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    with pytest.raises(ValueError, match="credentials"):
        svc.profile("import", "unsafe", {"roles": [], "api_key": "do-not-store"})
    with pytest.raises(ValueError):
        svc.profile("import", "unsafe", {"roles": [{"role": "main", "mode": "manual", "model_id": "x", "api_key": "SECRET"}]})
    with pytest.raises(ValueError):
        svc.assign("../../../bad", "off")
    with pytest.raises(ValueError):
        svc.assign("delegation.leaf", "manual", "imaginary::model")
    assert svc.profile("list")["profiles"] == [{"name": "default", "active": True}]


def test_policy_blocks_unproven_claims_and_never_chooses_ambient_cloud(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    svc.assign("delegation.leaf", "auto", "openrouter::anthropic/claude-test")
    assert svc.resolve("delegation.leaf") is not None
    assert svc.resolve("delegation.leaf", require_capabilities=["tools"]) is None
    assert svc.resolve("delegation.leaf", min_context=1) is None
    svc.profile("save", "default", {"roles": svc.profile("export")["roles"], "policy": {"max_input_cost_per_million": 0}})
    assert svc.resolve("delegation.leaf") is None  # no verified cost data
    svc.profile("save", "default", {"roles": svc.profile("export")["roles"], "policy": {"local_only": True}})
    assert svc.resolve("delegation.leaf") is None
    assert svc.resolve("main") is None


def test_litellm_aliases_require_hermes_proxy_configuration(tmp_path):
    file = tmp_path / "litellm.yaml"
    file.write_text("model_list:\n  - model_name: code-local\n    litellm_params:\n      model: ollama/qwen3\n      api_key: LITELLM_SECRET_DO_NOT_SERIALIZE\n  - model_name: agent-cloud\n    litellm_params:\n      model: anthropic/claude\n")
    svc = ModelService(home=tmp_path, config={}, litellm_paths=[file])
    assert {row["availability"] for row in svc.status()["models"]} == {"unconfigured"}
    target = "custom:litellm::code-local"
    svc.assign("delegation.leaf", "manual", target)
    assert svc.resolve("delegation.leaf") is None
    svc_configured = ModelService(home=tmp_path, config={"providers": {"litellm": {"base_url": "http://localhost:4000/v1"}}}, litellm_paths=[file])
    assert svc_configured.resolve("delegation.leaf")["id"] == target
    assert "LITELLM_SECRET_DO_NOT_SERIALIZE" not in json.dumps(svc_configured.status())


def test_explicit_custom_loopback_is_discovered_when_refreshing(tmp_path, monkeypatch):
    calls = []

    def fake_probe(endpoints):
        calls.extend(endpoints)
        return []

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", fake_probe)
    settings = {"providers": {"my_server": {"base_url": "http://127.0.0.1:45678/v1/private?token=dont-leak"}}}
    svc = ModelService(home=tmp_path, config=settings, local_endpoints=[])
    svc.refresh()
    assert calls == ["http://127.0.0.1:45678"]
    assert "dont-leak" not in json.dumps(svc.status())


def test_cached_live_observation_expires_without_inventing_reachability(tmp_path, monkeypatch):
    from hermes_cli.universal_models.discovery import _row

    service = ModelService(home=tmp_path, config={})
    row = _row("custom", "qwen3:8b", "ollama-live", "reachable", endpoint="http://localhost:11434/v1")
    with sqlite3.connect(tmp_path / "universal_models.sqlite3") as connection:
        connection.execute("INSERT INTO observations(name,payload,measured_at) VALUES (?,?,?)", ("local-models", json.dumps([row]), 1000.0))
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1001.0)
    assert service.status()["models"][0]["availability"] == "reachable"
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1121.0)
    assert service.status()["models"] == []


def test_local_on_disk_is_discovered_but_not_runnable(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "small.gguf").write_bytes(b"gguf")
    svc = ModelService(home=tmp_path / "state", config={}, model_paths=[folder])
    model = next(row for row in svc.status()["models"] if row["source"] == "local-gguf")
    assert model["availability"] == "on_disk"
    svc.assign("delegation.leaf", "manual", model["id"])
    assert svc.resolve("delegation.leaf") is None


def test_opt_in_loopback_refresh_and_qualification(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/tags":
                payload = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        svc = ModelService(home=tmp_path, config={}, local_endpoints=[endpoint, "https://example.com/v1"])
        assert svc.status()["models"] == []
        fresh = svc.refresh()
        assert len(fresh["models"]) == 1
        model = fresh["models"][0]
        assert model["availability"] == "reachable"
        assert model["source"] == "ollama-live"
        assert model["endpoint"].endswith("/v1")
        # Gateway RPC creates a fresh service for each request.
        another_instance = ModelService(home=tmp_path, config={}, local_endpoints=[endpoint])
        assert another_instance.status()["models"][0]["id"] == model["id"]
        svc.assign("delegation.leaf", "manual", model["id"])
        svc.profile("save", "default", {"roles": svc.profile("export")["roles"], "policy": {"local_only": True}})
        assert svc.resolve("delegation.leaf")["id"] == model["id"]
        assert svc.qualify(model["id"]) == {"model_id": model["id"], "available": True, "level": "model-list-only", "inference_tested": False}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_safe_endpoint_strips_secret_data():
    assert safe_endpoint("https://user:password@example.com:443/private/key?token=secret#secret") == "https://example.com:443"
    assert safe_endpoint("http://[::1]:11434/api/tags") == "http://[::1]:11434"
    assert safe_endpoint("https://{TOKEN}@proxy.example/v1") == ""
    assert safe_endpoint("file:///etc/passwd") == ""


def test_same_model_on_two_local_endpoints_gets_distinct_stable_ids():
    from hermes_cli.universal_models.discovery import merge_models, _row

    first = _row("custom", "qwen3:8b", "ollama-live", "reachable", endpoint="http://127.0.0.1:11434/v1")
    second = _row("custom", "qwen3:8b", "ollama-live", "reachable", endpoint="http://127.0.0.1:11435/v1")
    assert len(merge_models([first, second])) == 2
    assert first["id"] != second["id"]
    assert first["id"] == _row("custom", "qwen3:8b", "ollama-live", "reachable", endpoint="http://127.0.0.1:11434/v1")["id"]


def test_profile_storage_only_contains_whitelisted_settings(tmp_path, settings):
    svc = ModelService(home=tmp_path, config=settings)
    svc.assign("delegation.orchestrator", "manual", "openrouter::anthropic/claude-test")
    with sqlite3.connect(tmp_path / "universal_models.sqlite3") as conn:
        stored = json.loads(conn.execute("SELECT payload FROM profiles WHERE name='default'").fetchone()[0])
    assert set(stored) == {"roles", "policy"}
    assert stored["roles"][0]["role"] == "delegation.orchestrator"
