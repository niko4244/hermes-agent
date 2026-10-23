"""Saved universal model roles integrated into real TaskRoute resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.universal_models import ModelService
from tools.delegate_tool_routes import resolve_task_routes


class Backend:
    name = "probe-delegation"

    def __init__(self, routes):
        self.routes = routes

    def task_parameters(self, config):
        return {}

    def route_task(self, task, config):
        result = self.routes[task["index"]]
        return dict(config) if result is None else dict(result)


@pytest.fixture
def routing(tmp_path, monkeypatch):
    cfg = {
        "model": {"provider": "openrouter", "default": "billable/parent"},
        "providers": {
            "deepseek": {"models": ["coder", "orchestrator"]},
            "anthropic": {"models": ["backup"]},
        },
        "delegation": {"fallback_providers": [{"provider": "openrouter", "model": "legacy/paid-fallback"}]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service = ModelService(home=tmp_path, config=cfg, local_endpoints=[], model_paths=[])

    def credentials(cfg, parent):
        provider = cfg.get("provider")
        return {"provider": provider, "model": cfg.get("model"), "api_key": f"key-{provider}" if provider else None,
                "base_url": f"https://{provider}.test/v1" if provider else None,
                "api_mode": "chat_completions", "request_overrides": None}

    monkeypatch.setattr("tools.delegate_tool._resolve_delegation_credentials", credentials)
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr("tools.delegate_tool._get_orchestrator_enabled", lambda: True)
    parent = SimpleNamespace(_delegate_depth=0, provider="openrouter", api_key="PARENT_SECRET")

    def execute(routes, *, delegation=None, depth=0):
        parent._delegate_depth = depth
        monkeypatch.setattr("tools.delegate_tool_routes._active_provider", lambda _cfg: Backend(routes))
        return resolve_task_routes([{"goal": f"task {n}"} for n in range(len(routes))],
                                   cfg["delegation"] if delegation is None else delegation, parent)

    return service, cfg, parent, execute


def test_saved_leaf_route_reaches_taskroute_and_preserves_role_fallback_order(routing):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder", ["anthropic::backup"])
    (selected,), error = execute([{"label": "specialist"}], depth=1)
    assert error is None and selected.error is None
    assert selected.creds["provider"] == "deepseek" and selected.creds["model"] == "coder"
    assert selected.creds["api_key"] == "key-deepseek"  # resolved by Hermes, not the parent's key
    assert selected.routing_cfg["fallback_providers"] == [
        {"provider": "anthropic", "model": "backup", "api_key": "key-anthropic",
         "base_url": "https://anthropic.test/v1", "api_mode": "chat_completions", "request_overrides": None},
    ]
    assert selected.report() == {"backend": "probe-delegation", "label": "specialist"}
    assert "PARENT_SECRET" not in str(selected.routing_cfg)
    assert "legacy/paid-fallback" not in str(selected.routing_cfg)


def test_child_effective_depth_selects_orchestrator_then_leaf_not_task_role(routing):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder")
    service.assign("delegation.orchestrator", "pinned", "deepseek::orchestrator")
    (first,), err = execute([{}], depth=0)
    assert err is None and first.creds["model"] == "orchestrator"
    (nested,), err = execute([{}], depth=1)
    assert err is None and nested.creds["model"] == "coder"


@pytest.mark.parametrize("route", [
    {"provider": "anthropic", "model": "task-owned"},
    {"model": "per-task-model"},
    {"base_url": "http://127.0.0.1:8080/v1", "model": "local-pin"},
    {"api_key": "task-key", "model": "endpoint-pin"},
])
def test_explicit_provider_or_task_model_always_beats_saved_assignment(routing, route):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder")
    (selected,), err = execute([route], depth=1)
    assert err is None and selected.error is None
    assert selected.creds["model"] == route.get("model")
    assert selected.routing_cfg["fallback_providers"] == [{"provider": "openrouter", "model": "legacy/paid-fallback"}]


def test_explicit_per_task_fallback_policy_beats_saved_role_fallbacks(routing):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder", ["anthropic::backup"])
    explicit = [{"provider": "openrouter", "model": "task/backup"}]
    (selected,), err = execute([{"fallback_providers": explicit}], depth=1)
    assert err is None and selected.error is None
    assert selected.creds["model"] == "coder"
    assert selected.routing_cfg["fallback_providers"] == explicit


def test_missing_pinned_role_refuses_task_without_inheriting_parent(routing):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder")
    # Remove its declared entry: the saved pin must not fall through to parent.
    path = service.home / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["providers"]["deepseek"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (refused,), error = execute([{}], depth=1)
    assert error is None and refused.creds == {}
    assert "unavailable" in (refused.error or "")
    assert "PARENT_SECRET" not in str(refused.error)


def test_explicit_cross_profile_route_still_has_priority(routing, monkeypatch, tmp_path):
    service, _, _, execute = routing
    service.assign("delegation.leaf", "pinned", "deepseek::coder")
    service.profile("save", "default", {"roles": service.profile("export")["roles"],
                                        "policy": {"local_only": True}})
    from tools.delegate_tool_routes import TaskRoute

    def profile_route(home):
        assert home == tmp_path / "other"
        return TaskRoute({"provider": "anthropic", "model": "different-profile", "api_key": "OTHER_PROFILE_KEY"},
                         {"fallback_providers": []}, profile_home=home)

    monkeypatch.setattr("tools.delegate_tool_routes._profile_home", lambda name: tmp_path / name)
    monkeypatch.setattr("tools.delegate_tool_routes._profile_route", profile_route)
    (selected,), error = execute([{"profile": "other", "label": "other-owned"}], depth=1)
    assert error is None and selected.error is None
    assert selected.profile_home == tmp_path / "other"
    assert selected.creds["model"] == "different-profile"
    assert selected.report() == {"backend": "probe-delegation", "label": "other-owned"}


def test_local_only_policy_refuses_explicit_cloud_pin_without_fallback(routing):
    service, _, _, execute = routing
    service.profile("save", "default", {"roles": [], "policy": {"local_only": True}})
    (refused,), error = execute([{"provider": "deepseek", "model": "coder"}], depth=1)
    assert error is None and refused.creds == {}
    assert "local-only" in (refused.error or "")


def test_local_only_explicit_native_pin_requires_exact_listed_endpoint(routing):
    import json
    import time

    service, _, _, execute = routing
    service.profile("save", "default", {"roles": [], "policy": {"local_only": True}})
    native = {"id": "custom@observed::model", "provider": "custom", "model": "model",
              "availability": "reachable", "source": "ollama-live",
              "endpoint": "http://127.0.0.1:11434/v1", "capabilities": []}
    with service._connect() as conn:
        conn.execute("INSERT INTO observations(name,payload,measured_at) VALUES (?,?,?)",
                     ("local-models", json.dumps([native]), time.time()))

    # The fixture's local provider credential resolver is a harmless mock.
    # Its endpoint is deliberately not the observed native endpoint, so even
    # though the host is local in the config it cannot inherit cloud creds.
    (refused,), error = execute([{"provider": "custom", "model": "model",
                                  "base_url": "http://127.0.0.1:1234/v1"}], depth=1)
    assert error is None and refused.creds == {}
    assert "local-only" in (refused.error or "")


def test_legacy_delegation_without_saved_registry_is_nonmutating(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import _route_to_task_route

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.delegate_tool._resolve_delegation_credentials", lambda route, parent: {
        "model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None,
        "request_overrides": None,
    })
    monkeypatch.setattr("tools.delegate_tool._get_orchestrator_enabled", lambda: False)
    parent = SimpleNamespace(_delegate_depth=0)
    route = _route_to_task_route({}, parent, {})
    assert route.creds["provider"] is None and route.routing_cfg == {}
    assert not (tmp_path / "universal_models.sqlite3").exists()
