"""Execute universal-model RPC -> live session handoff, not source-text checks."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tui_gateway import server
from tui_gateway.universal_model_runtime import (
    main_model_override,
    selected_profile_main_fallbacks,
    selected_profile_main_override,
)


def local_row(model="small"):
    return {
        "id": "custom@abc::small", "model": model, "provider": "custom",
        "source": "ollama-live", "availability": "reachable",
        "endpoint": "http://127.0.0.1:11434/v1",
    }


def test_profile_selection_only_overrides_pinned_main_and_refuses_offline():
    service = Mock()
    service.profile.return_value = {"roles": [], "policy": {}}
    assert selected_profile_main_override(service) is None
    service.resolve.assert_not_called()

    service.profile.return_value = {"roles": [{"role": "main", "mode": "pinned"}], "policy": {}}
    service.resolve.return_value = local_row()
    assert selected_profile_main_override(service)["base_url"] == "http://127.0.0.1:11434/v1"
    service.resolve.return_value = None
    with pytest.raises(ValueError, match="unavailable"):
        selected_profile_main_override(service)


def test_explicit_auto_role_can_select_local_but_never_auto_uses_cloud_when_absent():
    service = Mock()
    service.profile.return_value = {"roles": [{"role": "main", "mode": "auto"}], "policy": {}}
    service.resolve.return_value = local_row()
    assert selected_profile_main_override(service)["model"] == "small"
    service.resolve.return_value = None
    assert selected_profile_main_override(service) is None
    service.profile.return_value["policy"] = {"local_only": True}
    with pytest.raises(ValueError, match="routing policy"):
        selected_profile_main_override(service)


def test_non_ollama_local_proxy_and_unreachable_rows_are_not_routed_with_local_key():
    with pytest.raises(ValueError, match="registered"):
        main_model_override({**local_row(), "source": "openai-compatible-live"})
    with pytest.raises(ValueError, match="unavailable"):
        main_model_override({**local_row(), "availability": "on_disk"})
    with pytest.raises(ValueError, match="registered"):
        main_model_override({**local_row(), "source": "openai-compatible-live", "endpoint": "http://127.0.0.1:8888/v1"})


def test_local_idle_switch_changes_actual_agent_runtime_and_removes_cloud_fallback():
    agent = SimpleNamespace(
        model="old", provider="openrouter", api_key="cloud-secret", base_url="https://api.example",
        switch_model=Mock(), _fallback_chain=[{"provider": "openrouter", "model": "paid"}],
        _fallback_index=1,
    )
    session = {"agent": agent, "running": False}
    with patch.object(server, "_restart_slash_worker"), \
            patch.object(server, "_persist_live_session_runtime"), \
            patch.object(server, "_append_model_switch_marker"), \
            patch.object(server, "_session_info", return_value={}), \
            patch.object(server, "_emit"):
        server._apply_universal_local_switch("sid", session, local_row())
    agent.switch_model.assert_called_once_with(
        new_model="small", new_provider="custom", api_key="local",
        base_url="http://127.0.0.1:11434/v1", api_mode="chat_completions",
    )
    assert agent._fallback_chain == []
    assert session["model_override"]["model"] == "small"
    assert session["model_override"]["api_key"] != "cloud-secret"


def test_live_busy_switch_is_deferred_and_committed_without_mutating_agent():
    class Service:
        _allowed = staticmethod(lambda row, policy, required, context: True)

        def status(self, refresh=False):
            return {"models": [local_row()], "policy": {}, "roles": [], "profiles": []}

        def assign(self, role, mode, model_id, fallback_ids):
            assert (role, mode, model_id) == ("main", "pinned", "custom@abc::small")
            return {"role": role, "mode": mode, "model_id": model_id}

    agent = SimpleNamespace(switch_model=Mock())
    session = {"running": True, "agent": agent}
    with patch.dict(server._sessions, {"sid-test": session}), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=Service()):
        response = server._methods["orchestration.assign"](
            "rid", {"role": "main", "mode": "pinned", "model_id": "custom@abc::small", "session_id": "sid-test"}
        )
    assert response["result"]["deferred"] is True
    assert session["pending_universal_model_switch"] == "custom@abc::small"
    agent.switch_model.assert_not_called()


def test_local_only_policy_blocks_cloud_before_session_switch_or_assignment():
    class Service:
        _allowed = staticmethod(lambda row, policy, required, context: False)

        def status(self, refresh=False):
            return {"models": [{"id": "cloud:a", "provider": "deepseek", "model": "x", "availability": "configured"}],
                    "policy": {"local_only": True}}

        def assign(self, *args):
            raise AssertionError("policy must preflight before mutation")

    with patch("hermes_cli.universal_models.service.ModelService", return_value=Service()), \
            patch.object(server, "_methods", server._methods.copy()) as methods:
        methods["config.set"] = Mock(side_effect=AssertionError("must not execute"))
        response = server._methods["orchestration.assign"](
            "rid", {"role": "main", "mode": "pinned", "model_id": "cloud:a"}
        )
    assert response["error"]["code"] == 4004


def test_new_main_ollama_override_reaches_real_agent_constructor_without_cloud_resolver():
    fake_config = {"model": {"default": "legacy", "provider": "openrouter"}, "agent": {}}
    with patch("tui_gateway.server._load_cfg", return_value=fake_config), \
            patch("tui_gateway.server._get_db", return_value=Mock()), \
            patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"), \
            patch("tui_gateway.server._load_reasoning_config", return_value=None), \
            patch("tui_gateway.server._load_service_tier", return_value=None), \
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None), \
            patch("tui_gateway.server._resolve_runtime_with_fallback") as cloud_resolve, \
            patch("run_agent.AIAgent") as constructor:
        server._make_agent("sid", "key", model_override=main_model_override(local_row()))
    cloud_resolve.assert_not_called()
    assert constructor.call_args.kwargs["provider"] == "custom"
    assert constructor.call_args.kwargs["model"] == "small"
    assert constructor.call_args.kwargs["base_url"] == "http://127.0.0.1:11434/v1"
    assert constructor.call_args.kwargs["api_key"] == "local"
    assert constructor.call_args.kwargs["fallback_model"] == []


def test_main_profile_pin_never_inherits_legacy_cloud_fallbacks():
    service = Mock()
    service.status.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "model_id": "local:one", "fallback_ids": []}],
        "models": [local_row()], "policy": {}, "local_only": True,
    }
    assert selected_profile_main_fallbacks(service) == []
    service.status.return_value["roles"][0]["mode"] = "auto"
    assert selected_profile_main_fallbacks(service) is None


def test_local_only_cloud_fallback_is_rejected_before_live_agent_mutation():
    service = Mock()
    service.status.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "fallback_ids": ["cloud:paid"]}],
        "models": [{"id": "cloud:paid", "provider": "openrouter", "model": "paid",
                    "availability": "configured", "source": "hermes-config"}],
        "policy": {"local_only": True}, "local_only": True,
    }
    service._allowed.return_value = False
    with pytest.raises(ValueError, match="policy"):
        selected_profile_main_fallbacks(service)
    agent = SimpleNamespace(switch_model=Mock())
    with patch("tui_gateway.universal_model_runtime.selected_profile_main_fallbacks",
               side_effect=ValueError("policy invalid")):
        with pytest.raises(ValueError, match="policy invalid"):
            server._apply_universal_local_switch("sid", {"agent": agent}, local_row())
    agent.switch_model.assert_not_called()


def test_gateway_policy_roundtrip_with_real_sqlite(tmp_path):
    from hermes_cli.universal_models.service import ModelService

    def isolated_service():
        return ModelService(home=tmp_path, config={})

    with patch("hermes_cli.universal_models.service.ModelService", side_effect=isolated_service):
        change = server._methods["orchestration.policy"](
            "set", {"action": "set", "policy": {"local_only": True}}
        )
        read = server._methods["orchestration.policy"]("get", {})
        bad = server._methods["orchestration.policy"](
            "invalid", {"action": "set", "policy": {"local_only": "yes"}}
        )
    assert change["result"]["policy"] == {"local_only": True}
    assert read["result"]["policy"] == {"local_only": True}
    assert bad["error"]["code"] == 5073
    assert isolated_service().status()["policy"] == {"local_only": True}


def test_explicit_local_fallback_is_applied_to_live_agent_before_old_cloud_chain():
    fallback = {**local_row("spare"), "id": "local:spare"}
    service = Mock()
    service.status.return_value = {
        "models": [local_row(), fallback], "roles": [], "policy": {"local_only": True},
        "local_only": True,
    }
    service._allowed.return_value = True
    service.assign.return_value = {"role": "main", "mode": "pinned", "model_id": local_row()["id"]}
    agent = SimpleNamespace(
        switch_model=Mock(), _fallback_chain=[{"provider": "openrouter", "model": "paid"}],
        _fallback_index=0,
    )
    session = {"agent": agent, "running": False}
    with patch.dict(server._sessions, {"sid-fallback": session}), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_restart_slash_worker"), \
            patch.object(server, "_persist_live_session_runtime"), \
            patch.object(server, "_append_model_switch_marker"), \
            patch.object(server, "_session_info", return_value={}), \
            patch.object(server, "_emit"):
        response = server._methods["orchestration.assign"](
            "rid", {"role": "main", "mode": "pinned", "model_id": local_row()["id"],
                    "fallback_ids": [fallback["id"]], "session_id": "sid-fallback"}
        )
    assert response["result"]["mode"] == "pinned"
    assert len(agent._fallback_chain) == 1
    assert agent._fallback_chain[0]["model"] == "spare"
    assert agent._fallback_chain[0]["api_key"] == "local"


def test_qualify_rpc_is_bounded_and_never_calls_unverified_or_cloud_models():
    service = Mock()
    service.status.return_value = {"models": [local_row(), {
        "id": "cloud:paid", "model": "paid", "provider": "openrouter",
        "source": "hermes-config", "availability": "configured",
    }]}
    service.qualify.return_value = {"model_id": local_row()["id"], "status": "passed", "inference_tested": True}
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        good = server._methods["orchestration.qualify"](
            "r", {"model_id": local_row()["id"], "max_tokens": 24, "timeout_seconds": 8}
        )
        remote = server._methods["orchestration.qualify"](
            "cloud", {"model_id": "cloud:paid"}
        )
        oversized = server._methods["orchestration.qualify"](
            "oversized", {"model_id": local_row()["id"], "max_tokens": 65536}
        )
    assert good["result"]["status"] == "passed"
    service.qualify.assert_called_once_with(
        local_row()["id"], run_inference=True, timeout_seconds=8.0,
        max_tokens=24, require_loaded=True,
    )
    assert remote["error"]["code"] == 4004
    assert oversized["error"]["code"] == 4002
