"""Behavioral checks at the current Hermes gateway's profile and turn seams."""

from contextlib import contextmanager
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.contracts.orchestration import (
    OrchestrationPolicyResult, OrchestrationProfileResult, OrchestrationStatusResult,
)


LOCAL = {
    "id": "custom@abc::small", "model": "small", "provider": "custom",
    "source": "ollama-live", "availability": "reachable",
    "endpoint": "http://127.0.0.1:11434/v1",
}
CLOUD = {
    "id": "openrouter::example", "model": "example", "provider": "openrouter",
    "source": "hermes-config", "availability": "configured",
}


def test_real_registry_status_and_profiles_validate_against_rpc_contract(tmp_path):
    from hermes_cli.universal_models.service import ModelService

    service = ModelService(home=tmp_path, config={}, local_endpoints=[],
                           litellm_paths=[], model_paths=[])
    OrchestrationStatusResult.model_validate(service.status())
    OrchestrationProfileResult.model_validate(service.profile("list"))
    exported = service.profile("export")
    OrchestrationProfileResult.model_validate(exported)
    OrchestrationPolicyResult.model_validate({
        "policy": exported["policy"], "active_profile": exported["name"],
    })


def test_sessionless_cloud_assignment_only_saves_profile_and_never_calls_config_set():
    service = Mock()
    service.status.return_value = {"models": [CLOUD], "policy": {}}
    service._allowed.return_value = True
    service.assign.return_value = {
        "role": "main", "mode": "pinned", "model_id": CLOUD["id"], "fallback_ids": [],
    }
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_apply_model_switch", side_effect=AssertionError("unexpected switch")), \
            patch.dict(server._methods, {"config.set": Mock(side_effect=AssertionError("global mutation"))}):
        deferred = server._methods["orchestration.assign"](
            "r", {"role": "main", "mode": "pinned", "model_id": CLOUD["id"]})
        assert deferred["result"]["confirm_required"] is True
        service.assign.assert_not_called()
        result = server._methods["orchestration.assign"](
            "r", {"role": "main", "mode": "pinned", "model_id": CLOUD["id"],
                  "confirm_cloud": True})
    assert result["result"]["deferred"] is True
    service.assign.assert_called_once()


@pytest.mark.parametrize("role,mode,selected,fallback_ids", [
    ("main", "pinned", CLOUD["id"], []),
    ("delegation.leaf", "pinned", CLOUD["id"], []),
    ("delegation.orchestrator", "auto", None, [CLOUD["id"]]),
    ("main", "pinned", LOCAL["id"], [CLOUD["id"]]),
])
def test_remote_primary_or_fallback_requires_explicit_cost_consent_before_persistence(
    role, mode, selected, fallback_ids,
):
    service = Mock()
    service.status.return_value = {"models": [LOCAL, CLOUD], "policy": {}}
    service._allowed.return_value = True
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        response = server._methods["orchestration.assign"]("r", {
            "role": role, "mode": mode, "model_id": selected, "fallback_ids": fallback_ids,
        })
    assert response["result"]["confirm_required"] is True
    assert "charges" in response["result"]["warning"]
    service.assign.assert_not_called()


def test_explicit_remote_role_consent_permits_profile_only_save():
    service = Mock()
    service.status.return_value = {"models": [CLOUD], "policy": {}}
    service.assign.return_value = {
        "role": "delegation.leaf", "mode": "pinned", "model_id": CLOUD["id"],
        "fallback_ids": [],
    }
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        response = server._methods["orchestration.assign"]("r", {
            "role": "delegation.leaf", "mode": "pinned",
            "model_id": CLOUD["id"], "confirm_cloud": True,
        })
    assert response["result"]["model_id"] == CLOUD["id"]
    service.assign.assert_called_once()


def test_invalid_cloud_consent_type_fails_closed():
    with patch("hermes_cli.universal_models.service.ModelService") as service:
        response = server._methods["orchestration.assign"]("r", {
            "role": "delegation.leaf", "mode": "pinned",
            "model_id": CLOUD["id"], "confirm_cloud": "true",
        })
    assert response["error"]["code"] == 4002
    service.assert_not_called()


def test_queued_local_switch_is_preflighted_at_real_turn_boundary():
    service = Mock()
    service.profile.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "model_id": LOCAL["id"],
                   "fallback_ids": []}], "policy": {"local_only": True},
    }
    service.status.return_value = {
        "models": [LOCAL], "policy": {"local_only": True},
    }
    service._allowed.return_value = True
    session = {"agent": Mock(), "pending_universal_model_switch": LOCAL["id"]}
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_apply_universal_local_switch") as switch:
        server._apply_pending_model_switch("sid", session)
    switch.assert_called_once_with("sid", session, LOCAL, approved_fallbacks=[])
    assert "pending_universal_model_switch" not in session


def test_disallowed_queued_switch_refuses_turn_without_touching_runtime():
    service = Mock()
    service.profile.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "model_id": LOCAL["id"],
                   "fallback_ids": []}], "policy": {"local_only": True},
    }
    service.status.return_value = {"models": [LOCAL], "policy": {"local_only": True}}
    service._allowed.return_value = False
    agent = Mock()
    session = {"agent": agent, "pending_universal_model_switch": LOCAL["id"]}
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        with pytest.raises(ValueError, match="no longer available"):
            server._apply_pending_model_switch("sid", session)
    agent.switch_model.assert_not_called()


def test_orchestration_status_binds_secondary_session_scope_not_launch():
    service = Mock()
    service.status.return_value = {"models": [], "roles": [], "profiles": [],
                                  "policy": {}, "active_profile": "default"}
    seen = []

    @contextmanager
    def fake_scope(session):
        seen.append(session["profile_home"])
        yield

    with patch.dict(server._sessions, {"secondary": {"profile_home": "C:/profile-two"}}), \
            patch.object(server, "_session_profile_runtime_scope", side_effect=fake_scope), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        result = server._methods["orchestration.status"]("r", {"session_id": "secondary"})
    assert "result" in result
    assert seen == ["C:/profile-two"]


@pytest.mark.parametrize("method,params", [
    ("orchestration.status", {}),
    ("orchestration.refresh", {}),
    ("orchestration.assign", {"role": "main", "mode": "auto"}),
    ("orchestration.profile", {"action": "save", "name": "private-profile"}),
    ("orchestration.policy", {"action": "set", "policy": {"local_only": False}}),
    ("orchestration.qualify", {"model_id": LOCAL["id"]}),
])
def test_stale_session_cannot_fall_back_to_launch_profile(method, params):
    service = Mock()
    with patch.dict(server._sessions, {}, clear=True), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        result = server._methods[method]("rid", {**params, "session_id": "expired-sid"})
    assert "error" in result
    assert result["error"]["code"] == 4001
    service.assert_not_called()


def test_cannot_mix_session_authority_with_another_named_profile():
    with patch.dict(server._sessions, {"active": {"profile_home": "C:/private"}}):
        result = server._methods["orchestration.policy"](
            "rid", {"session_id": "active", "profile": "other", "action": "get"})
    assert result["error"]["code"] == 4001


def test_explicit_empty_session_id_does_not_count_as_sessionless_authority():
    with patch("hermes_cli.universal_models.service.ModelService") as service:
        result = server._methods["orchestration.policy"](
            "rid", {"session_id": "", "action": "get"})
    assert result["error"]["code"] == 4001
    service.assert_not_called()


def test_idle_native_ollama_assignment_uses_real_agent_switch_not_global_config():
    service = Mock()
    service.status.return_value = {"models": [LOCAL], "policy": {}}
    service._allowed.return_value = True
    service.assign.return_value = {
        "role": "main", "mode": "pinned", "model_id": LOCAL["id"], "fallback_ids": [],
    }
    agent = SimpleNamespace(
        model="previous", provider="openrouter", base_url="https://example.invalid/v1",
        api_key="prior", api_mode="chat_completions", _fallback_activated=False,
        _rate_limited_until=0, _fallback_chain=[], switch_model=Mock(),
    )
    session = {"agent": agent, "running": False, "session_key": "test"}
    with patch.dict(server._sessions, {"sid": session}), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_restart_slash_worker"), \
            patch.object(server, "_persist_live_session_runtime"), \
            patch.object(server, "_append_model_switch_marker"), \
            patch.object(server, "_emit_session_info"), \
            patch.dict(server._methods, {"config.set": Mock(side_effect=AssertionError("global mutation"))}):
        result = server._methods["orchestration.assign"]("r", {
            "session_id": "sid", "role": "main", "mode": "pinned", "model_id": LOCAL["id"],
        })
    assert result["result"]["deferred"] is False
    agent.switch_model.assert_called_once()
    assert agent.switch_model.call_args.kwargs["new_model"] == LOCAL["model"]
    assert agent.switch_model.call_args.kwargs["api_key"] == "local"
    assert session["model_override"]["_universal_local"] is True
    service.assign.assert_called_once()


def test_failed_registry_write_rolls_back_idle_agent_route_and_fallbacks():
    service = Mock()
    service.status.return_value = {"models": [LOCAL], "policy": {}}
    service._allowed.return_value = True
    service.assign.side_effect = OSError("database unavailable")

    agent = SimpleNamespace(
        model="old", provider="openrouter", base_url="https://example.invalid/v1",
        api_key="prior", api_mode="chat_completions",
        _fallback_activated=False, _rate_limited_until=0,
        _fallback_chain=[{"provider": "openrouter", "model": "old-fallback"}],
    )

    def fake_switch_model(**kwargs):
        for attribute, key in (("model", "new_model"), ("provider", "new_provider"),
                               ("api_key", "api_key"), ("base_url", "base_url"),
                               ("api_mode", "api_mode")):
            setattr(agent, attribute, kwargs[key])

    agent.switch_model = Mock(side_effect=fake_switch_model)
    old_override = {"model": "old", "provider": "openrouter"}
    session = {"agent": agent, "running": False, "model_override": old_override}

    with patch.dict(server._sessions, {"sid": session}), \
            patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_restart_slash_worker"), \
            patch.object(server, "_persist_live_session_runtime"), \
            patch.object(server, "_append_model_switch_marker"), \
            patch.object(server, "_emit_session_info"):
        result = server._methods["orchestration.assign"](
            "r", {"session_id": "sid", "role": "main", "mode": "pinned",
                  "model_id": LOCAL["id"]})
    assert result["error"]["code"] == 5071
    assert agent.switch_model.call_count == 2
    assert (agent.model, agent.provider, agent.api_key) == ("old", "openrouter", "prior")
    assert session["model_override"] == old_override
    assert agent._fallback_chain == [{"provider": "openrouter", "model": "old-fallback"}]
    assert not session.get("_universal_model_rollback_failed")


def test_unrecoverable_failed_assignment_never_reaches_inference():
    from tui_gateway.universal_model_runtime import enforce_main_turn_policy

    agent = SimpleNamespace(model="unexpected", provider="openrouter")
    with pytest.raises(ValueError, match="rollback failed"):
        enforce_main_turn_policy(agent, {"_universal_model_rollback_failed": True})


def test_saved_named_custom_provider_resolves_by_profile_key_not_inventory_namespace():
    selected = {"model": "model-alias", "provider": "custom:litellm",
                "_universal_selection": True, "_universal_model_id": "custom:litellm::model-alias"}
    resolved = SimpleNamespace(
        used_fallback=False, selected_model=None,
        runtime={"provider": "custom", "base_url": "https://proxy.example.test/v1",
                 "api_key": "secret", "api_mode": "chat_completions"},
    )
    with patch("tools.delegation_model_router._registered_custom_provider",
               return_value=("litellm", "https://proxy.example.test/v1")), \
            patch.object(server, "_resolve_runtime_with_fallback", return_value=resolved) as route:
        model, runtime = server._resolve_agent_model_runtime(selected, None)
    assert model == "model-alias"
    assert runtime["provider"] == "custom"
    assert route.call_args.args[0]["requested"] == "litellm"


def test_saved_named_custom_provider_rejects_wrong_provider_even_without_fallback_flag():
    selected = {"model": "model-alias", "provider": "custom:litellm",
                "_universal_selection": True}
    mismatched = SimpleNamespace(
        used_fallback=False, selected_model=None,
        runtime={"provider": "openrouter", "base_url": "https://paid.example.test/v1"},
    )
    with patch("tools.delegation_model_router._registered_custom_provider",
               return_value=("litellm", "https://proxy.example.test/v1")), \
            patch.object(server, "_resolve_runtime_with_fallback", return_value=mismatched):
        with pytest.raises(ValueError, match="different runtime"):
            server._resolve_agent_model_runtime(selected, None)


def test_deferred_build_takes_saved_main_model_without_overriding_session_pick():
    service = Mock()
    service.profile.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "model_id": LOCAL["id"]}],
        "policy": {},
    }
    service.resolve.return_value = LOCAL
    current = {"session_key": "test-session"}
    with patch("tui_gateway.universal_model_runtime._active_registry", return_value=service):
        first = server._deferred_build_agent_kwargs(current, Mock())
        assert first["model_override"]["model"] == "small"
        assert current["model_override"]["_universal_selection"] is True
        current["model_override"] = {"model": "explicit", "provider": "openrouter"}
        second = server._deferred_build_agent_kwargs(current, Mock())
    assert second["model_override"]["model"] == "explicit"


def test_unconfigured_main_turn_creates_no_registry_sidecar(tmp_path):
    from tui_gateway.universal_model_runtime import (
        enforce_main_turn_policy, selected_profile_main_override,
    )

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        assert selected_profile_main_override() is None
        enforce_main_turn_policy(Mock(), {"model_override": None})
    assert not (tmp_path / "universal_models.sqlite3").exists()
