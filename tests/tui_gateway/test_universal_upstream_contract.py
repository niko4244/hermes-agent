"""Behavioral checks at the current Hermes gateway's profile and turn seams."""

from contextlib import contextmanager
from unittest.mock import Mock, patch

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
        result = server._methods["orchestration.assign"](
            "r", {"role": "main", "mode": "pinned", "model_id": CLOUD["id"]})
    assert result["result"]["deferred"] is True
    service.assign.assert_called_once()


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
