"""Queued universal main-model switches remain an inference admission boundary."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import time

import pytest

from tui_gateway import server
from tui_gateway.universal_model_runtime import enforce_main_turn_policy


LOCAL = {
    "id": "custom@local::small", "model": "small", "provider": "custom",
    "source": "ollama-live", "availability": "reachable",
    "endpoint": "http://127.0.0.1:11434/v1",
}
CLOUD = {
    "id": "openrouter::candidate", "model": "candidate", "provider": "openrouter",
    "source": "hermes-config", "availability": "configured",
}


def _service(record, *, selected_id=None, allowed=True):
    service = Mock()
    service.profile.return_value = {
        "roles": [{"role": "main", "mode": "pinned",
                   "model_id": selected_id or record["id"], "fallback_ids": []}],
        "policy": {},
    }
    service.status.return_value = {"models": [record], "policy": {}}
    service._allowed.return_value = allowed
    return service


@pytest.mark.parametrize("failure", ["profile_changed", "removed", "denied", "unconfirmed"])
def test_failed_queued_assignment_cannot_disappear_before_subsequent_turn(failure):
    """No second turn may reach the old billable runtime after a failed live pin."""
    record = CLOUD if failure == "unconfirmed" else LOCAL
    service = _service(record)
    if failure == "profile_changed":
        service.profile.return_value["roles"][0]["model_id"] = "custom@other::new"
    if failure == "removed":
        service.status.return_value["models"] = []
    if failure == "denied":
        service._allowed.return_value = False

    agent = SimpleNamespace(
        model="old-paid", provider="openrouter", base_url="https://paid.example/v1",
        run_conversation=Mock(),
    )
    session = {
        "agent": agent,
        "model_override": {"model": "old-paid", "provider": "openrouter"},
        "pending_universal_model_switch": record["id"],
    }
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_apply_model_switch", return_value={"confirm_required": True}) as switch:
        for _ in range(2):
            with pytest.raises(ValueError):
                server._apply_pending_model_switch("sid", session)
                enforce_main_turn_policy(agent, session, service)
                agent.run_conversation("unexpected billing")

    assert session["pending_universal_model_switch"] == record["id"]
    agent.run_conversation.assert_not_called()
    assert switch.call_count == (2 if failure == "unconfirmed" else 0)


def test_failed_native_runtime_switch_is_retryable_without_running_previous_route():
    service = _service(LOCAL)
    agent = SimpleNamespace(model="old-paid", provider="openrouter", run_conversation=Mock())
    session = {"agent": agent, "pending_universal_model_switch": LOCAL["id"]}
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_apply_universal_local_switch",
                         side_effect=ValueError("native runtime unavailable")) as switch:
        for _ in range(2):
            with pytest.raises(ValueError, match="native runtime unavailable"):
                server._apply_pending_model_switch("sid", session)
    assert switch.call_count == 2
    assert session["pending_universal_model_switch"] == LOCAL["id"]
    agent.run_conversation.assert_not_called()


def test_successful_revalidated_native_switch_consumes_queued_assignment():
    service = _service(LOCAL)
    agent = SimpleNamespace(model="old-paid", provider="openrouter")
    session = {"agent": agent, "pending_universal_model_switch": LOCAL["id"]}

    def switch(*args, **kwargs):
        session["model_override"] = {"model": LOCAL["model"], "_universal_selection": True}

    with patch("hermes_cli.universal_models.service.ModelService", return_value=service), \
            patch.object(server, "_apply_universal_local_switch", side_effect=switch):
        server._apply_pending_model_switch("sid", session)
    assert "pending_universal_model_switch" not in session
    assert session["model_override"]["model"] == LOCAL["model"]


def test_universal_fallback_revocation_prunes_future_hops_during_active_cooldown():
    """Cooldown freezes provider restoration, not the operator's authorization."""
    current = {"provider": "anthropic", "model": "approved"}
    revoked = {"provider": "openrouter", "model": "removed-paid"}
    agent = SimpleNamespace(
        _fallback_activated=True, _provider_fallback_active=True,
        _rate_limited_until=time.monotonic() + 600,
        _fallback_chain=[current, revoked], _fallback_model=current,
        _fallback_index=1,
    )
    session = {"agent": agent, "model_override": {"_universal_selection": True}}
    with patch("tui_gateway.universal_model_runtime.selected_profile_main_fallbacks",
               return_value=[current]):
        server._sync_agent_fallback_with_config("sid", session)
    assert agent._fallback_chain == [current]
    assert agent._fallback_index == 1
    assert agent._fallback_index >= len(agent._fallback_chain)


def test_universal_fallback_refresh_can_add_approved_chain_on_primary_runtime():
    approved = {"provider": "anthropic", "model": "approved"}
    agent = SimpleNamespace(
        _fallback_activated=False, _rate_limited_until=0,
        _fallback_chain=[{"provider": "openrouter", "model": "removed-paid"}],
        _fallback_index=1,
    )
    session = {"agent": agent, "model_override": {"_universal_selection": True}}
    with patch("tui_gateway.universal_model_runtime.selected_profile_main_fallbacks",
               return_value=[approved]):
        server._sync_agent_fallback_with_config("sid", session)
    assert agent._fallback_chain == [approved]
    assert agent._fallback_index == 0
