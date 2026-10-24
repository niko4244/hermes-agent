"""Universal main-agent authorization after fallback binding and before its wire.

All would-be paid requests terminate at an in-memory fake OpenAI SDK client;
the test never creates a remote provider client or makes a cloud request.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _StreamingCall, _dispatch_nonstreaming_api_request
from agent.error_classifier import FailoverReason
from tests.tui_gateway.test_universal_midturn_fallback import (
    _install_mock_cloud_fallback, _no_cloud_creds, _save_policy,
)
from tests.tui_gateway.test_universal_turn_pipeline import local_ollama_wire, pipeline
from tui_gateway import server


def _intercept_paid_wire(agent, sent):
    """Return a fake per-request client, never a real HTTP client."""
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: sent.append(kwargs))))
    return patch.object(agent, "_create_request_openai_client", return_value=fake)


@pytest.mark.parametrize("revocation", [
    "local_only", "cost_cap", "remove_fallback", "activate_new_profile",
])
def test_revoked_after_fallback_bind_cannot_open_paid_stream(pipeline, revocation, monkeypatch):
    registry, session, _events, local_requests, local = pipeline
    paid = _install_mock_cloud_fallback(registry, local, monkeypatch)
    agent = session["agent"]
    original = (agent.model, agent.provider)
    binding = []
    sent = []

    def bind_then_revoke(target, entry, provider, model):
        assert target is agent
        assert entry["model"] == paid["model"]
        binding.append((provider, model))
        target.model, target.provider, target.requested_provider = model, provider, provider
        target.base_url = "https://no-network-allowed.invalid/v1"
        target.api_key = "DUMMY-NEVER-SENT"
        target._fallback_activated = True
        if revocation == "local_only":
            _save_policy(registry, {"local_only": True})
        elif revocation == "cost_cap":
            _save_policy(registry, {"max_output_cost_per_million": 0})
        elif revocation == "remove_fallback":
            registry.assign("main", "pinned", local["id"], [])
        else:
            registry.profile("save", "new-selection", {
                "name": "new-selection", "roles": [{"role": "main", "mode": "pinned",
                                                   "model_id": local["id"], "fallback_ids": []}],
                "policy": {"local_only": True},
            })
            registry.profile("activate", "new-selection")
        return original

    with patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_no_cloud_creds), \
            patch("agent.route_binding.bind_route_entry", side_effect=bind_then_revoke), \
            patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"), \
            patch("agent.chat_completion_helpers._buffer_fallback_notice"), \
            patch("agent.native_compaction.resolve_native_compaction_capabilities", return_value={}), \
            _intercept_paid_wire(agent, sent):
        token = server._current_runtime_session_record.set(session)
        try:
            # This fixture constructed the AIAgent before installing the test
            # assignment. Reconcile the saved chain as prompt_turn does before
            # invoking a real turn; otherwise there is no fallback to test.
            from tui_gateway.universal_model_runtime import selected_profile_main_fallbacks
            agent._fallback_chain = selected_profile_main_fallbacks()
            agent._fallback_index = 0
            assert [row["model"] for row in agent._fallback_chain] == [paid["model"]]
            # The existing *pre-bind* guard authorizes this fallback while the
            # old profile is still active. The binding then revokes it.
            assert agent._try_activate_fallback(reason=FailoverReason.model_not_found) is True
            assert binding == [("openrouter", "paid-standin")]
            assert agent._provider_fallback_active is True
            request = _StreamingCall(agent, {"model": "paid-standin"}, None)
            with pytest.raises(ValueError, match="(profile|policy|local|cost|authoriz|price)"):
                request._open_chat_stream({"model": "paid-standin", "stream": True})
        finally:
            server._current_runtime_session_record.reset(token)

    assert sent == [], "a revoked route reached the request-client network boundary"
    assert local_requests == []


def test_non_universal_legacy_gateway_agent_keeps_unmodified_wire(pipeline):
    _registry, session, _events, local_requests, _local = pipeline
    agent = session["agent"]
    session.pop("model_override", None)
    agent.provider, agent.model = "openrouter", "legacy-cloud"
    agent.base_url = "https://no-network-allowed.invalid/v1"
    sent = []
    with _intercept_paid_wire(agent, sent), patch(
        "agent.chat_completion_helpers.bypass_chat_sdk_request_transform",
        side_effect=lambda kwargs, _client: kwargs,
    ):
        token = server._current_runtime_session_record.set(session)
        try:
            _StreamingCall(agent, {"model": agent.model}, None)._open_chat_stream(
                {"model": agent.model, "stream": True})
        finally:
            server._current_runtime_session_record.reset(token)
    assert len(sent) == 1
    assert local_requests == []


def test_nonstreaming_fallback_rechecked_after_profile_change(pipeline, monkeypatch):
    registry, session, _events, requests, local = pipeline
    agent = session["agent"]
    _install_mock_cloud_fallback(registry, local, monkeypatch)
    agent.model, agent.provider, agent.base_url = (
        "paid-standin", "openrouter", "https://no-network-allowed.invalid/v1")
    agent.api_key = "DUMMY-NEVER-SENT"
    agent._fallback_activated = True
    agent._provider_fallback_active = True
    registry.assign("main", "pinned", local["id"], [])
    attempted = []

    def fake_client(*_args, **_kwargs):
        attempted.append("provider-client-created")
        return SimpleNamespace()

    token = server._current_runtime_session_record.set(session)
    try:
        with pytest.raises(ValueError, match="(profile|authoriz|policy|fallback)"):
            _dispatch_nonstreaming_api_request(
                agent, {"model": "paid-standin"}, make_client=fake_client)
    finally:
        server._current_runtime_session_record.reset(token)
    assert attempted == []
    assert requests == []


def test_gateway_scope_does_not_authorize_an_unrelated_agent(pipeline):
    _registry, session, _events, requests, _local = pipeline
    original_agent = session["agent"]
    different_agent = SimpleNamespace(model="legacy-model", provider="openrouter",
                                      base_url="https://no-network-allowed.invalid/v1")
    session["agent"] = different_agent
    sent = []
    try:
        with _intercept_paid_wire(original_agent, sent), patch(
            "agent.chat_completion_helpers.bypass_chat_sdk_request_transform",
            side_effect=lambda kwargs, _client: kwargs,
        ):
            token = server._current_runtime_session_record.set(session)
            try:
                # The ContextVar is not enough: the gateway session must own
                # this precise agent object, not any agent on this thread.
                _StreamingCall(original_agent, {"model": "unrelated"}, None)._open_chat_stream(
                    {"model": "unrelated", "stream": True})
            finally:
                server._current_runtime_session_record.reset(token)
    finally:
        session["agent"] = original_agent
    assert len(sent) == 1
    assert requests == []


def test_profile_revoked_during_request_client_construction_before_socket(pipeline):
    registry, session, _events, requests, _local = pipeline
    agent = session["agent"]
    sent = []
    constructed = []
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: sent.append(kwargs))))

    def create_request_client(**_kwargs):
        constructed.append(True)
        # Deliberate race after the first preflight but before SDK.create.
        _save_policy(registry, {"local_only": True, "max_output_cost_per_million": 0})
        return fake

    with patch.object(agent, "_create_request_openai_client", side_effect=create_request_client), \
            patch("agent.chat_completion_helpers.bypass_chat_sdk_request_transform",
                  side_effect=lambda kwargs, _client: kwargs):
        token = server._current_runtime_session_record.set(session)
        try:
            with pytest.raises(ValueError, match="(cost|price|profile|policy)"):
                _StreamingCall(agent, {"model": agent.model}, None)._open_chat_stream(
                    {"model": agent.model, "stream": True})
        finally:
            server._current_runtime_session_record.reset(token)
    assert constructed == [True]
    assert sent == []
    assert requests == []
