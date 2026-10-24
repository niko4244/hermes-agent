"""Saved-role gateway policy revalidation at special-provider wire release.

All expensive providers are synthetic registered models. Real AIAgent routing
and a temporary registry are used, but SDK sends are always in-memory spies.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from tests.tui_gateway.test_universal_turn_pipeline import local_ollama_wire, pipeline
from tui_gateway import server, universal_model_runtime


def _configure_saved_cloud_main(registry, session, monkeypatch, *, provider, model):
    registry._config_override = {
        **dict(registry._config_override or {}),
        "providers": {provider: {"models": [model]}},
    }
    records = [row for row in registry.status()["models"]
               if row["provider"] == provider and row["model"] == model]
    assert len(records) == 1, "configured provider must have a real catalog identity"
    record = records[0]
    assert record["source"] == "hermes-provider"
    monkeypatch.setattr(universal_model_runtime, "_active_registry", lambda: registry)
    registry.assign("main", "pinned", record["id"], [])
    profile = registry.profile("export")
    registry.profile("save", "default", {
        "name": "default", "roles": profile["roles"], "policy": {},
    }, confirm_cloud=True)
    agent = session["agent"]
    agent.model = model
    agent.provider = provider
    agent.requested_provider = provider
    agent.base_url = "https://never-sent.example.invalid/v1"
    agent.api_key = "DUMMY-NOT-A-SECRET"
    session["model_override"] = {
        "_universal_selection": True, "_universal_model_id": record["id"],
        "model": model, "provider": provider,
    }
    return agent


def _revoke_with_cost_cap(registry):
    profile = registry.profile("export")
    registry.profile("save", "default", {
        "name": "default", "roles": profile["roles"],
        "policy": {"local_only": True, "max_output_cost_per_million": 0},
    })


def test_codex_cloud_main_revoked_during_request_client_creation(pipeline, monkeypatch):
    registry, session, _events, local_requests, _local = pipeline
    agent = _configure_saved_cloud_main(
        registry, session, monkeypatch, provider="openai-codex", model="codex-fixture")
    agent.api_mode = "codex_responses"
    clients = []
    requests = []

    def make_client(_reason, **_kwargs):
        clients.append("created")
        _revoke_with_cost_cap(registry)
        # Any valid client-like object suffices; SDK/network never run.
        return SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: requests.append(kw)))

    with patch.object(agent, "_run_codex_stream", side_effect=lambda *_a, **_kw: requests.append("send")):
        token = server._current_runtime_session_record.set(session)
        try:
            with pytest.raises(ValueError, match="(cost|price|policy|local|authoriz)"):
                _dispatch_nonstreaming_api_request(agent, {"model": agent.model}, make_client=make_client)
        finally:
            server._current_runtime_session_record.reset(token)
    assert clients == ["created"]
    assert requests == [], "a revoked saved Codex route reached the Responses transport"
    assert local_requests == []


def test_legacy_codex_without_saved_universal_role_preserves_transport(pipeline):
    _registry, session, _events, local_requests, _local = pipeline
    agent = session["agent"]
    session.pop("model_override", None)
    agent.api_mode = "codex_responses"
    sent = []
    fake = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs)))
    with patch.object(agent, "_run_codex_stream", side_effect=lambda *_a, **_kw: sent.append("legacy-send")):
        token = server._current_runtime_session_record.set(session)
        try:
            _dispatch_nonstreaming_api_request(agent, {"model": "legacy-model"}, make_client=lambda *_a, **_kw: fake)
        finally:
            server._current_runtime_session_record.reset(token)
    assert sent == ["legacy-send"]
    assert local_requests == []


def test_moa_saved_role_revoked_during_facade_resolution(pipeline, monkeypatch):
    registry, session, _events, local_requests, _local = pipeline
    agent = _configure_saved_cloud_main(
        registry, session, monkeypatch, provider="moa", model="preset-fixture")
    agent.api_mode = "chat_completions"
    sent = []
    viewed = []

    class MoaClient:
        @property
        def chat(self):
            viewed.append("resolved")
            if len(viewed) == 1:
                # A concurrent profile RPC becomes visible while the virtual
                # MoA runtime is resolving its fan-out facade.
                _revoke_with_cost_cap(registry)
            return SimpleNamespace(completions=SimpleNamespace(
                prepare=lambda: None, create=lambda **kwargs: sent.append(kwargs)))

    original = agent.client
    agent.client = MoaClient()
    token = server._current_runtime_session_record.set(session)
    try:
        with pytest.raises(ValueError, match="(cost|price|policy|local|authoriz)"):
            _dispatch_nonstreaming_api_request(
                agent, {"model": agent.model, "stream": False},
                make_client=lambda *_a, **_kw: pytest.fail("MoA must use virtual client"))
    finally:
        server._current_runtime_session_record.reset(token)
        agent.client = original
    assert viewed
    assert sent == [], "revoked virtual provider must not fan out to expensive models"
    assert local_requests == []


def test_anthropic_saved_role_revoked_during_client_setup(pipeline, monkeypatch):
    registry, session, _events, requests, _local = pipeline
    agent = _configure_saved_cloud_main(
        registry, session, monkeypatch, provider="anthropic", model="claude-fixture")
    agent.api_mode = "anthropic_messages"
    sent = []

    def make_client(_reason, **_kwargs):
        _revoke_with_cost_cap(registry)
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: sent.append(kw)))

    with patch.object(agent, "_anthropic_messages_create", side_effect=lambda *_a, **_kw: sent.append("send")):
        token = server._current_runtime_session_record.set(session)
        try:
            with pytest.raises(ValueError, match="(cost|price|policy|local|authoriz)"):
                _dispatch_nonstreaming_api_request(
                    agent, {"model": agent.model}, make_client=make_client)
        finally:
            server._current_runtime_session_record.reset(token)
    assert sent == []
    assert requests == []


def test_bedrock_saved_role_rejected_at_dispatch_if_policy_revoked(pipeline, monkeypatch):
    registry, session, _events, requests, _local = pipeline
    agent = _configure_saved_cloud_main(
        registry, session, monkeypatch, provider="bedrock", model="bedrock-fixture")
    agent.api_mode = "bedrock_converse"
    sent = []
    _revoke_with_cost_cap(registry)
    with patch("agent.chat_completion_helpers._bedrock_converse_call",
               side_effect=lambda *_a, **_kw: sent.append("bedrock-send")):
        token = server._current_runtime_session_record.set(session)
        try:
            with pytest.raises(ValueError, match="(cost|price|policy|local|authoriz)"):
                _dispatch_nonstreaming_api_request(
                    agent, {"modelId": agent.model},
                    make_client=lambda *_a, **_kw: pytest.fail("Bedrock uses SDK, not an OpenAI client"))
        finally:
            server._current_runtime_session_record.reset(token)
    assert sent == []
    assert requests == []
