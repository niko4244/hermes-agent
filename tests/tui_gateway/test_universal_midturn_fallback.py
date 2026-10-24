"""Saved universal failover must honor a profile revoked DURING inference.

The primary failure and the would-be paid provider are hermetic: no hosted API
can be reached, and the real gateway/AIAgent turn runs under a temporary home.
"""

import json
import time
from unittest.mock import patch

import httpx
import pytest
from openai import NotFoundError

from agent.error_classifier import FailoverReason
from tests.tui_gateway.test_universal_turn_pipeline import local_ollama_wire, pipeline
from tui_gateway import server


def _save_policy(registry, policy):
    current = registry.profile("export")
    return registry.profile("save", "default", {
        "name": "default", "roles": current["roles"], "policy": policy,
    })


def _install_mock_cloud_fallback(registry, local):
    """Give the saved role a discovered billable-like provider, without a real credential."""
    paid = {
        **local,
        "id": "fixture:cloud-standin",
        "model": "paid-standin",
        "provider": "openrouter",
        "source": "configured",
        "availability": "configured",
        "endpoint": "https://no-network-allowed.invalid/v1",
        "loaded": False,
    }
    with registry._connect() as conn:
        observed = json.loads(conn.execute(
            "SELECT payload FROM observations WHERE name='local-models'").fetchone()[0])
        observed.append(paid)
        conn.execute("UPDATE observations SET payload=?,measured_at=? WHERE name='local-models'",
                     (json.dumps(observed), time.time()))
    assert any(row["id"] == paid["id"] for row in registry.status()["models"])
    registry.assign("main", "pinned", local["id"], [paid["id"]])
    _save_policy(registry, {})
    return paid


def _no_cloud_creds(_record, _parent):
    # Routes to a deliberately non-resolving host; route_binding is also
    # intercepted, so neither this hostname nor any paid service is contacted.
    return {"model": "paid-standin", "provider": "openrouter", "api_key": "DUMMY-NEVER-SENT",
            "base_url": "https://no-network-allowed.invalid/v1", "api_mode": "chat_completions"}


@pytest.mark.parametrize("revocation", ["local_only", "cost_cap", "removed_fallback_id"])
def test_midturn_profile_revocation_never_binds_cloud_fallback(pipeline, monkeypatch, revocation):
    registry, session, events, requests, local = pipeline
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    paid = _install_mock_cloud_fallback(registry, local)
    agent = session["agent"]
    agent._api_max_retries = 1
    seen = []
    attempts = []
    original_activate = agent._try_activate_fallback

    def fail_then_revoke(*args, **kwargs):
        # A real saved role's paid fallback was approved when the turn began.
        assert agent._fallback_chain and agent._fallback_chain[0]["model"] == paid["model"]
        seen.append("provider-failure")
        if revocation == "removed_fallback_id":
            registry.assign("main", "pinned", local["id"], [])
        else:
            _save_policy(registry, {"local_only": True} if revocation == "local_only"
                         else {"max_output_cost_per_million": 0})
        return original_activate(*args, **kwargs)

    def forbid_paid_binding(_agent, entry, provider, model):
        attempts.append((entry, provider, model))
        # no client creation, socket, or hosted inference

    request = httpx.Request("POST", "http://127.0.0.1/local-failure")
    failure = NotFoundError("primary model unavailable", response=httpx.Response(
        404, request=request, json={"error": {"message": "primary model unavailable"}}), body=None)

    # The real AIAgent runs its normal classifier/retry/fallback code through
    # server._run_prompt_submit. Only its primary I/O is replaced with a 404.
    with patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_no_cloud_creds), \
            patch("agent.route_binding.bind_route_entry", side_effect=forbid_paid_binding), \
            patch("agent.chat_completion_helpers._StreamingCall._open_chat_stream", side_effect=failure), \
            patch.object(agent, "_try_activate_fallback", side_effect=fail_then_revoke):
        assert server._run_prompt_submit("failover-rid", "turn-sid", session, "Trigger failover")

    assert seen, "the error must genuinely reach AIAgent's fallback activator"
    assert attempts == [], "a revoked provider may not reach the client-binding path"
    assert requests == [], "no paid or local HTTP request on a mocked primary failure"
    assert session["running"] is False
    assert [event for event, _payload in events if event == "message.complete"]


def test_legacy_agents_without_saved_roles_retain_fallback_behavior(pipeline):
    _registry, session, _events, _requests, _row = pipeline
    agent = session["agent"]
    session.pop("model_override", None)
    agent._fallback_chain = [{"provider": "openrouter", "model": "legacy-paid-fallback"}]
    agent._fallback_index = 0
    attempted = []

    def intercept(_agent, entry, *_args):
        attempted.append(entry["model"])

    with patch("agent.route_binding.bind_route_entry", side_effect=intercept):
        token = server._current_runtime_session_record.set(session)
        try:
            assert agent._try_activate_fallback(reason=FailoverReason.model_not_found) is False
        finally:
            server._current_runtime_session_record.reset(token)
    assert attempted == ["legacy-paid-fallback"]
