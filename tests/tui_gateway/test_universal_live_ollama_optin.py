"""Opt-in real local-generation smoke through the saved-role gateway pipeline.

Run with HERMES_LIVE_OLLAMA_SMOKE=1. Never pulls or loads a model: the target
must already be resident in Ollama /api/ps. Uses a temporary Hermes home,
no cloud credentials, and the regular AIAgent prompt turn pipeline.
"""

import json
import os
from urllib.request import urlopen

import pytest

from tests.tui_gateway.test_universal_turn_pipeline import pipeline
from tui_gateway import server


@pytest.fixture
def local_ollama_wire(monkeypatch):
    """Override the hermetic wire fixture with an *already-loaded* Ollama model."""
    endpoint = "http://127.0.0.1:11434"
    requested_model = os.environ.get("HERMES_LIVE_OLLAMA_MODEL", "llama3.2:3b")
    try:
        with urlopen(f"{endpoint}/api/ps", timeout=2) as response:
            rows = json.load(response).get("models", [])
    except (OSError, ValueError) as exc:
        pytest.skip(f"Local Ollama /api/ps unavailable: {exc}")
    resident = next((row.get("name") for row in rows
                     if row.get("name") == requested_model), None)
    if not resident:
        pytest.skip(f"{requested_model} must already be resident; never load a model for a smoke test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    yield endpoint, [], resident


@pytest.mark.skipif(os.getenv("HERMES_LIVE_OLLAMA_SMOKE") != "1",
                    reason="opt-in only: real local generation")
def test_saved_main_role_executes_real_resident_ollama_turn(pipeline):
    """Unlike hermetic emulator tests, this requires an actual model response."""
    _service, session, events, _emulator_requests, row = pipeline
    assert row["source"] == "ollama-live"
    assert row["loaded"] is True
    assert session["model_override"]["_universal_local"] is True
    assert session["agent"]._fallback_chain == []
    assert server._run_prompt_submit(
        "live-local-rid", "turn-sid", session,
        "Reply in one short sentence about a bright blue sky. No tools.",
    ) is True
    complete = [data for kind, data in events if kind == "message.complete"]
    deltas = [data for kind, data in events if kind == "message.delta"]
    assert len(complete) == 1
    assert complete[0]["status"] == "complete", complete[0]
    assert complete[0]["text"].strip(), "actual model must generate nonempty text"
    assert deltas, "actual streaming response must reach the gateway"
    assert "TURN_PIPELINE_OK" not in complete[0]["text"]
    assert session["running"] is False
