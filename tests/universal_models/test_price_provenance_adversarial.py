"""Unverified observations cannot grant remote-price authority or cloud consent."""

import json
import sqlite3
import time

import pytest

from hermes_cli.universal_models.discovery import _row
from hermes_cli.universal_models.service import CloudConsentRequired, ModelService


CLOUD_ID = "openrouter::premium"
CONFIG = {"providers": {"openrouter": {"models": ["premium"]}}}
ORIGIN = "http://127.0.0.1:11434/v1"


def _service(tmp_path):
    return ModelService(home=tmp_path, config=CONFIG, local_endpoints=[],
                        model_paths=[], litellm_paths=[])


def _observe(service, name, rows):
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO observations(name,payload,measured_at) VALUES (?,?,?) "
                     "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, measured_at=excluded.measured_at",
                     (name, json.dumps(rows), time.time()))


def _main_cloud_profile(policy=None):
    return {"roles": [{"role": "main", "mode": "pinned", "model_id": CLOUD_ID,
                       "fallback_ids": []}], "policy": policy or {}}


@pytest.mark.parametrize("price", [0, 0.01])
def test_forged_recent_observation_cannot_make_cloud_price_cap_pass(tmp_path, price):
    service = _service(tmp_path)
    service.assign("main", "pinned", CLOUD_ID)  # already operator-approved legacy assignment
    service.profile("save", "default", _main_cloud_profile({"max_input_cost_per_million": 0.05}))
    assert service.resolve("main") is None  # no independently verified pricing

    forged = {
        "id": CLOUD_ID, "provider": "openrouter", "model": "premium",
        "availability": "reachable", "source": "ollama-live", "endpoint": ORIGIN,
        "max_input_cost_per_million": price, "max_output_cost_per_million": 0,
    }
    _observe(service, "local-models", [forged])
    assert service.resolve("main") is None
    assert not any(row.get("id") == CLOUD_ID and row.get("availability") == "reachable"
                   for row in service.status()["models"])


def test_forged_historical_native_id_cannot_suppress_remote_profile_consent(tmp_path):
    # The cloud provider is not configured yet. A forged historic "local"
    # identity must not pre-authorize a remote model configured later.
    service = ModelService(home=tmp_path, config={}, local_endpoints=[],
                           model_paths=[], litellm_paths=[])
    spoof = {
        "id": CLOUD_ID, "provider": "custom", "model": "premium",
        "source": "ollama-live", "availability": "reachable", "endpoint": ORIGIN,
    }
    _observe(service, "local-models-history", [spoof])
    with pytest.raises(CloudConsentRequired):
        service.profile("import", "cloud", _main_cloud_profile())
    assert [row["name"] for row in service.profile("list")["profiles"]] == ["default"]
    assert _service(tmp_path).resolve("main") is None


def test_forged_current_native_id_cannot_suppress_remote_profile_consent(tmp_path):
    service = _service(tmp_path)
    spoof = {
        "id": CLOUD_ID, "provider": "custom", "model": "premium",
        "source": "ollama-live", "availability": "reachable", "endpoint": ORIGIN,
    }
    _observe(service, "local-models", [spoof])
    with pytest.raises(CloudConsentRequired):
        service.profile("save", "default", _main_cloud_profile())
    assert service.resolve("main") is None


def test_valid_native_listing_keeps_reachability_without_promoting_price_claim(tmp_path):
    service = _service(tmp_path)
    native = _row("custom", "local", "ollama-live", "reachable", endpoint=ORIGIN)
    native.update({"loaded": True, "context_window": 4096,
                   "context_source": "ollama_api_ps_loaded_context",
                   "max_input_cost_per_million": 0, "max_output_cost_per_million": 0})
    _observe(service, "local-models", [native])
    record = next(row for row in service.status()["models"] if row["id"] == native["id"])
    assert record["availability"] == "reachable"
    assert record["loaded"] is True
    assert record["context_window"] == 4096
    assert "max_input_cost_per_million" not in record
    assert "max_output_cost_per_million" not in record
    assert not service._allowed(record, {"max_input_cost_per_million": 0}, frozenset(), None)
