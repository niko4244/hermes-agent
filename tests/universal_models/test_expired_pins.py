"""An explicit native Ollama pin survives a short discovery cache TTL safely."""

import json
import sqlite3

from hermes_cli.universal_models.discovery import _row
from hermes_cli.universal_models.service import ModelService


def _cached_row(home, record, timestamp=1000.0):
    with sqlite3.connect(home / "universal_models.sqlite3") as connection:
        connection.execute(
            "INSERT INTO observations(name,payload,measured_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, measured_at=excluded.measured_at",
            ("local-models", json.dumps(record if isinstance(record, list) else [record]), timestamp),
        )


def test_expired_pinned_ollama_revalidates_only_prior_direct_origin(tmp_path, monkeypatch):
    primary = _row("custom", "small", "ollama-live", "reachable",
                   endpoint="http://127.0.0.1:11434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, primary)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)
    called = []

    def probe(endpoints):
        called.append(endpoints)
        return [primary]

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", probe)
    assert svc.status()["models"] == []  # stale reads do not claim reachability
    # An existing pin was saved while discovery was fresh.
    profile = svc.profile("export")
    profile["roles"] = [{"role": "main", "mode": "pinned", "model_id": primary["id"],
                         "fallback_ids": []}]
    svc.profile("save", "default", profile)
    result = svc.resolve("main")
    assert result["id"] == primary["id"]
    assert called == [("http://127.0.0.1:11434",)]
    assert svc.status()["models"][0]["id"] == primary["id"]


def test_expired_native_pin_does_not_fallback_to_other_model(tmp_path, monkeypatch):
    primary = _row("custom", "small", "ollama-live", "reachable",
                   endpoint="http://127.0.0.1:11434/v1")
    sibling = _row("custom", "other", "ollama-live", "reachable",
                   endpoint="http://127.0.0.1:11434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, primary)
    p = svc.profile("export")
    p["roles"] = [{"role": "main", "mode": "pinned", "model_id": primary["id"],
                   "fallback_ids": []}]
    svc.profile("save", "default", p)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)
    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", lambda origins: [sibling])
    assert svc.resolve("main") is None
    assert svc.status()["models"] == []


def test_expired_proxy_pin_never_performs_automatic_revalidation(tmp_path, monkeypatch):
    proxy = _row("custom", "proxy", "openai-compatible-live", "reachable",
                 endpoint="http://127.0.0.1:1234/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, proxy)
    p = svc.profile("export")
    p["roles"] = [{"role": "main", "mode": "pinned", "model_id": proxy["id"],
                   "fallback_ids": []}]
    svc.profile("save", "default", p)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unverified proxy must never be implicitly probed")

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", forbidden)
    assert svc.resolve("main") is None


def test_price_caps_reject_nonfinite_negative_or_boolean_discovery_claims():
    row = {"availability": "reachable", "source": "ollama-live",
           "max_input_cost_per_million": 0}
    cap = {"max_input_cost_per_million": 1.0}
    assert ModelService._allowed(row, cap, frozenset(), None)
    for bad in (float("nan"), float("inf"), -1.0, False, True, "0", None):
        assert not ModelService._allowed(
            {**row, "max_input_cost_per_million": bad}, cap, frozenset(), None
        )


def test_explicit_auto_main_survives_native_listing_ttl_without_cloud_discovery(tmp_path, monkeypatch):
    native = _row("custom", "local", "ollama-live", "reachable",
                  endpoint="http://127.0.0.1:11434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, native)
    svc.profile("save", "default", {"roles": [
        {"role": "main", "mode": "auto", "model_id": None, "fallback_ids": []},
    ], "policy": {}})
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)
    calls = []

    def probe(endpoints):
        calls.append(endpoints)
        return [native]

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", probe)
    assert svc.resolve("main")["id"] == native["id"]
    assert calls == [("http://127.0.0.1:11434",)]


def test_unsaved_auto_role_does_not_implicitly_probe_expired_local_service(tmp_path, monkeypatch):
    native = _row("custom", "local", "ollama-live", "reachable",
                  endpoint="http://127.0.0.1:11434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, native)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsaved profile role cannot implicitly probe local services")
    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", forbidden)
    assert svc.resolve("main") is None


def test_revalidating_one_origin_retains_other_expired_pin_as_historical_hint(tmp_path, monkeypatch):
    first = _row("custom", "one", "ollama-live", "reachable",
                 endpoint="http://127.0.0.1:11434/v1")
    second = _row("custom", "two", "ollama-live", "reachable",
                  endpoint="http://127.0.0.1:21434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, [first, second])
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1130.0)
    observed = []

    def probe(endpoints):
        observed.append(endpoints)
        return [first] if endpoints == ("http://127.0.0.1:11434",) else [second]

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", probe)
    for target in (first, second):
        p = svc.profile("export")
        p["roles"] = [{"role": "main", "mode": "pinned",
                       "model_id": target["id"], "fallback_ids": []}]
        svc.profile("save", "default", p)
        assert svc.resolve("main")["id"] == target["id"]
    assert observed == [("http://127.0.0.1:11434",), ("http://127.0.0.1:21434",)]
    with sqlite3.connect(tmp_path / "universal_models.sqlite3") as conn:
        historic = json.loads(conn.execute(
            "SELECT payload FROM observations WHERE name='local-models-history'"
        ).fetchone()[0])
    assert {r["id"] for r in historic} == {first["id"], second["id"]}


def test_future_dated_native_listing_cannot_trigger_pin_revalidation(tmp_path, monkeypatch):
    row = _row("custom", "small", "ollama-live", "reachable",
               endpoint="http://127.0.0.1:11434/v1")
    svc = ModelService(home=tmp_path, config={}, local_endpoints=[], model_paths=[])
    _cached_row(tmp_path, row, timestamp=2000.0)
    p = svc.profile("export")
    p["roles"] = [{"role": "main", "mode": "pinned", "model_id": row["id"],
                   "fallback_ids": []}]
    svc.profile("save", "default", p)
    monkeypatch.setattr("hermes_cli.universal_models.service.time.time", lambda: 1000.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("future-dated provenance must not trigger a probe")

    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", forbidden)
    assert svc.status()["models"] == []
    assert svc.resolve("main") is None
