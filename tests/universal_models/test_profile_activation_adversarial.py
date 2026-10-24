"""Profile persistence cannot bypass the remote-model assignment consent boundary."""

import json
import sqlite3
import threading

import pytest

from hermes_cli.universal_models.service import CloudConsentRequired, ModelService


CONFIG = {"providers": {"openrouter": {"models": ["costly-primary", "costly-fallback"]}}}
PRIMARY = "openrouter::costly-primary"
FALLBACK = "openrouter::costly-fallback"


def _svc(tmp_path):
    return ModelService(home=tmp_path, config=CONFIG, local_endpoints=[],
                        model_paths=[], litellm_paths=[])


def _profile(model_id=PRIMARY, *, fallback_ids=(), policy=None, role="main"):
    return {"roles": [{"role": role, "mode": "pinned", "model_id": model_id,
                       "fallback_ids": list(fallback_ids)}], "policy": policy or {}}


@pytest.mark.parametrize("action", ["save", "import"])
def test_unconfirmed_active_profile_write_never_makes_paid_primary_executable(tmp_path, action):
    svc = _svc(tmp_path)
    assert svc.resolve("main") is None
    with pytest.raises(CloudConsentRequired):
        svc.profile(action, "default", _profile())
    assert svc.profile("export")["roles"] == []
    assert svc.resolve("main") is None


def test_unconfirmed_remote_fallback_in_import_is_not_persisted(tmp_path):
    svc = _svc(tmp_path)
    svc.assign("main", "pinned", PRIMARY)
    with pytest.raises(CloudConsentRequired):
        svc.profile("import", "other", _profile(fallback_ids=[FALLBACK]))
    assert {r["name"] for r in svc.profile("list")["profiles"]} == {"default"}


def test_legacy_imported_cloud_profile_cannot_activate_without_consent(tmp_path):
    svc = _svc(tmp_path)
    with sqlite3.connect(svc.db_path) as conn:
        conn.execute("INSERT INTO profiles(name,payload) VALUES (?,?)",
                     ("legacy", json.dumps(_profile())))
    assert svc.resolve("main") is None
    with pytest.raises(CloudConsentRequired):
        svc.profile("activate", "legacy")
    assert svc.profile("export")["name"] == "default"
    assert svc.resolve("main") is None
    activated = svc.profile("activate", "legacy", confirm_cloud=True)
    assert activated["name"] == "legacy"
    assert svc.resolve("main")["id"] == PRIMARY


def test_confirmed_import_stages_remote_but_activation_still_checks_current_authority(tmp_path):
    svc = _svc(tmp_path)
    svc.profile("import", "cloud", _profile(), confirm_cloud=True)
    assert svc.profile("export")["name"] == "default"
    with pytest.raises(CloudConsentRequired):
        svc.profile("activate", "cloud")
    assert svc.resolve("main") is None
    svc.profile("activate", "cloud", confirm_cloud=True)
    assert svc.resolve("main")["id"] == PRIMARY


def test_copying_current_cloud_role_does_not_require_duplicate_consent(tmp_path):
    svc = _svc(tmp_path)
    # The assignment RPC already collected its cloud confirmation.
    svc.assign("main", "pinned", PRIMARY)
    current = svc.profile("export")
    svc.profile("save", "copied", current)
    svc.profile("activate", "copied")
    assert svc.resolve("main")["id"] == PRIMARY


def test_restricted_profile_can_be_edited_but_lifting_block_requires_confirmation(tmp_path):
    svc = _svc(tmp_path)
    svc.profile("import", "offline", _profile(policy={"local_only": True}))
    svc.profile("activate", "offline")
    assert svc.resolve("main") is None
    with pytest.raises(CloudConsentRequired):
        svc.profile("save", "offline", _profile())
    assert svc.profile("export")["policy"] == {"local_only": True}
    assert svc.resolve("main") is None


def test_explicit_auto_remote_and_unknown_remote_ids_require_confirmation(tmp_path):
    svc = _svc(tmp_path)
    for identifier in (PRIMARY, "unlisted::future-cloud"):
        profile = {"roles": [{"role": "delegation.leaf", "mode": "auto",
                              "model_id": identifier, "fallback_ids": []}], "policy": {}}
        with pytest.raises(CloudConsentRequired):
            svc.profile("save", "default", profile)


def test_expired_native_ollama_history_is_local_but_expired_proxy_still_needs_consent(tmp_path):
    from hermes_cli.universal_models.discovery import _row

    svc = _svc(tmp_path)
    native = _row("custom", "native", "ollama-live", "reachable",
                  endpoint="http://127.0.0.1:11434/v1")
    proxy = _row("custom", "proxy", "openai-compatible-live", "reachable",
                 endpoint="http://127.0.0.1:1234/v1")
    with sqlite3.connect(svc.db_path) as conn:
        conn.execute("INSERT INTO observations(name,payload,measured_at) VALUES (?,?,?)",
                     ("local-models", json.dumps([native, proxy]), 1.0))
    svc.profile("save", "native", _profile(native["id"]))
    assert svc.resolve("main") is None  # never revives expired listings
    with pytest.raises(CloudConsentRequired):
        svc.profile("save", "proxy", _profile(proxy["id"]))
    assert {r["name"] for r in svc.profile("list")["profiles"]} == {"default", "native"}


def test_confirmation_flag_must_be_boolean_and_invalid_value_never_writes(tmp_path):
    svc = _svc(tmp_path)
    with pytest.raises(ValueError, match="confirm_cloud"):
        svc.profile("import", "danger", _profile(), confirm_cloud="true")
    assert svc.profile("list")["profiles"] == [{"name": "default", "active": True}]


def test_delete_cannot_remove_profile_activated_between_check_and_write(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    svc.profile("save", "target", {"roles": [], "policy": {}})
    examined = threading.Event()
    resume = threading.Event()
    original = svc._active

    def outdated_active():
        value = original()
        examined.set()
        assert resume.wait(3)
        return value

    monkeypatch.setattr(svc, "_active", outdated_active)
    errors = []

    def delete():
        try:
            svc.profile("delete", "target")
        except ValueError as error:
            errors.append(str(error))

    worker = threading.Thread(target=delete)
    worker.start()
    try:
        assert examined.wait(3)
        other = _svc(tmp_path)
        other.profile("activate", "target")
    finally:
        resume.set()
        worker.join(3)
    assert errors == ["cannot delete the default or active profile"]
    assert _svc(tmp_path).profile("export")["name"] == "target"
