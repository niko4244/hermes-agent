"""Explicit Hermes cloud config inventory: no catalogs, guessing or secrets."""

from __future__ import annotations

import json
import urllib.request

import pytest

from hermes_cli.universal_models.discovery import config_aliases, configured_models, merge_models
from hermes_cli.universal_models.service import ModelService


@pytest.mark.parametrize(("provider", "expected"), [
    ("openai-api", "openai-api"), ("openai", "openai"),
    ("anthropic", "anthropic"), ("gemini", "gemini"),
    ("google", "gemini"), ("openrouter", "openrouter"),
    ("groq", "groq"), ("together", "together"),
    ("fireworks", "fireworks"), ("deepseek", "deepseek"),
    ("mistral", "mistral"), ("azure-foundry", "azure-foundry"),
    ("bedrock", "bedrock"), ("aws-bedrock", "bedrock"),
])
def test_explicit_current_cloud_ids_use_real_hermes_provider_identity(provider, expected):
    rows = configured_models({"model": {"provider": provider, "default": "deployment/the-model"}})
    assert [(row["id"], row["availability"], row["capabilities"]) for row in rows] == [
        (f"{expected}::deployment/the-model", "configured", [])
    ]
    assert rows[0]["source"] == "hermes-config"
    assert "endpoint" not in rows[0]


@pytest.mark.parametrize("provider", [
    "openai-api", "anthropic", "gemini", "openrouter", "groq", "together",
    "fireworks", "deepseek", "mistral", "azure-foundry", "bedrock",
])
def test_builtin_providers_models_without_endpoint_are_explicit_only(provider):
    config = {"providers": {provider: {"models": {
        "explicit-first": {"context_length": 200000, "api_key": "INTERNAL-SECRET"},
        "explicit-second": {},
    }}}}
    rows = configured_models(config)
    assert {row["id"] for row in rows} == {
        f"{provider}::explicit-first", f"{provider}::explicit-second"
    }
    assert all(row["availability"] == "configured" and "endpoint" not in row for row in rows)
    assert "INTERNAL-SECRET" not in json.dumps(rows)


def test_keyed_custom_endpoint_wins_even_when_name_collides_with_builtin():
    config = {"providers": {
        "openai": {"api": "https://proxy.example.test/v1?token=DO_NOT_ECHO", "default_model": "deploy-a",
                   "models": ["deploy-a", {"id": "deploy-b", "api_key": "DO_NOT_ECHO"}]},
        "Custom Team": {"base_url": "https://proxy.example.test/v1", "name": "Renamed Display",
                        "models": ["deploy-a", {"name": "deploy-c"}]},
    }}
    rows = configured_models(config)
    assert {row["id"] for row in rows} == {
        "custom:openai::deploy-a", "custom:openai::deploy-b",
        "custom:custom-team::deploy-a", "custom:custom-team::deploy-c",
    }
    assert "DO_NOT_ECHO" not in json.dumps(rows)
    assert all("endpoint" not in row for row in rows)


def test_explicit_provider_selections_cover_aux_fallback_and_aliases():
    cfg = {
        "model": {"provider": "google", "default": "gemini-test"},
        "fallback_providers": [{"provider": "aws", "model": "anthropic.claude-vx"},
                               {"provider": "openrouter", "model": "vendor/x:free"}],
        "fallback_model": {"provider": "deepseek", "model": "legacy-superseded"},
        "auxiliary": {"vision": {"provider": "fireworks", "model": "accounts/a/vision",
                                 "fallback_chain": [{"provider": "mistral", "model": "mistral-small"}]},
                      "compression": {"provider": "auto", "model": "should-not-invent"}},
        "model_aliases": {"Coder": {"provider": "openai-api", "model": "gpt-code"}},
        "providers": {"anthropic": {"request_timeout_seconds": 30,
                                    "models": ["claude-explicit", {"id": "claude-extra"}]}},
    }
    rows = configured_models(cfg)
    assert {row["id"] for row in rows} == {
        "gemini::gemini-test", "bedrock::anthropic.claude-vx",
        "openrouter::vendor/x:free", "fireworks::accounts/a/vision",
        "mistral::mistral-small", "openai-api::gpt-code",
        "anthropic::claude-explicit", "anthropic::claude-extra",
    }
    assert "legacy-superseded" not in json.dumps(rows)
    assert "should-not-invent" not in json.dumps(rows)


def test_alias_precedence_and_string_provider_slash_preserves_aggregator_model():
    cfg = {
        "model_aliases": {"FAST": {"provider": "google", "model": "gemini-explicit"}},
        "model": {"provider": "anthropic", "aliases": {
            "fast": "deepseek/should-be-ignored", "Router": "openrouter/openai/gpt-test",
            "bare": "claude-bare", "unsafe": "Bearer SECRET_DO_NOT_SERIALIZE",
        }},
    }
    aliases = config_aliases(cfg)
    assert {x["alias"].lower() for x in aliases} == {"fast", "router", "bare"}
    assert {r["id"] for r in configured_models(cfg)} == {
        "gemini::gemini-explicit", "openrouter::openai/gpt-test", "anthropic::claude-bare",
    }
    assert "SECRET_DO_NOT_SERIALIZE" not in json.dumps(aliases)


def test_no_model_enumeration_from_provider_secrets_or_ambient_credentials(tmp_path, monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("provider enumeration must be completely offline")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr("hermes_cli.universal_models.service.local_server_models", fail_network)
    cfg = {
        "providers": {"anthropic": {"api_key": "sk-SUPERPRIVATE"},
                      "openai-api": {"key_env": "OPENAI_API_KEY"},
                      "groq": {"request_timeout_seconds": 30},
                      "my-proxy": {"base_url": "https://example.test/v1", "api_key": "sk-NOTPUBLIC"}},
        "custom_providers": [{"name": "Empty", "base_url": "https://example.test/v1", "api_key": "sk-NEVER"}],
    }
    monkeypatch.setenv("OPENAI_API_KEY", "sk-DO_NOT_ENCODE")
    svc = ModelService(home=tmp_path, config=cfg, local_endpoints=[])
    state = svc.status(refresh=False)
    assert state["models"] == []
    assert "sk-" not in json.dumps(state)
    assert "sk-" not in (tmp_path / "universal_models.sqlite3").read_bytes().decode("latin1")


def test_malformed_secret_like_model_ids_and_disabled_providers_are_rejected():
    cfg = {
        "model": {"provider": "openrouter", "default": "?api_key=SECRET"},
        "providers": {
            "anthropic": {"enabled": False, "models": ["should-not-appear"]},
            "mykey": {"api": "https://example.test/v1", "models": [
                "good", "sk-SECRET", "token=SECRET", "Bearer SECRET",
                {"id": "good-next", "api_key": "sk-HIDDEN"},
                {"id": "https://user:secret@example.test/v1"},
                None, 123, {"api_key": "another-secret"},
            ]},
        },
    }
    rows = configured_models(cfg)
    assert {r["id"] for r in rows} == {"custom:mykey::good", "custom:mykey::good-next"}
    assert "SECRET" not in json.dumps(rows)
    assert "HIDDEN" not in json.dumps(rows)


def test_disabled_legacy_endpoints_cannot_create_cloud_models():
    cfg = {"custom_providers": [
        {"name": "Disallowed", "enabled": False,
         "base_url": "https://api.example.test/v1", "models": ["bad-model"]},
        {"name": "Allowed", "base_url": "https://api.example.test/v1", "models": ["okay-model"]},
    ]}
    assert {row["id"] for row in configured_models(cfg)} == {"custom:allowed::okay-model"}


def test_litellm_proxy_needs_valid_enabled_hermes_endpoint_and_preserves_alias_identity(tmp_path):
    alias_file = tmp_path / "litellm.yaml"
    alias_file.write_text("""model_list:
  - model_name: alpha
    litellm_params: {model: openai/gpt-unknown, api_key: sk-UNSAFE}
  - model_name: beta
    litellm_params: {model: anthropic/claude-unknown, api_key: sk-UNSAFE}
""", encoding="utf-8")
    base = {"litellm_paths": [alias_file], "home": tmp_path / "state"}
    unavailable = [
        {},
        {"providers": {"litellm": {"api_key": "sk-ANY"}}},
        {"providers": {"litellm": {"enabled": False, "api": "https://proxy.example.test/v1"}}},
        {"custom_providers": [{"name": "litellm", "enabled": False,
                                "base_url": "https://proxy.example.test/v1"}]},
    ]
    for cfg in unavailable:
        rows = ModelService(config=cfg, **base).status()["models"]
        assert {r["id"] for r in rows} == {"custom:litellm::alpha", "custom:litellm::beta"}
        assert {r["availability"] for r in rows} == {"unconfigured"}
    cfg = {"providers": {"litellm": {"api": "https://proxy.example.test/v1",
                                    "api_key": "sk-NOT-SERIALIZED", "models": ["alpha"]}}}
    rows = ModelService(config=cfg, **base).status()["models"]
    assert {r["id"] for r in rows} == {"custom:litellm::alpha", "custom:litellm::beta"}
    assert {r["availability"] for r in rows} == {"configured"}
    assert "sk-NOT-SERIALIZED" not in json.dumps(rows)


def test_duplicate_aliases_and_repeated_provider_models_produce_stable_unique_rows():
    cfg = {
        "model": {"provider": "openrouter", "default": "vendor/coder"},
        "fallback_providers": [{"provider": "openrouter", "model": "vendor/coder"}],
        "auxiliary": {"compression": {"provider": "openrouter", "model": "vendor/coder"}},
        "model_aliases": {"favorite": {"provider": "openrouter", "model": "vendor/coder"}},
        "providers": {"openrouter": {"models": ["vendor/coder", "vendor/coder", "other"]}},
    }
    rows = configured_models(cfg)
    assert sorted(r["id"] for r in rows) == ["openrouter::other", "openrouter::vendor/coder"]
    assert next(row for row in rows if row["model"] == "vendor/coder")["source"] == "hermes-config"
    assert len(merge_models(rows + rows)) == 2


def test_real_hermes_yaml_config_path_surfaces_only_explicit_models(tmp_path, monkeypatch):
    import yaml

    from hermes_cli.config import load_config_readonly

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = {
        "model": {"provider": "openai-api", "default": "gpt-configured"},
        "providers": {
            "anthropic": {"models": {"claude-configured": {"timeout_seconds": 600}}},
            "team": {"api": "https://team.example.test/v1", "models": [{"id": "team-coder"}],
                     "extra_headers": {"Authorization": "Bearer DO_NOT_INCLUDE"}},
        },
        "fallback_providers": [{"provider": "bedrock", "model": "anthropic.claude-provisioned"}],
        "model_aliases": {"quick": {"provider": "deepseek", "model": "deepseek-configured"}},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_config_readonly()
    assert loaded["model"]["default"] == "gpt-configured"
    svc = ModelService(home=tmp_path, local_endpoints=[])
    payload = svc.status()
    assert {row["id"] for row in payload["models"]} == {
        "openai-api::gpt-configured", "anthropic::claude-configured",
        "custom:team::team-coder", "bedrock::anthropic.claude-provisioned",
        "deepseek::deepseek-configured",
    }
    assert "DO_NOT_INCLUDE" not in json.dumps(payload)
