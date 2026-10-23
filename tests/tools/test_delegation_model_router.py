"""Behavioral tests for model roles driving real delegate child credentials."""

from types import SimpleNamespace

import pytest

from tools.delegation_model_router import delegation_model_route


class Registry:
    def __init__(self, row=None, *, role=None, fallback_rows=(), local_only=False):
        self.row = row
        self.role = role
        self.fallback_rows = list(fallback_rows)
        self.local_only = local_only

    def status(self, refresh=False):
        return {
            "roles": [self.role] if self.role else [],
            "models": [r for r in [self.row, *self.fallback_rows] if r],
            "local_only": self.local_only,
        }

    def profile(self, action, name=None, data=None):
        assert action == "export"
        return {"roles": [self.role] if self.role else [], "policy": {"local_only": self.local_only}}

    def resolve(self, role):
        return self.row


def cloud_resolver(config, parent):
    if config["provider"] == "deepseek":
        return {
            "model": config["model"],
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "own-provider-key",
            "api_mode": "chat_completions",
        }
    raise ValueError("unconfigured provider")


def assigned(role="delegation.leaf", fallback_ids=(), mode="pinned"):
    return {"role": role, "mode": mode, "fallback_ids": list(fallback_ids)}


def test_unconfigured_route_preserves_existing_delegation_behavior():
    previous = {"model": "default-model", "provider": None}
    result, fallbacks = delegation_model_route(
        "leaf", previous, SimpleNamespace(api_key="secret"), cloud_resolver, service=Registry()
    )
    assert result is previous
    assert fallbacks is None


def test_display_only_auto_role_does_not_change_existing_provider():
    previous = {"model": "legacy", "provider": "anthropic"}

    class DisplayOnlyRegistry(Registry):
        def status(self, refresh=False):
            state = super().status(refresh)
            state["roles"] = [{"role": "delegation.leaf", "mode": "auto", "fallback_ids": []}]
            return state

        def resolve(self, role):
            raise AssertionError("display-only default may not select a new model")

    original, fallbacks = delegation_model_route(
        "leaf", previous, SimpleNamespace(), cloud_resolver,
        service=DisplayOnlyRegistry(),
    )
    assert original is previous
    assert fallbacks is None


def test_local_openai_compatible_model_uses_own_sentinel_never_parent_key():
    local = {
        "id": "custom/local/mistral",
        "provider": "custom",
        "model": "mistral",
        "endpoint": "http://127.0.0.1:11434/v1",
        "availability": "reachable",
    }
    parent = SimpleNamespace(api_key="sensitive-cloud-token", provider="openrouter")
    result, fallbacks = delegation_model_route(
        "leaf", {}, parent, cloud_resolver, service=Registry(local, role=assigned())
    )
    assert result["model"] == "mistral"
    assert result["provider"] == "custom"
    assert result["api_key"] == "local"
    assert result["api_mode"] == "chat_completions"
    assert fallbacks == []  # never inherit unrelated paid-cloud fallback chain


def test_cloud_provider_uses_existing_credential_resolver():
    cloud = {"id": "deepseek/default/coder", "provider": "deepseek", "model": "coder", "availability": "configured"}
    result, fallbacks = delegation_model_route(
        "orchestrator", {}, SimpleNamespace(), cloud_resolver,
        service=Registry(cloud, role=assigned("delegation.orchestrator")),
    )
    assert result["api_key"] == "own-provider-key"
    assert result["model"] == "coder"
    assert fallbacks == []


def test_unregistered_remote_proxy_rejected_before_any_inference():
    proxy = {
        "id": "litellm/proxy/alpha", "provider": "litellm",
        "model": "alpha", "endpoint": "https://proxy.example/v1", "availability": "configured",
    }
    with pytest.raises(ValueError, match="authenticated Hermes provider"):
        delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver, service=Registry(proxy, role=assigned()))


def test_missing_pinned_model_fails_closed_and_does_not_inherit_parent():
    with pytest.raises(ValueError, match="unavailable"):
        delegation_model_route("leaf", {"model": "parent"}, SimpleNamespace(), cloud_resolver,
                               service=Registry(role=assigned()))


def test_explicit_fallback_order_preserved_and_local_only_blocks_cloud():
    local = {"id": "local:one", "model": "one", "provider": "custom", "availability": "reachable", "endpoint": "http://localhost:11434/v1"}
    local_fb = {"id": "local:two", "model": "two", "provider": "custom", "availability": "reachable", "endpoint": "http://localhost:1234/v1"}
    cloud = {"id": "cloud:coder", "model": "coder", "provider": "deepseek", "availability": "configured"}
    role = assigned(fallback_ids=["local:two", "cloud:coder"])
    with pytest.raises(ValueError, match="local-only"):
        delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver,
                               service=Registry(local, role=role, fallback_rows=[local_fb, cloud], local_only=True))
    result, fallbacks = delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver,
                                               service=Registry(local, role=role, fallback_rows=[local_fb, cloud]))
    assert [f["model"] for f in fallbacks] == ["two", "coder"]
    assert [f["api_key"] for f in fallbacks] == ["local", "own-provider-key"]
