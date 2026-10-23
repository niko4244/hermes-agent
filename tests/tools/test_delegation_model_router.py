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


def test_saved_auto_role_with_no_local_model_cannot_use_paid_parent_defaults():
    paid_parent = {"model": "paid", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}
    saved_auto = assigned(mode="auto")
    with pytest.raises(ValueError, match="no approved model"):
        delegation_model_route(
            "leaf", paid_parent, SimpleNamespace(), cloud_resolver,
            service=Registry(row=None, role=saved_auto, local_only=False),
        )


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
    _, fallbacks = delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver,
                                               service=Registry(local, role=role, fallback_rows=[local_fb, cloud]))
    assert [f["model"] for f in fallbacks] == ["two", "coder"]
    assert [f["api_key"] for f in fallbacks] == ["local", "own-provider-key"]


def test_local_only_rejects_compatible_proxy_even_on_loopback():
    proxy = {"id": "custom@proxy::coder", "provider": "custom", "model": "coder",
             "source": "openai-compatible-live", "availability": "reachable",
             "endpoint": "http://127.0.0.1:8765/v1"}
    with pytest.raises(ValueError, match="local-only"):
        delegation_model_route("leaf", {}, SimpleNamespace(api_key="parent-cloud-key"), cloud_resolver,
                               service=Registry(proxy, role=assigned(), local_only=True))


def test_local_only_rejects_cloud_explicit_fallback_and_allows_verified_native_chain():
    local = {"id": "custom@native::one", "provider": "custom", "model": "one",
             "source": "ollama-live", "availability": "reachable", "endpoint": "http://127.0.0.1:11434/v1"}
    local_fb = {**local, "id": "custom@native::two", "model": "two"}
    registry = Registry(local, role=assigned(), fallback_rows=[local_fb], local_only=True)
    with pytest.raises(ValueError, match="local-only"):
        delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver,
                               service=registry, declared_fallback=[{"provider": "openrouter", "model": "paid"}])
    with pytest.raises(ValueError, match="local-only"):
        delegation_model_route("leaf", {}, SimpleNamespace(api_key="PARENT_KEY"), cloud_resolver,
                               service=registry, declared_fallback=[{"provider": "custom", "model": "two",
                                                                      "base_url": "http://127.0.0.1:11434/v1"}])
    own = [{"provider": "custom", "model": "two", "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "local"}]
    credentials, chain = delegation_model_route("leaf", {}, SimpleNamespace(), cloud_resolver,
                                                service=registry, declared_fallback=own)
    assert credentials["api_key"] == "local" and chain == own


def test_cloud_resolution_error_never_exposes_credentials_to_child_tool_result():
    cloud = {"id": "deepseek::coder", "provider": "deepseek", "model": "coder",
             "availability": "configured"}

    def leaking_resolver(*args):
        raise ValueError("API request failed, bearer sk-secret-from-provider")

    with pytest.raises(ValueError, match="check its configuration") as failure:
        delegation_model_route("leaf", {}, SimpleNamespace(api_key="parent-secret"), leaking_resolver,
                               service=Registry(cloud, role=assigned()))
    assert "sk-secret-from-provider" not in str(failure.value)
    assert "parent-secret" not in str(failure.value)


def test_saved_registered_custom_provider_uses_configured_name_not_inventory_namespace(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"providers": {
        "cloudbird": {"base_url": "https://cloudbird.example/v1", "models": ["coder"],
                      "api_key": "TARGET_KEY"}}})
    selected = {"id": "custom:cloudbird::coder", "provider": "custom:cloudbird", "model": "coder",
                "availability": "configured", "source": "hermes-provider"}
    calls = []

    def resolver(request, parent):
        calls.append(request)
        return {"provider": request["provider"], "model": request["model"], "api_key": "TARGET_KEY",
                "base_url": "https://cloudbird.example/v1", "api_mode": "chat_completions"}

    creds, fallbacks = delegation_model_route("leaf", {}, SimpleNamespace(api_key="PARENT_KEY"), resolver,
                                               service=Registry(selected, role=assigned()))
    assert calls == [{"provider": "cloudbird", "model": "coder"}]
    assert creds["api_key"] == "TARGET_KEY" and fallbacks == []


def test_saved_custom_identity_cannot_resolve_to_unrelated_provider(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"providers": {
        "cloudbird": {"base_url": "https://cloudbird.example/v1", "models": ["coder"],
                      "api_key": "TARGET_KEY"}}})
    selected = {"id": "custom:cloudbird::coder", "provider": "custom:cloudbird", "model": "coder",
                "availability": "configured", "source": "hermes-provider"}

    def resolver(request, parent):
        return {"provider": "openrouter", "model": "coder", "api_key": "PARENT_KEY",
                "base_url": "https://parent.invalid/v1"}

    with pytest.raises(ValueError, match="identity"):
        delegation_model_route("leaf", {}, SimpleNamespace(), resolver,
                               service=Registry(selected, role=assigned()))


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -1.0, True, None])
def test_cost_cap_rejects_unverified_bad_numeric_values_and_unknown_fallback(monkeypatch, price):
    from tools.delegation_model_router import enforce_explicit_route_policy

    rows = [
        {"provider": "deepseek", "model": "coder", "availability": "configured",
         "source": "hermes-provider", "max_input_cost_per_million": price},
        {"provider": "openrouter", "model": "fallback", "availability": "configured",
         "source": "hermes-provider"},
    ]

    class CostRegistry:
        def status(self, refresh=False):
            return {"policy": {"max_input_cost_per_million": 1.0}, "models": rows}

    monkeypatch.setattr("tools.delegation_model_router._active_registry", lambda: CostRegistry())
    route = {"provider": "deepseek", "model": "coder", "base_url": "https://deepseek.test/v1"}
    with pytest.raises(ValueError, match="cost cap"):
        enforce_explicit_route_policy(route, SimpleNamespace(), {"fallback_providers": []})

    rows[0]["max_input_cost_per_million"] = 0.5
    with pytest.raises(ValueError, match="fallback"):
        enforce_explicit_route_policy(route, SimpleNamespace(), {"fallback_providers": [
            {"provider": "openrouter", "model": "fallback"},
        ]})
