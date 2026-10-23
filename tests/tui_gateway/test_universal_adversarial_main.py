"""Adversarial runtime policy checks for saved main-model assignments."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tui_gateway.universal_model_runtime import enforce_main_turn_policy, main_model_override


def _cloud_row(model="primary", provider="openrouter", model_id="cloud:primary"):
    return {
        "id": model_id, "model": model, "provider": provider,
        "availability": "configured", "source": "hermes-config",
    }


def _profile_service(*, fallback_ids=(), models=(), policy=None):
    service = Mock()
    service.profile.return_value = {
        "roles": [{"role": "main", "mode": "pinned", "model_id": "cloud:primary",
                   "fallback_ids": list(fallback_ids)}],
        "policy": {} if policy is None else policy,
    }
    service.status.return_value = {"models": list(models), "policy": policy or {}}
    service._allowed.return_value = True
    return service


def test_removed_fallback_cannot_answer_next_turn_under_unrestricted_policy():
    """A previously activated fallback must lose authority when its ID is removed."""
    selected = main_model_override(_cloud_row())
    session = {"model_override": selected}
    agent = SimpleNamespace(
        model="old-fallback", provider="anthropic", base_url="https://api.anthropic.com",
        _provider_fallback_active=True, _fallback_activated=True,
        _fallback_chain=[{"provider": "anthropic", "model": "old-fallback"}],
    )
    service = _profile_service(models=[_cloud_row()])
    with pytest.raises(ValueError, match="authorizes this session fallback"):
        enforce_main_turn_policy(agent, session, service)


def test_saved_main_model_rejects_unexpected_provider_even_with_same_slug():
    """An unrelated provider's identically named model is not the saved pin."""
    agent = SimpleNamespace(
        model="primary", provider="anthropic", base_url="https://api.anthropic.com",
        _fallback_activated=False, _provider_fallback_active=False,
    )
    service = _profile_service(models=[_cloud_row()])
    with pytest.raises(ValueError, match="saved model selection"):
        enforce_main_turn_policy(agent, {"model_override": main_model_override(_cloud_row())}, service)


@pytest.mark.parametrize("runtime_url,allowed", [
    ("https://fallback.example/v1", True),
    ("https://unrelated.example/v1", False),
])
def test_saved_fallback_authorizes_only_its_approved_runtime(runtime_url, allowed):
    """The active fallback must match its currently approved provider and endpoint."""
    approved = {"model": "backup", "provider": "custom", "base_url": "https://fallback.example/v1"}
    agent = SimpleNamespace(
        model="backup", provider="custom", base_url=runtime_url,
        _fallback_activated=True, _provider_fallback_active=True,
    )
    service = _profile_service(fallback_ids=["custom:backup"], models=[_cloud_row()])
    with patch("tui_gateway.universal_model_runtime.approved_main_fallbacks_for_ids",
               return_value=[approved]) as validate:
        if allowed:
            enforce_main_turn_policy(agent, {"model_override": main_model_override(_cloud_row())}, service)
        else:
            with pytest.raises(ValueError, match="authorizes this session fallback"):
                enforce_main_turn_policy(agent, {"model_override": main_model_override(_cloud_row())}, service)
    validate.assert_called_once_with(service, ["custom:backup"])


@pytest.mark.parametrize("runtime_url,allowed", [
    ("https://registered.example/v1", True),
    ("https://different.example/v1", False),
])
def test_registered_custom_main_pin_is_bound_to_provider_origin(runtime_url, allowed):
    row = _cloud_row(provider="custom:my-provider")
    agent = SimpleNamespace(
        model="primary", provider="custom", requested_provider="my-provider",
        base_url=runtime_url, _fallback_activated=False,
    )
    service = _profile_service(models=[row])
    with patch("tools.delegation_model_router._registered_custom_provider",
               return_value=("my-provider", "https://registered.example/v1")):
        if allowed:
            enforce_main_turn_policy(agent, {"model_override": main_model_override(row)}, service)
        else:
            with pytest.raises(ValueError, match="saved model selection"):
                enforce_main_turn_policy(agent, {"model_override": main_model_override(row)}, service)
