"""Validate registry model selection before constructing a TUI agent.

Existing /model explicit session selections and restored session histories take
precedence over the persisted universal-model profile. This adapter adds only
the new-profile tier; actual provider requests remain in Hermes' transports.
"""

from __future__ import annotations

from typing import Any

from tools.delegation_model_router import _loopback_endpoint


def main_model_override(record: dict[str, Any]) -> dict[str, Any]:
    """Build a credential-free session runtime override from a verified row."""
    provider, model = record.get("provider"), record.get("model")
    if not isinstance(provider, str) or not isinstance(model, str) or not provider or not model:
        raise ValueError("selected main model has no executable identity")
    if record.get("availability") not in {"reachable", "configured"}:
        raise ValueError("selected main model is unavailable")
    endpoint = record.get("endpoint")
    if record.get("source") == "ollama-live" and _loopback_endpoint(endpoint):
        # Only a directly observed Ollama /api/tags result is treated as a
        # local provider. A localhost OpenAI proxy might bill for cloud calls.
        if provider != "custom":
            raise ValueError("unexpected local model provider")
        return {
            "model": model, "provider": "custom", "base_url": endpoint,
            "api_key": "local", "api_mode": "chat_completions",
            "_universal_local": True,
        }
    if provider == "custom" and _loopback_endpoint(endpoint):
        # A generic OpenAI-compatible proxy may route to paid external
        # services. Let Hermes require an explicitly configured provider.
        raise ValueError("non-Ollama endpoint must be registered in Hermes")
    if provider in {"litellm", "custom"}:
        raise ValueError("unregistered proxy provider")
    return {"model": model, "provider": provider}


def selected_profile_main_override(service: Any = None) -> dict | None:
    """Resolve persisted main pin at new-agent build time; fail closed."""
    if service is None:
        from hermes_cli.universal_models.service import ModelService

        service = ModelService()
    settings = next((r for r in service.status().get("roles", []) if r.get("role") == "main"), None)
    if not settings or settings.get("mode") not in {"pinned", "manual"}:
        return None
    model = service.resolve("main")
    if model is None:
        raise ValueError("pinned main model is currently unavailable under the active routing policy")
    return main_model_override(model)
