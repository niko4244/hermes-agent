"""Translate persisted agent model assignments into Hermes child credentials.

This is intentionally a narrow adapter: the universal model registry selects
an identity, while the existing delegation credential resolver constructs the
real provider-specific transport.  It never reads or persists API-key values.
"""

from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlsplit


def _loopback_endpoint(url: Any) -> bool:
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "http" and parsed.hostname in {
            "localhost", "127.0.0.1", "::1"
        } and not parsed.username and not parsed.password)
    except ValueError:
        return False


def _credentials_for_model(record: dict, parent_agent: Any, resolver: Callable) -> dict:
    """Resolve a role model without leaking a parent's key to another host."""
    model = record.get("model")
    provider = record.get("provider")
    endpoint = record.get("endpoint")
    if not isinstance(model, str) or not model or not isinstance(provider, str) or not provider:
        raise ValueError("model assignment is missing its executable identity")
    if record.get("availability") in {"on_disk", "offline", "unavailable", "unconfigured", "disconnected"}:
        raise ValueError("selected agent model is not executable")

    if endpoint and _loopback_endpoint(endpoint) and provider in {
        "custom", "ollama", "lmstudio", "llama.cpp", "llama-server",
        "localai", "jan", "koboldcpp", "vllm"
    }:
        # Hermes' OpenAI-compatible transport accepts a keyless local endpoint
        # with a harmless sentinel; NEVER inherit the parent's cloud API key.
        return {
            "model": model,
            "provider": "custom",
            "base_url": endpoint,
            "api_key": "local",
            "api_mode": "chat_completions",
            "request_overrides": None,
            "max_output_tokens": None,
        }
    if endpoint and provider in {"custom", "litellm"}:
        # A remote custom/LiteLLM endpoint must be explicitly registered in
        # Hermes' credential configuration. Supplying just a URL is not safe:
        # delegation's legacy URL resolver inherits the parent's API key.
        raise ValueError("remote proxy model requires a configured authenticated Hermes provider")
    # Named providers resolve through Hermes' existing credential, API mode,
    # model normalization, and rate-limit infrastructure.
    return resolver({"provider": provider, "model": model}, parent_agent)


def delegation_model_route(
    role: str,
    defaults: dict,
    parent_agent: Any,
    credential_resolver: Callable,
    *,
    service: Any = None,
) -> tuple[dict, list[dict] | None]:
    """Return (credentials, fallback) for a *new* delegated child.

    ``None`` fallback preserves existing delegation behavior when no role has
    been configured. An assigned role defaults to an empty fallback chain,
    preventing the parent's unrelated cloud fallback policy from escaping a
    pinned/local-only agent. Explicit fallback identities are resolved in
    their persisted order and validated before any child is constructed.
    """
    if service is None:
        from hermes_cli.universal_models.service import ModelService

        service = ModelService()

    assignment_role = f"delegation.{role}"
    state = service.status(refresh=False)
    roles = state.get("roles", [])
    settings = next((r for r in roles if r.get("role") == assignment_role), None)
    if not settings:
        return defaults, None

    selected = service.resolve(assignment_role)
    if selected is None:
        if settings.get("mode") == "pinned":
            raise ValueError(f"pinned model for {assignment_role} is unavailable")
        # Automatic routing with no safe candidate must not silently select a
        # cloud service when the policy explicitly constrained local routing.
        if state.get("local_only") or settings.get("local_only") or state.get("policy"):
            raise ValueError(f"no approved model under the active routing policy for {assignment_role}")
        return defaults, None

    result = _credentials_for_model(selected, parent_agent, credential_resolver)
    known = {m.get("id"): m for m in state.get("models", []) if isinstance(m, dict)}
    fallbacks: list[dict] = []
    for model_id in settings.get("fallback_ids") or []:
        candidate = known.get(model_id)
        if candidate is None:
            raise ValueError(f"fallback for {assignment_role} is missing from the registry")
        if state.get("local_only") or settings.get("local_only"):
            if candidate.get("source") != "ollama-live" or not _loopback_endpoint(candidate.get("endpoint")):
                raise ValueError("local-only delegation cannot fall back to cloud inference")
        if state.get("policy") and not service._allowed(candidate, state["policy"], frozenset(), None):
            raise ValueError("delegation fallback violates active routing policy")
        fb = _credentials_for_model(candidate, parent_agent, credential_resolver)
        fallbacks.append(fb)
    return result, fallbacks
