"""Translate persisted agent model assignments into Hermes child credentials.

This is intentionally a narrow adapter: the universal model registry selects
an identity, while the existing delegation credential resolver constructs the
real provider-specific transport.  It never reads or persists API-key values.
"""

from __future__ import annotations

import math
from typing import Any, Callable
from urllib.parse import urlsplit


def _active_registry() -> Any:
    """Read saved assignments in this turn's profile, not a launch-profile global.

    A session without registry state retains legacy delegation without
    creating an unused SQLite sidecar or scanning the user's model directories.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli.universal_models import ModelService

    home = get_hermes_home()
    if not (home / "universal_models.sqlite3").is_file():
        return None
    return ModelService(home=home, local_endpoints=[], model_paths=[])


def _loopback_endpoint(url: Any) -> bool:
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "http" and parsed.hostname in {
            "localhost", "127.0.0.1", "::1"
        } and parsed.path == "/v1" and not parsed.query and not parsed.fragment
            and not parsed.username and not parsed.password)
    except ValueError:
        return False


def _registered_custom_provider(identity: str) -> tuple[str, str] | None:
    """Turn a saved inventory namespace into its *current profile's* route.

    Hermes runs a named custom backend using its configured provider key, not
    the discovery-only ``custom:<key>`` namespace. This must be revalidated on
    every route: a stale saved ID from another profile cannot lease its key.
    """
    from hermes_cli.config import (
        get_compatible_custom_providers, is_provider_enabled, load_config_readonly,
    )
    from hermes_cli.providers import custom_provider_slug

    config = load_config_readonly()
    if not isinstance(config, dict):
        return None
    view = dict(config)
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        view["custom_providers"] = [entry for entry in legacy if isinstance(entry, dict)
                                     and is_provider_enabled(entry)]
    for entry in get_compatible_custom_providers(view):
        if not isinstance(entry, dict) or not is_provider_enabled(entry):
            continue
        name = str(entry.get("name") or entry.get("provider_key") or "").strip()
        key = str(entry.get("provider_key") or name).strip()
        endpoint = entry.get("base_url")
        if name and key and isinstance(endpoint, str) and endpoint.strip():
            if custom_provider_slug(name, key).lower() == identity.lower():
                return key, endpoint.strip().rstrip("/")
    return None


def _credentials_for_model(record: dict, parent_agent: Any, resolver: Callable) -> dict:
    """Resolve a role model without leaking a parent's key to another host."""
    model = record.get("model")
    provider = record.get("provider")
    endpoint = record.get("endpoint")
    if not isinstance(model, str) or not model or not isinstance(provider, str) or not provider:
        raise ValueError("model assignment is missing its executable identity")
    if record.get("availability") not in {"configured", "reachable"}:
        raise ValueError("selected agent model is not executable")

    if endpoint and record.get("availability") == "reachable" and _loopback_endpoint(endpoint) and provider in {
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
    if provider in {"custom", "litellm"}:
        # A remote custom/LiteLLM endpoint must be explicitly registered in
        # Hermes' credential configuration. Supplying just a URL is not safe:
        # delegation's legacy URL resolver inherits the parent's API key.
        raise ValueError("remote proxy model requires a configured authenticated Hermes provider")
    configured_custom = None
    if provider.startswith("custom:"):
        try:
            configured_custom = _registered_custom_provider(provider)
        except Exception as exc:
            raise ValueError("saved custom provider identity could not be verified in this profile") from exc
        if configured_custom is None:
            raise ValueError("saved custom provider identity is not registered in this profile")
        provider = configured_custom[0]
    # Named providers resolve through Hermes' existing credential, API mode,
    # model normalization, and rate-limit infrastructure. The resolver may
    # include credentials/URLs in an exception; delegation route errors become
    # model-visible tool output, so never echo the underlying exception text.
    try:
        resolved = resolver({"provider": provider, "model": model}, parent_agent)
    except Exception as exc:
        raise ValueError("assigned model provider could not be resolved; check its configuration and credentials") from exc
    if configured_custom is not None:
        if (not isinstance(resolved, dict)
                or resolved.get("provider") not in {"custom", provider}
                or resolved.get("model") != model
                or str(resolved.get("base_url") or "").rstrip("/") != configured_custom[1]):
            raise ValueError("saved custom provider identity resolved to a different runtime route")
    return resolved


def _native_local_model(record: dict) -> bool:
    """Proof of local execution requires a live native Ollama observation."""
    return (record.get("source") == "ollama-live"
            and record.get("provider") == "custom"
            and record.get("availability") == "reachable"
            and _loopback_endpoint(record.get("endpoint")))


def _explicit_local_fallbacks(chain: Any, models: list[dict]) -> bool:
    """Guard a backend's preexisting chain from escaping local-only policy.

    A localhost LiteLLM or OpenAI-compatible proxy may still bill the cloud.
    Match the provider, exact model and endpoint against native Ollama listings.
    """
    if not isinstance(chain, list):
        return False
    return all(
        isinstance(item, dict) and item.get("provider") == "custom"
        # A fallback without a key can inherit the parent's paid provider key
        # downstream; an explicit harmless local sentinel is required.
        and item.get("api_key") in ("local", "no-key-required")
        and any(_native_local_model(model) and model.get("model") == item.get("model")
                and model.get("endpoint") == item.get("base_url") for model in models)
        for item in chain
    )


_COST_KEYS = ("max_input_cost_per_million", "max_output_cost_per_million")


def _matches_verified_route(row: dict, *, provider: Any, model: Any, endpoint: Any) -> bool:
    """A registry row describes this *exact* transport, not just the model name.

    For nonlocal configured cloud rows, the catalog does not carry endpoints;
    accept the same canonical provider ID only (the credentials remain Hermes'
    resolver's responsibility). Native listings require an exact loopback URL.
    """
    return (row.get("provider") == provider and row.get("model") == model
            and row.get("availability") in {"configured", "reachable"}
            and (row.get("endpoint") == endpoint if row.get("endpoint") else True))


def _cost_approved(row: dict, policy: dict) -> bool:
    """Unknown, malformed or excessive pricing is never proof of affordability."""
    from hermes_cli.universal_models import ModelService

    for key in _COST_KEYS:
        if key not in policy:
            continue
        number = row.get(key)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0):
            return False
    return ModelService._allowed(row, policy, frozenset(), None)


def _route_within_price_policy(route: dict, rows: list[dict], policy: dict) -> bool:
    if not any(key in policy for key in _COST_KEYS):
        return True
    return any(
        _matches_verified_route(row, provider=route.get("provider"), model=route.get("model"),
                                endpoint=route.get("base_url"))
        and _cost_approved(row, policy)
        for row in rows
    )


def enforce_explicit_route_policy(creds: dict, parent_agent: Any, routing_cfg: dict) -> None:
    """The selected owner route may not bypass its profile's safety policy.

    Cross-profile routes never invoke this helper: their identity, credentials
    and policy belong entirely to the target profile's route resolver.
    """
    service = _active_registry()
    if service is None:
        return
    state = service.status(refresh=False)
    policy = state.get("policy") or {}
    local_only = bool(state.get("local_only") or policy.get("local_only"))
    if not local_only and not any(key in policy for key in _COST_KEYS):
        return
    provider = creds.get("provider") or getattr(parent_agent, "provider", None)
    model_id = creds.get("model") or getattr(parent_agent, "model", None)
    endpoint = creds.get("base_url") or getattr(parent_agent, "base_url", None)
    candidates = [model for model in state.get("models", []) if isinstance(model, dict)]
    if local_only:
        if not (provider == "custom" and creds.get("api_key") in ("local", "no-key-required")
                and any(_native_local_model(model) and model.get("model") == model_id
                        and model.get("endpoint") == endpoint for model in candidates)):
            raise ValueError("local-only delegation refuses an unverified explicit provider/model route")
    if not _route_within_price_policy({"provider": provider, "model": model_id, "base_url": endpoint},
                                      candidates, policy):
        raise ValueError("delegation cost cap refuses a route without verified affordable pricing")
    fallbacks = routing_cfg.get("fallback_providers")
    if local_only and fallbacks is not None and not _explicit_local_fallbacks(fallbacks, candidates):
        raise ValueError("local-only delegation refuses an unverified explicit fallback chain")
    if fallbacks is not None and any(
        not isinstance(item, dict) or not _route_within_price_policy(item, candidates, policy)
        for item in (fallbacks if isinstance(fallbacks, list) else [fallbacks])
    ):
        raise ValueError("delegation cost cap refuses an unverified or excessive fallback")


def delegation_model_route(
    role: str,
    defaults: dict,
    parent_agent: Any,
    credential_resolver: Callable,
    *,
    service: Any = None,
    declared_fallback: Any = None,
) -> tuple[dict, list[dict] | None]:
    """Return (credentials, fallback) for a *new* delegated child.

    ``None`` fallback preserves existing delegation behavior when no role has
    been configured. An assigned role defaults to an empty fallback chain,
    preventing the parent's unrelated cloud fallback policy from escaping a
    pinned/local-only agent. Explicit fallback identities are resolved in
    their persisted order and validated before any child is constructed.
    """
    if service is None:
        service = _active_registry()
        if service is None:
            return defaults, None

    assignment_role = f"delegation.{role}"
    # status() synthesizes baseline auto rows for built-in roles. Those rows
    # are presentation defaults, not user authorization to change runtime.
    # Read the persisted profile to preserve legacy delegation until the
    # operator has explicitly assigned this role through the TUI.
    profile = service.profile("export")
    settings = next((r for r in profile.get("roles", []) if r.get("role") == assignment_role), None)
    if not settings:
        return defaults, None

    state = service.status(refresh=False)
    local_only = bool(state.get("local_only") or (state.get("policy") or {}).get("local_only"))

    selected = service.resolve(assignment_role)
    if selected is None:
        if settings.get("mode") == "pinned":
            raise ValueError(f"pinned model for {assignment_role} is unavailable")
        # An explicitly saved auto assignment asks the registry to choose a
        # model. The registry chooses only permitted candidates; falling back
        # to an inherited paid-cloud default would violate that assignment
        # even when the profile has no additional spending policy.
        raise ValueError(f"no approved model under the active routing policy for {assignment_role}")

    if local_only and not _native_local_model(selected):
        raise ValueError("local-only delegation cannot use an unverified proxy or cloud model")
    result = _credentials_for_model(selected, parent_agent, credential_resolver)
    known = {m.get("id"): m for m in state.get("models", []) if isinstance(m, dict)}
    if declared_fallback is not None:
        if local_only and not _explicit_local_fallbacks(declared_fallback, list(known.values())):
            raise ValueError("local-only delegation cannot use an unverified fallback chain")
        # A backend-owned chain outranks unused saved fallback IDs. A stale
        # saved fallback must not displace an explicit task route.
        return result, declared_fallback
    fallbacks: list[dict] = []
    for model_id in settings.get("fallback_ids") or []:
        candidate = known.get(model_id)
        if candidate is None:
            raise ValueError(f"fallback for {assignment_role} is missing from the registry")
        if local_only and not _native_local_model(candidate):
            raise ValueError("local-only delegation cannot fall back to cloud inference")
        if state.get("policy") and not service._allowed(candidate, state["policy"], frozenset(), None):
            raise ValueError("delegation fallback violates active routing policy")
        fb = _credentials_for_model(candidate, parent_agent, credential_resolver)
        fallbacks.append(fb)
    return result, fallbacks
