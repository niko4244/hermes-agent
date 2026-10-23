"""Validate registry model selection before constructing a TUI agent.

Existing /model explicit session selections and restored session histories take
precedence over the persisted universal-model profile. This adapter adds only
the new-profile tier; actual provider requests remain in Hermes' transports.
"""

from __future__ import annotations

from typing import Any

from tools.delegation_model_router import _loopback_endpoint


def _active_registry() -> Any:
    """Do not create a model sidecar on ordinary, unconfigured Hermes turns."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    if not (home / "universal_models.sqlite3").is_file():
        return None
    from hermes_cli.universal_models.service import ModelService

    return ModelService(home=home, local_endpoints=[], model_paths=[])


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
            "_universal_local": True, "_universal_selection": True,
            "_universal_model_id": record.get("id"),
        }
    if provider == "custom" and _loopback_endpoint(endpoint):
        # A generic OpenAI-compatible proxy may route to paid external
        # services. Let Hermes require an explicitly configured provider.
        raise ValueError("non-Ollama endpoint must be registered in Hermes")
    if provider in {"litellm", "custom"}:
        raise ValueError("unregistered proxy provider")
    return {"model": model, "provider": provider, "_universal_selection": True,
            "_universal_model_id": record.get("id")}


def selected_profile_main_override(service: Any = None) -> dict | None:
    """Resolve an explicitly configured main role at new-agent build time.

    Fresh installs have a UI-visible default ``auto`` role for discoverability,
    but only a role saved by the operator should change Hermes' old default.
    Automatic routing only selects safe local candidates approved by the
    registry policy, so it cannot silently incur remote inference charges.
    """
    if service is None:
        service = _active_registry()
        if service is None:
            return None
    stored = service.profile("export")
    settings = next((r for r in stored.get("roles", []) if r.get("role") == "main"), None)
    if not settings or settings.get("mode") not in {"pinned", "manual", "auto"}:
        return None
    model = service.resolve("main")
    if model is None:
        # A saved auto choice is an explicit opt-in to registry-based routing.
        # With no approved local candidate, never silently fall back to the
        # previous (potentially billable) global default.
        raise ValueError("selected main model is currently unavailable under the active routing policy")
    return main_model_override(model)


def selected_profile_main_fallbacks(service: Any = None) -> list[dict] | None:
    """Return only explicit, policy-approved main-agent fallback entries.

    ``None`` means no universal main pin and preserves legacy fallbacks.
    A pinned role with an empty configured chain gets ``[]`` so its agent
    cannot unexpectedly move to a global paid provider on error.
    """
    if service is None:
        service = _active_registry()
        if service is None:
            return None
    state = service.status()
    settings = next((r for r in state.get("roles", []) if r.get("role") == "main"), None)
    if not settings or settings.get("mode") not in {"pinned", "auto"}:
        return None
    fallback_ids = settings.get("fallback_ids") or []
    if not fallback_ids:
        return []

    return approved_main_fallbacks_for_ids(service, fallback_ids, state=state)


def approved_main_fallbacks_for_ids(
    service: Any, fallback_ids: list[str], *, state: dict | None = None,
) -> list[dict]:
    """Preflight a proposed fallback chain before a live model switch."""
    if state is None:
        state = service.status()
    index = {model.get("id"): model for model in state.get("models", [])}

    from tools.delegation_model_router import _credentials_for_model
    from tools.delegate_tool import _resolve_delegation_credentials

    result: list[dict] = []
    for model_id in fallback_ids:
        row = index.get(model_id)
        if not isinstance(row, dict) or not service._allowed(
            row, state.get("policy") or {}, frozenset(), None
        ):
            raise ValueError("main fallback model is unavailable or excluded by policy")
        # A localhost proxy is not verified native local inference, even when
        # its /v1/models endpoint returns model names. Never route a local-only
        # fallback to a server that could be forwarding to a paid upstream.
        if (state.get("policy") or {}).get("local_only") and (row.get("source") != "ollama-live" or not _loopback_endpoint(row.get("endpoint"))):
            raise ValueError("main fallback violates local-only routing")
        result.append(_credentials_for_model(row, None, _resolve_delegation_credentials))
    return result


def enforce_main_turn_policy(agent: Any, session: dict, service: Any = None) -> None:
    """Check active profile restrictions before a main-agent inference turn.

    A new policy may be activated while a session stays open. Recheck the
    current runtime after queued switches and config synchronization so a
    local-only/cost-capped profile cannot authorize a previously pinned cloud
    agent merely because its runtime was built under an older profile.
    """
    if session.get("_universal_model_rollback_failed"):
        raise ValueError("Model assignment rollback failed; rebuild the session before inference")
    if service is None:
        service = _active_registry()
        if service is None:
            if isinstance(session.get("model_override"), dict) and session["model_override"].get("_universal_selection"):
                raise ValueError("Saved model selection no longer has an active routing profile")
            return
    # Read the active profile directly. Scanning configured/disk model roots
    # on *every* turn would add unrelated I/O and latency to ordinary chats.
    active_profile = service.profile("export")
    policy = active_profile.get("policy") or {}
    saved_main = next((role for role in active_profile.get("roles", [])
                       if role.get("role") == "main"), None)
    saved_ids = ((saved_main.get("fallback_ids") or [])
                 if saved_main and saved_main.get("mode") in {"pinned", "auto"} else [])
    override = session.get("model_override")
    if isinstance(override, dict) and override.get("_universal_selection") and isinstance(active_profile.get("roles"), list):
        if saved_main is None or (saved_main.get("mode") == "pinned" and
                                  saved_main.get("model_id") != override.get("_universal_model_id")):
            # A profile activation or later pin edit invalidates a session's
            # prior role authorization; otherwise an old cloud agent could
            # continue although the user selected a different profile.
            raise ValueError("Active profile no longer authorizes this session model")
    if isinstance(override, dict) and override.get("_universal_selection") and not getattr(agent, "_fallback_activated", False):
        selected_provider = override.get("provider")
        current_provider = str(getattr(agent, "provider", "") or "")
        if isinstance(selected_provider, str) and selected_provider.startswith("custom:"):
            # Runtime providers use Hermes' configured key, while registry
            # records use the discovery namespace. Keep the configured origin
            # bound even if another named provider serves the same model slug.
            from tools.delegation_model_router import _registered_custom_provider

            configured = _registered_custom_provider(selected_provider)
            provider_matches = (configured is not None
                                and current_provider in {"custom", configured[0]}
                                and getattr(agent, "requested_provider", None) == configured[0]
                                and str(getattr(agent, "base_url", "") or "").rstrip("/") == configured[1])
        else:
            provider_matches = current_provider == selected_provider
        if (str(getattr(agent, "model", "") or "") != override.get("model")
                or not provider_matches
                or (override.get("_universal_local") and
                    (str(getattr(agent, "base_url", "") or "") != override.get("base_url")
                     or getattr(agent, "api_key", None) != "local"))):
            raise ValueError("Current session does not match its saved model selection")
    if (isinstance(override, dict) and override.get("_universal_selection")
            and saved_main and saved_main.get("mode") in {"pinned", "auto"}
            and getattr(agent, "_provider_fallback_active", False)):
        # A fallback may remain active between turns when the primary route
        # cannot be restored (for example after an entitlement rejection).
        # Profile edits revoke prior fallback IDs even when no local-only or
        # price cap is configured; a refreshed future chain alone is not enough.
        authorized = (approved_main_fallbacks_for_ids(service, saved_ids)
                      if saved_ids else [])
        current_model = str(getattr(agent, "model", "") or "")
        current_provider = str(getattr(agent, "provider", "") or "")
        current_endpoint = str(getattr(agent, "base_url", "") or "").rstrip("/")
        if not any(
            entry.get("model") == current_model
            and entry.get("provider") == current_provider
            and (not entry.get("base_url")
                 or str(entry["base_url"]).rstrip("/") == current_endpoint)
            and (entry.get("api_key") != "local" or getattr(agent, "api_key", None) == "local")
            for entry in authorized
        ):
            raise ValueError("Active profile no longer authorizes this session fallback")
    if not (policy.get("local_only") is True or
            "max_input_cost_per_million" in policy or
            "max_output_cost_per_million" in policy):
        # A user choosing local_only=False has *disabled* the restriction.
        # Preserve the host's legacy provider/fallback behavior in that case.
        return

    runtime_model = str(getattr(agent, "model", "") or "")
    runtime_provider = str(getattr(agent, "provider", "") or "")
    runtime_endpoint = str(getattr(agent, "base_url", "") or "")

    if policy.get("local_only"):
        verified_local = (
            isinstance(override, dict)
            and override.get("_universal_local") is True
            and override.get("model") == runtime_model
            and runtime_provider in {"custom", "ollama"}
            and override.get("provider") == "custom"
            and override.get("base_url") == runtime_endpoint
            and _loopback_endpoint(runtime_endpoint)
            and getattr(agent, "api_key", None) == "local"
        )
        if not verified_local and getattr(agent, "_fallback_activated", False):
            # A genuine, explicitly approved native-Ollama fallback is still
            # local. Do not reject it merely because its model differs from
            # the pinned primary; do not trust an arbitrary loopback proxy.
            verified_local = any(
                row.get("id") in saved_ids
                and row.get("source") == "ollama-live"
                and row.get("model") == runtime_model
                and row.get("endpoint") == runtime_endpoint
                and runtime_provider in {"custom", "ollama"}
                and getattr(agent, "api_key", None) == "local"
                and service._allowed(row, policy, frozenset(), None)
                for row in service.status(refresh=False).get("models", [])
            )
        if not verified_local:
            raise ValueError("Active profile requires an approved local model before inference")

    if "max_input_cost_per_million" in policy or "max_output_cost_per_million" in policy:
        state = service.status(refresh=False)
        # Only provider-verified pricing can satisfy a cap. A zero price is not
        # inferred from a loopback address or absent price metadata.
        candidate = next((row for row in state.get("models", []) if
                          row.get("model") == runtime_model
                          and row.get("provider") == runtime_provider
                          and (not row.get("endpoint") or row.get("endpoint") == runtime_endpoint)), None)
        if candidate is None or not service._allowed(candidate, policy, frozenset(), None):
            raise ValueError("Active profile cost limit cannot verify this model's price")

    # Provider fallback is executed later inside Hermes' conversation loop.
    # Reconcile its chain with *current* policy here, otherwise a local agent
    # created under an older profile can silently fail over to a paid provider
    # after local-only is enabled during an unrelated turn.
    old_chain = getattr(agent, "_fallback_chain", None) or []
    if old_chain or saved_ids:
        authorized = approved_main_fallbacks_for_ids(service, saved_ids) if saved_ids else []
        if old_chain != authorized:
            agent._fallback_chain = authorized
            agent._fallback_index = 0
            agent._unavailable_fallback_keys = set()
