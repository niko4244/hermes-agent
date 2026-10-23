"""Native TUI model-discovery and per-agent routing RPCs.

The registry owns user profiles and role assignments; the gateway owns live
session changes.  Keep the two operations distinct so a failed live model
switch never persists a misleading selected model.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


def _selected_model(record: dict) -> str:
    """Format a registry model for Hermes' existing /model switch parser."""
    model = record.get("model")
    provider = record.get("provider")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("selected model has no executable model ID")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("selected model has no provider")
    # Hermes switches via --provider; colon is part of valid model IDs.
    if any(c in model for c in ("\x00", "\r", "\n", " --")) or any(c in provider for c in (" ", "\x00", "\r", "\n", "--")):
        raise ValueError("invalid provider or model identity")
    return f"{model} --provider {provider} --session"


@method("orchestration.status")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().status(refresh=bool(params.get("refresh", False))))
    except (OSError, ValueError, TypeError) as exc:
        return _err(rid, 5070, f"Model registry unavailable: {type(exc).__name__}")


@method("orchestration.refresh")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().status(refresh=True))
    except (OSError, ValueError, TypeError) as exc:
        return _err(rid, 5070, f"Model registry refresh failed: {type(exc).__name__}")


@method("orchestration.assign")
def _(rid, params: dict) -> dict:
    from hermes_cli.universal_models.service import ModelService

    role = params.get("role")
    mode = params.get("mode")
    selected_id = params.get("model_id")
    fallbacks = params.get("fallback_ids")
    if not isinstance(role, str) or not role or not isinstance(mode, str):
        return _err(rid, 4002, "role and mode are required")
    if mode not in {"auto", "pinned"}:
        return _err(rid, 4002, "mode must be auto or pinned")
    if mode == "pinned" and (not isinstance(selected_id, str) or not selected_id):
        return _err(rid, 4002, "a pinned model_id is required")
    if fallbacks is not None and (not isinstance(fallbacks, list) or any(not isinstance(i, str) for i in fallbacks)):
        return _err(rid, 4002, "fallback_ids must be model IDs")

    try:
        service = ModelService()
        switch = None
        # Let the model-switch controller validate the actual executable
        # provider and queue a running session's change at a safe turn boundary.
        # A profile-only assignment has no active main session to switch.
        if role == "main" and mode == "pinned":
            state = service.status()
            models = state.get("models", [])
            model = next((m for m in models if m.get("id") == selected_id), None)
            if model is None:
                return _err(rid, 4004, "model is not in the current registry")
            if model.get("availability") not in {"configured", "reachable"}:
                return _err(rid, 4004, "selected model is offline")
            # A UI selection must satisfy the policy *before* invoking the
            # live switch. The persisted registry resolver applies the same
            # check at the next child/session build.
            if not service._allowed(model, state.get("policy") or {}, frozenset(), None):
                return _err(rid, 4004, "selected model is excluded by the active routing policy")
            all_ids = {item.get("id") for item in models}
            if fallbacks and (len(fallbacks) != len(set(fallbacks)) or any(item not in all_ids for item in fallbacks)):
                return _err(rid, 4004, "fallback model is missing or repeated")
            if model.get("source") == "ollama-live":
                # Observed native Ollama supports a direct keyless session
                # switch. Non-Ollama proxies use the existing credentialed
                # model switch and never inherit a cloud key by accident.
                from tui_gateway.universal_model_runtime import main_model_override

                main_model_override(model)
                live_session = _sessions.get(params.get("session_id", ""))
                if live_session is not None and not live_session.get("running"):
                    _apply_universal_local_switch(params.get("session_id", ""), live_session, model)
                switch = {"result": {"deferred": bool(live_session and live_session.get("running"))}}
            else:
                from tui_gateway.methods_orchestration import _selected_model

                switch = _methods["config.set"](
                    rid,
                    {
                        "key": "model",
                        "value": _selected_model(model),
                        "session_id": params.get("session_id", ""),
                    },
                )
                if switch.get("error"):
                    return switch
                if (switch.get("result") or {}).get("confirm_required"):
                    return _ok(rid, {"confirm_required": True, "warning": "Provider requires explicit cost confirmation before selecting"})
        assigned = service.assign(role, mode, selected_id, fallbacks)
        if switch is not None:
            assigned = dict(assigned)
            assigned["deferred"] = bool((switch.get("result") or {}).get("deferred"))
            if assigned["deferred"] and role == "main" and model.get("source") == "ollama-live":
                live_session.pop("pending_model_switch", None)
                live_session["pending_universal_model_switch"] = selected_id
        return _ok(rid, assigned)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        # Avoid reflecting config secrets or credentialed URLs in RPC errors.
        return _err(rid, 5071, f"Model assignment failed: {type(exc).__name__}")


@method("orchestration.profile")
def _(rid, params: dict) -> dict:
    action = params.get("action")
    if action not in {"activate", "save", "import", "export", "list", "delete"}:
        return _err(rid, 4002, "unsupported profile action")
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().profile(action, params.get("name"), params.get("data")))
    except (ValueError, TypeError, OSError, KeyError) as exc:
        return _err(rid, 5072, f"Model profile failed: {type(exc).__name__}")


def register(server) -> None:
    _registry.install(server)
