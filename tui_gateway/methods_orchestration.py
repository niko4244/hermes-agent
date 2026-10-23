"""Native TUI model-discovery and per-agent routing RPCs.

The registry owns user profiles and role assignments; the gateway owns live
session changes.  Keep the two operations distinct so a failed live model
switch never persists a misleading selected model.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _selected_model(record: dict) -> str:
    """Format a registry model for Hermes' existing /model switch parser."""
    model = record.get("model")
    provider = record.get("provider")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("selected model has no executable model ID")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("selected model has no provider")
    # Hermes switches via --provider; colon is part of valid model IDs.
    if any(c.isspace() for c in model + provider) or model.startswith("-") or provider.startswith("-"):
        raise ValueError("invalid provider or model identity")
    return f"{model} --provider {provider} --session"


def _orchestration_session_error(rid: str, params: dict) -> dict | None:
    """Refuse a stale/mismatched scope rather than using launch HERMES_HOME.

    The common profile wrapper supports legacy fallback for unknown sessions;
    model-management requests cannot safely inherit that behavior.
    """
    session_id = params.get("session_id")
    if session_id is None and ("session_id" not in params or params["session_id"] is None):
        return None
    if not isinstance(session_id, str) or session_id not in _sessions:
        return _err(rid, 4001, "orchestration session is not active")
    if params.get("profile"):
        return _err(rid, 4001, "use either a session or an explicit profile")
    return None


@method("orchestration.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    if error := _orchestration_session_error(rid, params):
        return error
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().status(refresh=bool(params.get("refresh", False))))
    except (OSError, ValueError, TypeError) as exc:
        return _err(rid, 5070, f"Model registry unavailable: {type(exc).__name__}")


@method("orchestration.refresh")
@_profile_scoped
def _(rid, params: dict) -> dict:
    if error := _orchestration_session_error(rid, params):
        return error
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().status(refresh=True))
    except (OSError, ValueError, TypeError) as exc:
        return _err(rid, 5070, f"Model registry refresh failed: {type(exc).__name__}")


@method("orchestration.assign")
@_profile_scoped
def _(rid, params: dict) -> dict:
    if error := _orchestration_session_error(rid, params):
        return error
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
        session_id = params.get("session_id")
        if session_id:
            # A stale/foreign session must not turn a profile edit into a
            # switch of an unrelated live agent or global config change.
            if session_id not in _sessions:
                return _err(rid, 4001, "session is not active")
            if params.get("profile"):
                return _err(rid, 4001, "use either a session or an explicit profile")
        live_session = _sessions.get(session_id) if session_id else None
        rollback = None
        if (role == "main" and mode == "pinned" and isinstance(live_session, dict)
                and not live_session.get("running") and live_session.get("agent") is not None):
            agent = live_session["agent"]
            rollback = {
                "runtime": _snapshot_agent_model_runtime(agent),
                "fallbacks": list(getattr(agent, "_fallback_chain", []) or []),
                "override_set": "model_override" in live_session,
                "override": live_session.get("model_override"),
            }
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
            from tui_gateway.universal_model_runtime import (
                approved_main_fallbacks_for_ids, main_model_override,
            )
            # Validate the entire proposed runtime before changing any live
            # session or saving a profile.
            main_model_override(model)
            approved_fallbacks = approved_main_fallbacks_for_ids(
                service, fallbacks or [], state=state)
            if model.get("source") == "ollama-live":
                if live_session is not None and not live_session.get("running"):
                    if live_session.get("agent") is not None:
                        _apply_universal_local_switch(
                            session_id, live_session, model,
                            approved_fallbacks=approved_fallbacks)
                switch = {"result": {"deferred": not (live_session and live_session.get("agent") is not None
                                                        and not live_session.get("running"))}}
            elif live_session is not None and not live_session.get("running") and live_session.get("agent") is not None:
                # Use the existing per-session transactional switch, never
                # config.set (which is global/session selection UI machinery).
                result = _apply_model_switch(
                    session_id, live_session, _selected_model(model),
                    persist_override=False)
                if result.get("confirm_required"):
                    return _ok(rid, {"confirm_required": True,
                                     "warning": "Provider requires explicit cost confirmation before selecting"})
                from gateway.run import GatewayRunner
                GatewayRunner._apply_fallback_chain_to_agent(live_session["agent"], approved_fallbacks)
                live_session["model_override"] = main_model_override(model)
                switch = {"result": {"deferred": False}}
            else:
                switch = {"result": {"deferred": True}}
        try:
            assigned = service.assign(role, mode, selected_id, fallbacks)
        except Exception:
            # The live swap is prepared before persistence. If the profile
            # write fails, restore the previous runtime and session provenance;
            # otherwise an RPC error would leave a different model running.
            if rollback is not None:
                agent = live_session["agent"]
                try:
                    _restore_agent_model_runtime(agent, rollback["runtime"])
                    from gateway.run import GatewayRunner
                    GatewayRunner._apply_fallback_chain_to_agent(agent, rollback["fallbacks"])
                    if rollback["override_set"]:
                        live_session["model_override"] = rollback["override"]
                    else:
                        live_session.pop("model_override", None)
                    live_session.pop("_universal_model_rollback_failed", None)
                    _persist_live_session_runtime(live_session)
                except Exception:
                    # Even a failed rollback must never send a subsequent
                    # request using the unconfirmed new route.
                    live_session["_universal_model_rollback_failed"] = True
            raise
        if role == "main" and mode == "auto":
            assigned = dict(assigned)
            assigned["deferred"] = True
            assigned["warning"] = "Automatic main routing applies when a new agent session starts"
        if switch is not None:
            assigned = dict(assigned)
            assigned["deferred"] = bool((switch.get("result") or {}).get("deferred"))
            if assigned["deferred"] and role == "main" and live_session is not None:
                if live_session.get("running"):
                    live_session.pop("pending_model_switch", None)
                    live_session["pending_universal_model_switch"] = selected_id
                elif live_session.get("agent") is None:
                    # A deferred build must use the newly persisted profile.
                    live_session.pop("model_override", None)
        return _ok(rid, assigned)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        # Avoid reflecting config secrets or credentialed URLs in RPC errors.
        return _err(rid, 5071, f"Model assignment failed: {type(exc).__name__}")


@method("orchestration.profile")
@_profile_scoped
def _(rid, params: dict) -> dict:
    if error := _orchestration_session_error(rid, params):
        return error
    action = params.get("action")
    if action not in {"activate", "save", "import", "export", "list", "delete"}:
        return _err(rid, 4002, "unsupported profile action")
    try:
        from hermes_cli.universal_models.service import ModelService

        return _ok(rid, ModelService().profile(action, params.get("name"), params.get("data")))
    except (ValueError, TypeError, OSError, KeyError) as exc:
        return _err(rid, 5072, f"Model profile failed: {type(exc).__name__}")


@method("orchestration.policy")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Read or atomically replace routing limits for the active profile."""
    if error := _orchestration_session_error(rid, params):
        return error
    action = params.get("action", "get")
    if action not in {"get", "set"}:
        return _err(rid, 4002, "policy action must be get or set")
    if action == "set" and not isinstance(params.get("policy"), dict):
        return _err(rid, 4002, "policy must be a JSON object")
    try:
        from hermes_cli.universal_models.service import ModelService

        service = ModelService()
        if action == "set":
            # ModelService.profile validates the complete profile (including
            # unknown keys and negative cost caps) before writing it.
            current = service.profile("export")
            replacement = {
                "name": current["name"], "roles": current["roles"],
                "policy": params["policy"],
            }
            service.profile("save", current["name"], replacement)
        result = service.status(refresh=False)
        return _ok(rid, {"policy": result.get("policy", {}), "active_profile": result.get("active_profile")})
    except (ValueError, TypeError, OSError, KeyError):
        return _err(rid, 5073, "Model policy update failed; review policy fields")


@method("orchestration.qualify")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """One explicitly requested, bounded, already-loaded Ollama inference test.

    Running this RPC never enumerates or calls any cloud model. The concrete
    model identity must have a recent native Ollama observation, and the
    service independently rechecks /api/tags plus /api/ps before generation.
    """
    if error := _orchestration_session_error(rid, params):
        return error
    model_id = params.get("model_id")
    max_tokens = params.get("max_tokens", 16)
    timeout = params.get("timeout_seconds", 15)
    if not isinstance(model_id, str) or not 0 < len(model_id) <= 650:
        return _err(rid, 4002, "model_id must identify a discovered model")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 64:
        return _err(rid, 4002, "max_tokens must be 1–64")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 30:
        return _err(rid, 4002, "timeout_seconds must be 1–30")
    try:
        from hermes_cli.universal_models.service import ModelService

        service = ModelService()
        candidates = service.status(refresh=False).get("models", [])
        item = next((m for m in candidates if m.get("id") == model_id), None)
        if item is None or item.get("source") != "ollama-live" or item.get("availability") != "reachable":
            return _err(rid, 4004, "refresh a native local Ollama model before qualification")
        from hermes_cli.universal_models.qualification import _valid_origin

        if _valid_origin(item) is None:
            return _err(rid, 4004, "only direct loopback Ollama inference can be qualified")
        result = service.qualify(
            model_id, run_inference=True, timeout_seconds=float(timeout),
            max_tokens=max_tokens, require_loaded=True,
        )
        return _ok(rid, result)
    except (ValueError, TypeError, OSError, KeyError):
        # HTTP failures, URLs or exception messages may contain credentials;
        # report a stable generic error instead of reflecting raw metadata.
        return _err(rid, 5074, "Local model qualification failed; check Ollama availability")


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
    # The generic profile-scope wrapper chooses a named profile *before*
    # entering the handler. Validate orchestration's mutually exclusive
    # session/profile authority outside that wrapper, otherwise an invalid
    # profile can be consulted before the inner guard runs.
    for name in _registry.names():
        scoped = server._methods[name]

        def guarded(rid, params, scoped=scoped):
            p = params if isinstance(params, dict) else {}
            if error := server._orchestration_session_error(rid, p):
                return error
            return scoped(rid, params)

        server.register_method(name, guarded)
