"""Profile-scoped, credential-free model assignments and policy resolution.

The registry is deliberately a sidecar. Hermes's existing runtime provider
resolver remains the authority for credentials, transports and API requests.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from hermes_constants import get_hermes_home

from .discovery import (
    config_aliases,
    configured_models,
    hardware_summary,
    litellm_models,
    local_server_models,
    merge_models,
    on_disk_models,
    safe_endpoint,
)
from .qualification import qualify_native_ollama


_PROFILE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_ROLE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")
_ALLOWED_MODES = {"pinned", "auto"}
_VALID_AVAILABILITY = {"configured", "reachable"}
_MAX_FALLBACKS = 16


def _require_identifier(value: Any, pattern: re.Pattern, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}; expected letters/digits, dots, underscores or hyphens")
    return value


def _parse_policy(data: Any) -> dict:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("policy must be an object")
    if set(data) - {"local_only", "max_input_cost_per_million", "max_output_cost_per_million"}:
        raise ValueError("unknown policy field")
    result: dict[str, Any] = {}
    if "local_only" in data:
        if not isinstance(data["local_only"], bool):
            raise ValueError("local_only must be a boolean")
        result["local_only"] = data["local_only"]
    for key in ("max_input_cost_per_million", "max_output_cost_per_million"):
        if key in data:
            number = data[key]
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not 0 <= number < float("inf"):
                raise ValueError(f"{key} must be a finite non-negative number")
            result[key] = float(number)
    return result


def _parse_assignment(item: Any) -> dict:
    if not isinstance(item, dict) or set(item) - {"role", "mode", "model_id", "fallback_ids"}:
        raise ValueError("assignment must contain only role/mode/model_id/fallback_ids")
    role = _require_identifier(item.get("role"), _ROLE_RE, "role")
    mode = item.get("mode")
    if mode == "manual":
        mode = "pinned"  # compatibility for early API clients
    if mode not in _ALLOWED_MODES:
        raise ValueError("mode must be pinned or auto")
    model_id = item.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not 0 < len(model_id) <= 650 or any(ord(c) < 32 for c in model_id)):
        raise ValueError("model_id is invalid")
    if mode == "pinned" and not model_id:
        raise ValueError("pinned assignments require model_id")
    fallbacks = item.get("fallback_ids", [])
    if not isinstance(fallbacks, list) or len(fallbacks) > _MAX_FALLBACKS or any(
        not isinstance(candidate, str) or not 0 < len(candidate) <= 650 or any(ord(c) < 32 for c in candidate)
        for candidate in fallbacks
    ):
        raise ValueError("fallback_ids must be a list of at most 16 model IDs")
    if len(set(fallbacks)) != len(fallbacks):
        raise ValueError("fallback_ids must be unique")
    return {"role": role, "mode": mode, "model_id": model_id, "fallback_ids": list(fallbacks)}


def _parse_profile(data: Any) -> dict:
    if not isinstance(data, dict) or set(data) - {"name", "roles", "policy"}:
        raise ValueError("profile accepts only name, roles and policy; credentials are never stored")
    roles = data.get("roles", [])
    if not isinstance(roles, list) or len(roles) > 200:
        raise ValueError("roles must be a list with at most 200 assignments")
    parsed = [_parse_assignment(row) for row in roles]
    if len({row["role"] for row in parsed}) != len(parsed):
        raise ValueError("duplicate role in profile")
    return {"roles": parsed, "policy": _parse_policy(data.get("policy"))}


class ModelService:
    """Native TUI and gateway API for discovery, profiles and deterministic routing.

    New settings live in a SQLite DB beneath profile-aware HERMES_HOME.
    No provider tokens, API keys, cookies or raw provider configuration is saved.
    Discovery is offline by default; refresh explicitly probes loopback only.
    """

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        local_endpoints: list[str] | tuple[str, ...] | None = None,
        litellm_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    ) -> None:
        self.home = Path(home) if home is not None else get_hermes_home()
        self.db_path = self.home / "universal_models.sqlite3"
        self._config_override = config
        self.local_endpoints = tuple(local_endpoints) if local_endpoints is not None else (
            "http://127.0.0.1:11434", "http://127.0.0.1:1234", "http://127.0.0.1:8000"
        )
        self.litellm_paths = tuple(Path(p) for p in litellm_paths) if litellm_paths is not None else (
            self.home / "litellm.yaml", self.home / "litellm" / "config.yaml",
            Path.home() / ".config" / "litellm" / "config.yaml",
            Path.home() / "litellm" / "config.yaml",
        )
        # Explicit home overrides are used by isolated tests/other profiles;
        # only a real operator session should scan default user model roots.
        self.model_paths = tuple(Path(p) for p in model_paths) if model_paths is not None else (
            (Path.home() / "models", Path.home() / ".ollama" / "models" / "manifests")
            if home is None else ()
        )
        self._live: list[dict] = []
        self._hardware: dict | None = None
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.db_path.is_symlink():
            raise ValueError("refusing symlinked model registry")
        connection = sqlite3.connect(self.db_path, timeout=5)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            # sqlite3.Connection.__exit__ commits/rolls back but does not
            # close the handle. This must close deterministically: gateway RPCs
            # create short-lived instances and Windows locks even an unused
            # registry database until the connection is finalized by the GC.
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        if self.home.is_symlink():
            raise ValueError("refusing symlinked Hermes home")
        self.home.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS profiles (name TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS observations (name TEXT PRIMARY KEY, payload TEXT NOT NULL, measured_at REAL NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS qualifications (model_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO profiles(name,payload) VALUES (?,?)", ("default", json.dumps({"roles": [], "policy": {}})))
            conn.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES ('active_profile','default')")
        if os.name != "nt":
            self.db_path.chmod(0o600)

    def _config(self) -> Mapping[str, Any]:
        if self._config_override is not None:
            return self._config_override
        from hermes_cli.config import load_config_readonly

        return load_config_readonly()

    def _active(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key='active_profile'").fetchone()
        return row[0] if row else "default"

    def _load(self, name: str | None = None) -> dict:
        selected = _require_identifier(name or self._active(), _PROFILE_RE, "profile name")
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM profiles WHERE name=?", (selected,)).fetchone()
        if row is None:
            raise KeyError(f"unknown profile: {selected}")
        parsed = _parse_profile(json.loads(row[0]))
        return {"name": selected, **parsed}

    def _save(self, name: str, payload: dict) -> dict:
        selected = _require_identifier(name, _PROFILE_RE, "profile name")
        validated = _parse_profile(payload)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO profiles(name,payload) VALUES (?,?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                (selected, json.dumps(validated, sort_keys=True, separators=(",", ":"))),
            )
        return {"name": selected, **validated}

    def _inventory(self) -> list[dict]:
        config = self._config()
        rows = configured_models(config) + litellm_models(self.litellm_paths, config)
        rows += on_disk_models(self.model_paths)
        rows += self._observations()
        merged = merge_models(rows)
        with self._connect() as conn:
            details = dict(conn.execute("SELECT model_id,payload FROM qualifications"))
        for row in merged:
            payload = details.get(row["id"])
            if payload is not None:
                try:
                    record = json.loads(payload)
                except (ValueError, TypeError):
                    continue
                if isinstance(record, dict):
                    # Historic evidence is visibly timestamped; it never
                    # upgrades an expired or configured row to 'reachable'.
                    row["last_qualification"] = record
        return merged

    def _probe_endpoints(self) -> tuple[str, ...]:
        """Include the operator's explicitly configured local endpoints."""
        config = self._config()
        endpoints = set(self.local_endpoints)
        active = config.get("model")
        candidates: list[Any] = [active.get("base_url") if isinstance(active, dict) else None]
        try:
            from hermes_cli.config import get_compatible_custom_providers

            candidates.extend(entry.get("base_url") for entry in get_compatible_custom_providers(dict(config)))
        except (ImportError, TypeError, ValueError):
            pass
        from urllib.parse import urlsplit

        for candidate in candidates:
            endpoint = safe_endpoint(candidate)
            if endpoint and urlsplit(endpoint).hostname in {"127.0.0.1", "localhost", "::1"}:
                endpoints.add(endpoint)
        return tuple(sorted(endpoints))

    def _recent(self, name: str) -> Any:
        with self._connect() as conn:
            observation = conn.execute("SELECT payload,measured_at FROM observations WHERE name=?", (name,)).fetchone()
        if not observation or not 0 <= time.time() - observation[1] <= 120:
            return None
        try:
            return json.loads(observation[0])
        except (ValueError, TypeError):
            return None

    def _observations(self) -> list[dict]:
        # TUI RPC handlers construct fresh ModelService instances per request.
        # Persist only metadata, with a short freshness window; a model that
        # answered /models yesterday must not still appear executable today.
        rows = self._recent("local-models")
        return rows if isinstance(rows, list) else []

    def status(self, refresh: bool = False) -> dict:
        if refresh:
            self._live = local_server_models(self._probe_endpoints())
            self._hardware = hardware_summary()
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO observations(name,payload,measured_at) VALUES ('local-models',?,?) "
                    "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, measured_at=excluded.measured_at",
                    (json.dumps(self._live, sort_keys=True), time.time()),
                )
                conn.execute(
                    "INSERT INTO observations(name,payload,measured_at) VALUES ('hardware',?,?) "
                    "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload, measured_at=excluded.measured_at",
                    (json.dumps(self._hardware, sort_keys=True), time.time()),
                )
        config = self._config()
        with self._connect() as conn:
            profile_names = [row[0] for row in conn.execute("SELECT name FROM profiles ORDER BY name")]
        selected = self._load()
        saved_roles = {row["role"]: row for row in selected["roles"]}
        host_roles = ["main", "delegation.leaf", "delegation.orchestrator"]
        auxiliary = config.get("auxiliary")
        if isinstance(auxiliary, dict):
            host_roles.extend(
                f"auxiliary.{name}"
                for name, settings in auxiliary.items()
                if isinstance(name, str) and _ROLE_RE.fullmatch(name) and isinstance(settings, dict)
                and ("provider" in settings or "model" in settings)
            )
        roles = [saved_roles.get(role, {"role": role, "mode": "auto", "model_id": None, "fallback_ids": []}) for role in host_roles]
        roles.extend(row for role, row in sorted(saved_roles.items()) if role not in host_roles)
        return {
            "models": self._inventory(),
            "roles": roles,
            "profiles": [{"name": name, "active": name == selected["name"]} for name in profile_names],
            "active_profile": selected["name"],
            "policy": selected["policy"],
            "local_only": bool(selected["policy"].get("local_only", False)),
            "aliases": config_aliases(config),
            "hardware": self._hardware if self._hardware is not None else (self._recent("hardware") or {"gpu": [], "gpu_discovery": "not_measured"}),
        }

    def refresh(self) -> dict:
        return self.status(refresh=True)

    def assign(self, role: str, mode: str, model_id: str | None = None, fallback_ids: list[str] | None = None) -> dict:
        assignment = _parse_assignment({"role": role, "mode": mode, "model_id": model_id, "fallback_ids": fallback_ids if fallback_ids is not None else []})
        # Retain assignments to discoverable offline models for later activation,
        # while resolve() refuses dispatch until a valid executable source exists.
        candidates = {row["id"] for row in self._inventory()}
        ids = [x for x in [assignment["model_id"], *assignment["fallback_ids"]] if x]
        if any(x not in candidates for x in ids):
            raise ValueError("model_id and fallbacks must refer to discovered model IDs")
        name = self._active()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = conn.execute("SELECT payload FROM profiles WHERE name=?", (name,)).fetchone()
            if record is None:
                raise KeyError(f"unknown profile: {name}")
            profile = _parse_profile(json.loads(record[0]))
            profile["roles"] = [row for row in profile["roles"] if row["role"] != role] + [assignment]
            profile["roles"].sort(key=lambda row: row["role"])
            conn.execute("UPDATE profiles SET payload=? WHERE name=?", (json.dumps(profile, sort_keys=True), name))
        return assignment

    def profile(self, action: str, name: str | None = None, data: dict | None = None) -> dict:
        if action == "list":
            return {"profiles": self.status()["profiles"], "active_profile": self._active()}
        if action == "export":
            return self._load(name)
        if action in {"save", "import"}:
            if action == "import" and data is None:
                raise ValueError("import requires profile data")
            destination = name or (data.get("name") if isinstance(data, dict) else None)
            if destination is None:
                destination = self._active()
            content = data if data is not None else self._load()
            return self._save(destination, content)
        if action == "activate":
            selected = _require_identifier(name, _PROFILE_RE, "profile name")
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute("SELECT 1 FROM profiles WHERE name=?", (selected,)).fetchone() is None:
                    raise KeyError(f"unknown profile: {selected}")
                conn.execute("UPDATE metadata SET value=? WHERE key='active_profile'", (selected,))
            return self._load(selected)
        if action == "delete":
            selected = _require_identifier(name, _PROFILE_RE, "profile name")
            if selected in {"default", self._active()}:
                raise ValueError("cannot delete the default or active profile")
            with self._connect() as conn:
                conn.execute("DELETE FROM profiles WHERE name=?", (selected,))
            return {"deleted": selected}
        raise ValueError("action must be list, save, import, export, activate or delete")

    @staticmethod
    def _allowed(row: dict, policy: dict, required: frozenset[str], min_context: int | None) -> bool:
        if row["availability"] not in _VALID_AVAILABILITY:
            return False
        if required and not required.issubset(row.get("capabilities", [])):
            return False
        if min_context is not None and (
            not isinstance(row.get("context_window"), int) or row["context_window"] < min_context
        ):
            return False
        if policy.get("local_only"):
            # Only a native Ollama instance is identifiable as local execution.
            # An OpenAI-compatible localhost proxy may still call paid cloud APIs.
            if row["source"] != "ollama-live":
                return False
        if any(key in policy for key in ("max_input_cost_per_million", "max_output_cost_per_million")):
            # Discovery does not establish verified pricing; fail closed instead
            # of assuming that a configured or locally proxied model is free.
            for key in ("max_input_cost_per_million", "max_output_cost_per_million"):
                if key in policy and (
                    not isinstance(row.get(key), (float, int)) or row[key] > policy[key]
                ):
                    return False
        return True

    def resolve(self, role: str, require_capabilities: list[str] | tuple[str, ...] | None = None, min_context: int | None = None) -> dict | None:
        """Return a usable row or None. Never silently choose unapproved cloud models."""
        _require_identifier(role, _ROLE_RE, "role")
        if min_context is not None and (not isinstance(min_context, int) or isinstance(min_context, bool) or min_context <= 0):
            raise ValueError("min_context must be a positive integer")
        required = frozenset(require_capabilities or ())
        if any(not isinstance(value, str) or not value for value in required):
            raise ValueError("require_capabilities must contain nonempty strings")
        active = self._load()
        assignments = {row["role"]: row for row in active["roles"]}
        assignment = assignments.get(role)
        if assignment is None and role == "main":
            model = self._config().get("model")
            if isinstance(model, dict):
                provider, name = model.get("provider"), model.get("default") or model.get("name") or model.get("model")
                if provider and name:
                    # Current Hermes model is an explicit operator selection,
                    # so do not override it with an implicit local-auto choice.
                    assignment = {"mode": "pinned", "model_id": f"{provider}::{name}", "fallback_ids": []}
        if assignment is None:
            if role.startswith("delegation.") or role.startswith("auxiliary."):
                # Status displays these untouched built-in roles as auto. Keep
                # runtime resolution consistent with the displayed setting.
                assignment = {"mode": "auto", "model_id": None, "fallback_ids": []}
            else:
                return None
        index = {row["id"]: row for row in self._inventory()}
        if assignment["mode"] == "pinned":
            # An unavailable pin never silently runs another model. Fallback
            # decisions belong to the execution layer after an actual failure.
            candidates = [assignment.get("model_id")]
        else:
            candidates = [assignment.get("model_id"), *assignment.get("fallback_ids", [])]
            if not any(candidates):
                # Default auto discovery may only select independently listed
                # native local Ollama models; never surprise-call paid cloud.
                candidates = [row["id"] for row in sorted(index.values(), key=lambda m: m["id"])
                              if row.get("source") == "ollama-live" and row.get("availability") == "reachable"]
        for model_id in candidates:
            if model_id in index and self._allowed(index[model_id], active["policy"], required, min_context):
                return dict(index[model_id])
        return None

    def qualify(
        self, model_id: str, *, run_inference: bool = False,
        timeout_seconds: float = 15.0, max_tokens: int = 16, require_loaded: bool = True,
    ) -> dict:
        """Explicit bounded inference on a verified *native local Ollama* model.

        For compatibility, `qualify(model_id)` performs listing-only discovery.
        Generation requires `run_inference=True`; it probes the selected native
        Ollama endpoint only, runs one fixed prompt, and stores sanitized metrics.
        """
        if not isinstance(model_id, str) or len(model_id) > 650 or not model_id:
            raise ValueError("model_id must be a nonempty discovered identifier")
        if not isinstance(run_inference, bool):
            raise ValueError("run_inference must be a boolean")
        if not run_inference:
            self.refresh()
            row = next((r for r in self._inventory() if r["id"] == model_id), None)
            return {"model_id": model_id, "available": bool(row and row["availability"] == "reachable"), "level": "model-list-only", "inference_tested": False}
        # No discovery-wide refresh here: that could probe unrelated servers.
        # Only a recently listed, endpoint-scoped Ollama identity is eligible.
        row = next((r for r in self._observations() if r.get("id") == model_id), None)
        if row is None:
            raise ValueError("inference qualification requires a recent explicit local discovery refresh")
        record = qualify_native_ollama(row, timeout_seconds=timeout_seconds,
                                       max_tokens=max_tokens, require_loaded=require_loaded)
        with self._connect() as conn:
            conn.execute("INSERT INTO qualifications(model_id,payload) VALUES (?,?) "
                         "ON CONFLICT(model_id) DO UPDATE SET payload=excluded.payload",
                         (model_id, json.dumps(record, sort_keys=True, separators=(",", ":"))))
        return record

    def qualification(self, model_id: str) -> dict | None:
        """Read historic qualification evidence, including expired local models."""
        if not isinstance(model_id, str) or not model_id or len(model_id) > 650:
            raise ValueError("model_id must be a nonempty discovered identifier")
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM qualifications WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("stored qualification is malformed")
        return payload
