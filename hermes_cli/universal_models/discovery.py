"""Read-only model discovery. No credentials, downloads, or model executions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


_ID_LIMIT = 300
_SECRET_LIKE = re.compile(r"(?:^sk[-_]|^gh[pousr]_|^Bearer\s|(?:api[_-]?key|token|password|secret)\s*[=:])", re.IGNORECASE)


def _name(value: Any) -> str:
    if not isinstance(value, str) or not 0 < len(value.strip()) <= _ID_LIMIT:
        return ""
    value = value.strip()
    if (any(ord(c) < 32 or ord(c) == 127 for c in value)
            or "://" in value or "?" in value or "#" in value or _SECRET_LIKE.search(value)):
        return ""
    return value


def safe_endpoint(url: Any) -> str:
    """Only show scheme/host/port; credentials, query strings, and paths are private."""
    if not isinstance(url, str) or "{" in url or "}" in url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return ""


def _row(provider: str, model: str, source: str, availability: str, *, endpoint: str = "") -> dict:
    # Same model slug can be served by two independent local processes.
    # Hash the endpoint into the identity so a pinned role never drifts to
    # another localhost port after discovery merges observations.
    namespace = provider
    if endpoint and source in {"ollama-live", "openai-compatible-live"}:
        namespace += "@" + hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    result = {
        "id": f"{namespace}::{model}",
        "provider": provider,
        "model": model,
        "display_name": model,
        "source": source,
        "availability": availability,
        "capabilities": [],  # unknown until actual qualification: never infer from a model name
    }
    if endpoint:
        result["endpoint"] = endpoint
    return result


def _declared_models(value: Any) -> list[str]:
    """Hermes model_switch._declared_model_ids shapes, minus unsafe text."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, dict):
        values = list(value)
    elif isinstance(value, (list, tuple)):
        values = [item if isinstance(item, str) else item.get("id") or item.get("name")
                  for item in value[:1000] if isinstance(item, (str, dict))]
    else:
        return []
    unique: dict[str, str] = {}
    for value in values[:1000]:
        valid = _name(value)
        if valid:
            unique.setdefault(valid.lower(), valid)
    return list(unique.values())


def _providers(config: Mapping[str, Any]) -> tuple[set[str], dict[str, str], list[dict]]:
    """Hermes registry, keyed provider override identities, legacy endpoints.

    The built-in catalog identifies provider *names*, not models the user may
    access. We only enumerate model IDs explicitly written in the user's YAML.
    No auth pool, provider catalog API, models.dev fetch, or network probe.
    """
    from hermes_cli.models import _PROVIDER_MODELS, normalize_provider
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.providers import HERMES_OVERLAYS, custom_provider_slug
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import get_compatible_custom_providers, is_provider_enabled

    known = set(_PROVIDER_MODELS) | set(HERMES_OVERLAYS) | set(PROVIDER_REGISTRY) | set(PROVIDER_TO_MODELS_DEV)
    # Together is a valid explicitly configured OpenAI-compatible backend but
    # lacks a curated Hermes registry entry. This recognizes the provider name
    # only; it does not discover or grant access to a Together model catalog.
    known.add("together")
    known.add("openrouter")
    keyed: dict[str, str] = {}
    raw_providers = config.get("providers")
    if isinstance(raw_providers, dict):
        for key, entry in raw_providers.items():
            if not isinstance(key, str) or not isinstance(entry, dict) or not is_provider_enabled(entry):
                continue
            raw_name = _name(key)
            if not raw_name:
                continue
            # A keyed endpoint always wins over a built-in alias, matching
            # resolve_provider_full(), including providers.openai and aliases.
            # Built-in options-only entries do not become invented custom APIs.
            has_endpoint = any(isinstance(entry.get(k), str) and entry[k].strip()
                               for k in ("api", "url", "base_url"))
            if has_endpoint:
                keyed[raw_name.lower()] = custom_provider_slug(raw_name, raw_name)
            else:
                canonical = normalize_provider(raw_name)
                if canonical in known:
                    keyed[raw_name.lower()] = canonical
    # Hermes' compatibility normalizer filters disabled *keyed* providers,
    # but the legacy list path does not filter them. Do it here without
    # mutating the caller's config or copying any credential data into rows.
    view = dict(config)
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        view["custom_providers"] = [entry for entry in legacy
                                    if isinstance(entry, dict) and is_provider_enabled(entry)]
    entries = get_compatible_custom_providers(view)
    return known, keyed, entries


def _canonical_provider(raw: Any, known: set[str], keyed: dict[str, str]) -> str:
    from hermes_cli.models import normalize_provider

    original = _name(raw)
    if not original or original.lower() == "auto":
        return ""
    direct = keyed.get(original.lower())
    if direct:
        return direct
    if original.startswith("custom:"):
        # Explicit named custom: identities only if their named endpoint was
        # validated in the current config, not arbitrary invented providers.
        return original.lower() if original.lower() in keyed.values() else ""
    canonical = normalize_provider(original)
    return canonical if canonical in known else ""


def _explicit_pairs(config: Mapping[str, Any], known: set[str], keyed: dict[str, str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []

    def add(provider: Any, model: Any, source: str) -> None:
        identity, model_id = _canonical_provider(provider, known, keyed), _name(model)
        if identity and model_id:
            pairs.append((identity, model_id, source))

    def fallback(value: Any) -> None:
        if isinstance(value, dict):
            add(value.get("provider"), value.get("model"), "hermes-fallback")
        elif isinstance(value, (list, tuple)):
            for item in value[:100]:
                if isinstance(item, dict):
                    add(item.get("provider"), item.get("model"), "hermes-fallback")

    active = config.get("model")
    if isinstance(active, dict):
        add(active.get("provider"), active.get("default") or active.get("name") or active.get("model"), "hermes-config")
    if "fallback_providers" in config and isinstance(config["fallback_providers"], list) and config["fallback_providers"]:
        fallback(config["fallback_providers"])
    else:
        fallback(config.get("fallback_model"))
    auxiliary = config.get("auxiliary")
    if isinstance(auxiliary, dict):
        for settings in auxiliary.values():
            if not isinstance(settings, dict):
                continue
            add(settings.get("provider"), settings.get("model"), "hermes-auxiliary")
            fallback(settings.get("fallback_chain"))
    for alias in config_aliases(config):
        add(alias["provider"], alias["model"], "hermes-alias")
    return pairs


def configured_models(config: Mapping[str, Any]) -> list[dict]:
    """Only explicit Hermes model selections, never credential-derived catalogs.

    Configured does NOT mean authenticated, enabled on the provider's account,
    reachable, affordable or capable of tool calling.
    """
    known, keyed, entries = _providers(config)
    pairs = _explicit_pairs(config, known, keyed)
    raw_providers = config.get("providers")
    if isinstance(raw_providers, dict):
        for name, entry in raw_providers.items():
            if not isinstance(entry, dict) or not isinstance(name, str) or name.lower() not in keyed:
                continue
            identity = keyed[name.lower()]
            for model in _declared_models(entry.get("models")):
                pairs.append((identity, model, "hermes-provider"))
            for model in (entry.get("default_model"), entry.get("model")):
                if _name(model):
                    pairs.append((identity, model, "hermes-provider"))

    # Both modern keyed endpoints and legacy custom_providers are normalized
    # by Hermes, including disable flags, ID/name lists, and aliases. The same
    # custom name in providers: and custom_providers: deduplicates by stable ID.
    from hermes_cli.providers import custom_provider_slug

    for entry in entries:
        name = _name(entry.get("provider_key") or entry.get("name"))
        if not name:
            continue
        provider = custom_provider_slug(str(entry.get("name") or name), str(entry.get("provider_key") or ""))
        for model in [*_declared_models(entry.get("models")), entry.get("model")]:
            if _name(model):
                pairs.append((provider, model, "hermes-provider"))

    # Source priority follows active selection, then declared provider entries,
    # then fallback/aux/aliases. Never collapse across distinct provider IDs.
    priority = {"hermes-config": 0, "hermes-provider": 1, "hermes-alias": 2,
                "hermes-auxiliary": 3, "hermes-fallback": 4}
    rows: dict[str, dict] = {}
    for provider, model, source in pairs:
        row = _row(provider, model, source, "configured")
        current = rows.get(row["id"])
        if current is None or priority[source] < priority[current["source"]]:
            rows[row["id"]] = row
    return list(rows.values())


def config_aliases(config: Mapping[str, Any]) -> list[dict]:
    """Describe existing Hermes aliases without changing runtime alias resolution."""
    aliases: dict[str, dict] = {}
    explicit = config.get("model_aliases")
    if isinstance(explicit, dict):
        for alias, item in explicit.items():
            if not isinstance(item, dict):
                continue
            alias_name = _name(alias)
            model, provider = _name(item.get("model")), _name(item.get("provider"))
            if alias_name and model and provider:
                aliases[alias_name.lower()] = {"alias": alias_name, "provider": provider, "model": model}
    main = config.get("model")
    if isinstance(main, dict) and isinstance(main.get("aliases"), dict):
        for alias, value in main["aliases"].items():
            alias_name, target = _name(alias), _name(value)
            if not alias_name or not target or alias_name.lower() in aliases:
                continue
            provider = _name(main.get("provider"))
            if "/" in target:
                provider, target = target.split("/", 1)
            if provider and target:
                aliases[alias_name.lower()] = {"alias": alias_name, "provider": provider, "model": target}
    return list(aliases.values())


def litellm_models(paths: tuple[Path, ...], config: Mapping[str, Any]) -> list[dict]:
    """Read LiteLLM model aliases; never inspect or serialize litellm_params secrets."""
    try:
        import yaml
    except ImportError:
        return []
    entries: list[dict] = []
    # A bare provider name, API key or disabled/malformed endpoint does not
    # establish a usable LiteLLM proxy. Reuse Hermes' endpoint validation.
    from hermes_cli.config import get_compatible_custom_providers, is_provider_enabled

    view = dict(config)
    legacy = config.get("custom_providers")
    if isinstance(legacy, list):
        view["custom_providers"] = [entry for entry in legacy
                                    if isinstance(entry, dict) and is_provider_enabled(entry)]
    proxy_configured = any(
        str(p.get("provider_key") or p.get("name") or "").strip().lower() == "litellm"
        for p in get_compatible_custom_providers(view)
    )
    provider = "custom:litellm"
    for path in paths:
        try:
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("model_list"), list):
            continue
        for item in raw["model_list"][:1000]:
            if not isinstance(item, dict):
                continue
            alias = _name(item.get("model_name"))
            if alias:
                entries.append(_row(provider, alias, "litellm-config", "configured" if proxy_configured else "unconfigured"))
    return entries


def on_disk_models(paths: tuple[Path, ...]) -> list[dict]:
    """Bounded directory walk of local GGUF and Hugging Face model identities."""
    results: list[dict] = []
    directories_seen = 0
    for root in paths:
        if not root.is_dir() or root.is_symlink():
            continue
        # Only three levels: walk large storage without traversing unrelated trees.
        for directory, subdirs, files in os.walk(root, followlinks=False):
            directories_seen += 1
            if directories_seen > 2000:
                return results
            here = Path(directory)
            depth = len(here.relative_to(root).parts)
            subdirs[:] = sorted(x for x in subdirs if not (here / x).is_symlink())[:100] if depth < 3 else []
            if "manifests" in root.parts and depth >= 3:
                # A manifest file is a local Ollama *cache* identity. It does
                # not establish that its corresponding model is being served.
                for file in sorted(files)[:500]:
                    manifest_path = here / file
                    if manifest_path.is_file() and not manifest_path.is_symlink():
                        model_name = f"{here.name}:{file}"
                        results.append(_row("custom", model_name, "ollama-manifest", "on_disk"))
            for file in sorted(files)[:500]:
                path = here / file
                if file.lower().endswith(".gguf") and path.is_file() and not path.is_symlink():
                    results.append(_row("custom", str(path), "local-gguf", "on_disk"))
                elif file == "config.json" and any(here.glob("*.safetensors")):
                    results.append(_row("custom", str(here), "local-hf", "on_disk"))
                if len(results) >= 1000:
                    return results
    return results


def _get_json(url: str, timeout: float) -> dict | None:
    try:
        # Do not allow a local-looking server to redirect read-only discovery
        # to an arbitrary external service.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            return None
        result = json.loads(payload)
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def local_server_models(endpoints: tuple[str, ...], *, timeout: float = 0.6) -> list[dict]:
    """Opt-in live, unauthenticated probes of loopback services only."""
    entries: list[dict] = []
    for endpoint in endpoints:
        clean = safe_endpoint(endpoint)
        if not clean:
            continue
        parsed = urllib.parse.urlsplit(clean)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            continue
        tags = _get_json(f"{clean}/api/tags", timeout)
        if tags is not None and isinstance(tags.get("models"), list):
            # Actual *loaded* context is measured by Ollama /api/ps. Config
            # model max context is not the runtime's active context budget.
            process_list = _get_json(f"{clean}/api/ps", timeout)
            loaded: dict[str, dict] = {}
            if isinstance(process_list, dict) and isinstance(process_list.get("models"), list):
                for process in process_list["models"][:1000]:
                    if isinstance(process, dict):
                        identity = _name(process.get("name") or process.get("model"))
                        if identity:
                            loaded[identity] = process
            for item in tags["models"][:1000]:
                model = _name(item.get("name") if isinstance(item, dict) else None)
                if model:
                    row = _row("custom", model, "ollama-live", "reachable", endpoint=f"{clean}/v1")
                    running = loaded.get(model)
                    if running is not None:
                        row["loaded"] = True
                        context = running.get("context_length")
                        if isinstance(context, int) and not isinstance(context, bool) and 0 < context <= 10_000_000:
                            row["context_window"] = context
                            row["context_source"] = "ollama_api_ps_loaded_context"
                    entries.append(row)
            continue
        data = _get_json(f"{clean}/v1/models", timeout)
        if data is None:
            continue
        if isinstance(data.get("data"), list):
            for item in data["data"][:1000]:
                model = _name(item.get("id") if isinstance(item, dict) else None)
                if model:
                    entries.append(_row("custom", model, "openai-compatible-live", "reachable", endpoint=f"{clean}/v1"))
    return entries


def hardware_summary() -> dict:
    """Best-effort local capacity, not a model fit/benchmark claim."""
    logical_cpus = os.cpu_count()
    result: dict[str, Any] = {
        "gpu": [], "gpu_discovery": "unavailable",
        "cpu_logical_cores": logical_cpus if isinstance(logical_cpus, int) and logical_cpus > 0 else None,
        "ram_total_mib": None,
        "ram_source": "unavailable",
    }
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong),
                ]

            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                result["ram_total_mib"] = int(memory.total_physical // (1024 * 1024))
                result["ram_available_mib"] = int(memory.available_physical // (1024 * 1024))
                result["ram_source"] = "windows_global_memory_status"
        elif hasattr(os, "sysconf"):
            total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
            if total > 0:
                result["ram_total_mib"] = total // (1024 * 1024)
                result["ram_source"] = "sysconf_physical_memory"
    except (AttributeError, OSError, ValueError):
        pass
    if not shutil.which("nvidia-smi"):
        return result
    try:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        for line in run.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                result["gpu"].append({"name": parts[0], "memory_total_mib": int(parts[1]), "memory_free_mib": int(parts[2])})
        result["gpu_discovery"] = "measured"
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def merge_models(rows: list[dict]) -> list[dict]:
    """Deduplicate observations without turning disk/config presence into proof of serving."""
    priority = {"reachable": 4, "configured": 3, "on_disk": 2, "unconfigured": 1}
    result: dict[str, dict] = {}
    for row in rows:
        existing = result.get(row["id"])
        if existing is None or priority.get(row["availability"], 0) > priority.get(existing["availability"], 0):
            result[row["id"]] = row
    return sorted(result.values(), key=lambda row: (row["provider"].lower(), row["model"].lower()))
