"""Typed, credential-free model management wire for the native TUI and Desktop.

Only locally approved status probes can return runtime availability. The
schema deliberately distinguishes provider-configured names from live tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import JsonValue, Result
from .common import OpenModel, ProfileParams
from .registry import method


class OrchestrationModel(OpenModel):
    id: str
    provider: str
    model: str
    display_name: str = ""
    source: str = ""
    availability: str
    capabilities: list[str] | dict[str, JsonValue] = Field(default_factory=list)
    context_window: int | None = None
    endpoint: str | None = None


class OrchestrationRole(Result):
    role: str
    mode: str = "auto"
    model_id: str | None = None
    fallback_ids: list[str] = Field(default_factory=list)


class OrchestrationProfileRow(OpenModel):
    name: str
    active: bool | None = None


class OrchestrationPolicy(Result):
    local_only: bool | None = None
    max_input_cost_per_million: float | None = None
    max_output_cost_per_million: float | None = None


class OrchestrationScopeParams(ProfileParams):
    session_id: str | None = None


class OrchestrationStatusParams(OrchestrationScopeParams):
    refresh: bool = False


class OrchestrationStatusResult(Result):
    models: list[OrchestrationModel]
    roles: list[OrchestrationRole]
    profiles: list[OrchestrationProfileRow]
    active_profile: str | None = None
    policy: OrchestrationPolicy = Field(default_factory=OrchestrationPolicy)
    local_only: bool = False
    aliases: list[JsonValue] = Field(default_factory=list)
    hardware: JsonValue | None = None


method("orchestration.status", params=OrchestrationStatusParams, result=OrchestrationStatusResult,
       doc="Read normalized model inventory and the active per-agent routing profile.")
method("orchestration.refresh", params=OrchestrationScopeParams, result=OrchestrationStatusResult,
       doc="Explicitly refresh approved local endpoint model inventory.")


class OrchestrationAssignParams(OrchestrationScopeParams):
    role: str
    mode: Literal["auto", "pinned"]
    model_id: str | None = None
    fallback_ids: list[str] | None = None


class OrchestrationAssignResult(OpenModel):
    role: str | None = None
    mode: str | None = None
    model_id: str | None = None
    fallback_ids: list[str] | None = None
    deferred: bool | None = None
    warning: str | None = None
    confirm_required: bool | None = None


method("orchestration.assign", params=OrchestrationAssignParams, result=OrchestrationAssignResult,
       doc="Persist a role model assignment, safely switching the active main session when provided.")


class OrchestrationProfileParams(OrchestrationScopeParams):
    action: Literal["activate", "save", "import", "export", "list", "delete"]
    name: str | None = None
    data: JsonValue | None = None


class OrchestrationProfileResult(OpenModel):
    name: str | None = None
    roles: list[OrchestrationRole] | None = None
    policy: OrchestrationPolicy | None = None
    profiles: list[OrchestrationProfileRow] | None = None
    active_profile: str | None = None
    deleted: str | None = None


method("orchestration.profile", params=OrchestrationProfileParams, result=OrchestrationProfileResult,
       doc="Save, select, import, export or list non-secret agent routing profiles.")


class OrchestrationPolicyParams(OrchestrationScopeParams):
    action: Literal["get", "set"] = "get"
    policy: OrchestrationPolicy | None = None


class OrchestrationPolicyResult(Result):
    active_profile: str | None = None
    policy: OrchestrationPolicy


method("orchestration.policy", params=OrchestrationPolicyParams, result=OrchestrationPolicyResult,
       doc="Read or atomically replace the active profile's local-only and price controls.")


class OrchestrationQualifyParams(OrchestrationScopeParams):
    model_id: str
    max_tokens: int = 16
    timeout_seconds: float = 15


class OrchestrationQualifyResult(OpenModel):
    model_id: str
    available: bool = False
    status: str = ""
    inference_tested: bool = False
    reason: str | None = None
    level: str | None = None
    metrics: JsonValue | None = None
    provenance: JsonValue | None = None
    quality: JsonValue | None = None
    settings: JsonValue | None = None
    timestamp: JsonValue | None = None


method("orchestration.qualify", params=OrchestrationQualifyParams, result=OrchestrationQualifyResult,
       doc="Explicit, bounded inference qualification on an already loaded native Ollama model.")
