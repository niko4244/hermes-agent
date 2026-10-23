# Universal model orchestration — upstream port

This is the upstream semantic port of the isolated Hermes model registry. It is
an **opt-in, profile-scoped sidecar**; Hermes' existing model provider resolver
remains authoritative for credentials, transport and actual agent execution.

## Public Python contract

```python
from hermes_cli.universal_models import ModelService

service = ModelService()                       # uses active HERMES_HOME
snapshot = service.status(refresh=False)       # no network discovery
snapshot = service.refresh()                   # only loopback discovery
service.assign("delegation.leaf", "pinned", "openrouter::vendor/model")
selected = service.resolve("delegation.leaf")  # dict model row or None
service.profile("export")                      # list/save/import/export/activate/delete

# Does not execute inference by default (listing only).
service.qualify("<observed-model-id>")

# Explicit, bounded native Ollama inference after a recent refresh:
report = service.qualify(
    "<observed-ollama-id>", run_inference=True,
    timeout_seconds=15, max_tokens=16, require_loaded=True,
)
history = service.qualification("<observed-ollama-id>")
```

`status` returns `models`, `roles`, `profiles`, `active_profile`, `policy`,
`local_only`, `aliases` and `hardware`. Explicitly declared cloud models retain
`availability="configured"`, which is **not** proof of authentication or model
access. Local `reachable` rows are evidence of loopback model listing, not
generation. Disk-only models are not runnable. LiteLLM aliases retain their
configured proxy identity. Unassigned `auto` models never initiate calls to
paid providers. A pinned role does not silently substitute its fallbacks.

Persisted SQLite state is under the profile's `HERMES_HOME` and contains no
provider tokens, raw LiteLLM upstream configuration or inference bodies.
Loopback model observations expire after 120 seconds; historical qualification
reports remain timestamped and cannot upgrade an expired row to reachable.
When rebuilding an explicitly saved pinned/auto role after that expiration,
the resolver may revalidate a **previously observed direct native Ollama**
endpoint with a bounded read-only model-list query. Only a matching currently
served model is eligible; an unavailable pin never falls back to a different
model, and an expired proxy identity is never implicitly probed. Automatic
roles without any saved selection recheck at most three previously observed
native loopback endpoints. Unassigned roles never trigger such revalidation.
These checks do not run inference, load models, query remote providers or
extend historical qualification evidence. Active input/output price caps
continue to reject unknown, nonfinite, negative or boolean cost claims.

## Upstream-specific hardware adaptation

Machine facts come only from `hermes_platform.host.facts` in this port. `hardware`
reports control-host OS family, native architecture, CPU model, physical RAM,
GPU class and available NVIDIA device *names*. It deliberately reports
`gpu_vram_discovery="not_measured"` and `cpu_logical_cores=None`, because the
current upstream platform facts API does not provide free/total VRAM or core
count. The older feature used `nvidia-smi` plus platform-specific memory and
core APIs; that implementation was removed to honor upstream's single hardware
source. No CUDA, downloads, processes or cloud services are invoked.

Qualification uses a fixed public arithmetic prompt over native Ollama
`/api/chat`; it requires a recent matching native `/api/tags` response and,
by default, confirms `/api/ps` already lists the selected loaded model. It
records client-observed TTFT, server-reported evaluation token throughput,
timestamp/settings, exact-match result and SHA-256 provenance. It does **not**
certify general model quality or tool-use capability.

Verification: `scripts/run_tests.sh tests/universal_models -j 4 -q` and
`python -m ruff check hermes_cli/universal_models tests/universal_models`.

## Current-upstream TUI and execution integration

Open `/orchestrate` in the native Hermes TUI to inspect discovered models,
save profile-specific role assignments and ordered fallbacks, set a local-only
restriction or per-million-token price caps, and request a bounded local
qualification. Qualifying a model is **never implicit** in refresh or browsing;
the user must confirm inference, and a model not already present in native
Ollama `/api/ps` is refused without loading it. The regular `/model` picker
continues to work independently.

All six `orchestration.*` JSON-RPC methods are declared in
`tui_gateway/contracts/orchestration.py`, generated into the shared TS/OpenRPC
contract, and carry the active `session_id` when opened from a live conversation.
The gateway binds that session's profile home and secret scope before every
model-registry operation. Sessions without an identifier edit only the current
profile. Every RPC refuses stale, empty, or unknown explicit session IDs and
rejects mixed session/profile requests before profile resolution, so a stale
conversation cannot silently read or overwrite the launch profile.

Main-agent saved assignments are consulted on current-upstream deferred/eager
agent construction, after explicit session picks and resumable runtime
identities. Native Ollama pins use a directly observed loopback endpoint and a
harmless `local` API-key sentinel; they never inherit ambient cloud keys. A
running session queues its selected pin for the next safe turn boundary; idle
sessions switch through the real agent runtime, not a simulated request.
Pending main-role switches remain blocking admission requests until their
runtime transition succeeds. A rejected or removed pin cannot silently fall
back to the old provider on a later turn; a subsequently confirmed idle
replacement (or deliberate automatic selection) clears that stale pending
request. Approved fallback changes also remove revoked future hops during an
already active provider's cooldown.
Configured cloud providers use Hermes' existing credential and model-switch
resolvers. Remote primary or fallback assignments, including proxied localhost
models and saved delegated roles, require explicit `confirm_cloud=true`
consent before persistence; the native TUI asks for `y` and never sends the
confirm flag during browsing. This also covers deferred assignments that could
otherwise execute after the interaction ends. The existing live-switch
large-model confirmation remains an additional check, not a substitute.
Confirmation is permission to select a potentially billable provider, **not**
a price or entitlement verification.
Profile-only assignments do not silently edit Hermes' global default model.
An idle live-agent switch that succeeds before its profile write commits is
rolled back if that write fails; if restoration also fails, the session is
marked unsafe and refuses subsequent inference rather than silently using
the unconfirmed model.

At each TUI inference turn, the active profile's policy is checked **after**
the normal model/fallback synchronization and **before** any LLM request.
Activating another profile with a different main model invalidates an older
session's universal pin: that session refuses subsequent inference until its
model is explicitly switched or a new session is built under the new profile.
When local-only is active, the runtime must match a native Ollama selection or
an explicitly permitted native Ollama fallback. A pinned local agent has zero
cloud fallbacks unless the operator explicitly configures policy-approved
fallbacks. Invalid or unverified prices fail closed under a price cap; turning
the restriction off leaves unconfigured legacy sessions alone. Refused turns
still release their session's running state.
The saved primary is matched against the executable provider identity as well
as the model slug. Previously activated fallbacks must remain in the active
profile's approved fallback IDs and match the executable provider/endpoint,
including saved automatic main roles without any cost or local-only restriction.
Removing a fallback revokes it before the next turn.

For delegated children, the upstream `TaskRoute` / `DelegationProvider` path
retains explicitly selected per-task providers, models, and cross-profile
routes. Otherwise an explicitly saved `delegation.orchestrator` or
`delegation.leaf` model is chosen from the child's **actual depth-derived
capability**, never from an untrusted display role. An unassigned role leaves
the normal delegation provider unchanged. Explicit local-only restrictions
also guard provider-owned child routes and inherited fallback chains.
An explicit task route or an otherwise unassigned inherited route cannot
bypass a saved local-only or per-million-token price cap. Cross-profile child
routes are checked against their **target** profile's policy and credentials,
not the launching profile's. Unknown, nonfinite, negative or otherwise
unverified model/fallback pricing fails closed whenever a cap is active.
Named custom providers are resolved from their current profile's enabled
registry entry and the resolved runtime must match that entry's endpoint;
their inventory namespace is never mistaken for an executable provider key.

### Scope and verification limits

This port covers the native TUI, its hosted Desktop gateway sessions, and the
supported `delegate_task` child-construction path. It does **not** yet promise
uniform enforcement for all CLI, cron, standalone messaging-gateway, hosted
agent or arbitrary custom role entrypoints. It does not verify account
entitlements or quotas, perform unapproved paid-cloud inference, measure all
GPU/VRAM configurations, or implement global monetary usage accounting.
Configured cloud names remain **configured**, not confirmed accessible.

The current-upstream integration is tested by
`tests/tui_gateway/test_universal_model_routing.py`,
`tests/tui_gateway/test_universal_upstream_contract.py`, and
`tests/tools/test_universal_delegate_routes.py`, in addition to the registry,
gateway-contract and native TUI tests. Hermetic loopback-stream tests in
`tests/tui_gateway/test_universal_hermetic_e2e.py` and
`tests/tui_gateway/test_universal_turn_pipeline.py` construct the real
`AIAgent` and execute real gateway turn handling with a local test server.
They prove the route and streaming integration, not a model's generated
intelligence. Real model-generated inference on the new upstream still
requires a safely usable, already-loaded local Ollama model; no generation
should be inferred from a stub response or listing-installed models.
Ordinary Hermes sessions with no saved model profiles do not create a
universal-model SQLite sidecar on their normal turn/build path.
