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
