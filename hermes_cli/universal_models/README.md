# Universal model registry

`ModelService` provides a profile-scoped, credential-free model inventory and
explicit agent-role assignment API for native clients. The existing Hermes
provider and credential resolver remains responsible for actual inference.

```python
from hermes_cli.universal_models import ModelService

models = ModelService()
snapshot = models.status()                 # no live HTTP probes
snapshot = models.refresh()                # loopback discovery, explicit action
models.assign("delegation.leaf", "pinned", "openrouter::example/model")
selected = models.resolve("delegation.leaf")  # model row or None
```

Each model row has `id`, `provider`, `model`, `display_name`, `source`,
`availability`, and `capabilities`. `endpoint` is supplied only for locally
observed OpenAI-compatible servers. Configured providers have no endpoint
override: Hermes must resolve their authentic configured transport and keys.
Capabilities and context are left unknown until actually established.

Availability means:

* `configured` — present in Hermes/LiteLLM configuration; authentication and
  serving have **not** been verified.
* `reachable` — appeared in a live loopback `/api/tags` or `/v1/models` query;
  this verifies listing only, not successful generation or tool calling.
* `on_disk` — recognized local GGUF/HF asset or Ollama manifest, which alone
  cannot accept inference requests.
* `unconfigured` — LiteLLM alias found without a registered Hermes proxy.

`status(refresh=False)` reads configuration, bounded local asset roots and the
last 120 seconds of loopback observations. `refresh()` probes loopback only.
Observed servers have endpoint-qualified IDs so equal model names on different
ports remain separate. `qualify(model_id)` remains a *listing-only* check and
returns `inference_tested=False`. Ollama listing also reads `/api/ps` when
available: `loaded`, `context_window`, and `context_source` describe the
currently loaded context, rather than a model's theoretical maximum. CPU
logical cores, physical RAM and GPU VRAM are measured on explicit refresh.

Actual generation requires separate, explicit opt-in:

```python
result = models.qualify(
    model_id, run_inference=True, timeout_seconds=15,
    max_tokens=16, require_loaded=True,
)
previous = models.qualification(model_id)
```

Only recently discovered native Ollama models on loopback HTTP qualify.
The service rechecks `/api/tags` and, by default, `/api/ps` to confirm the
model is already loaded. This avoids loading a new model and potentially
displacing another. `require_loaded=False` explicitly permits a load.
Generation uses one fixed public 2+2 prompt through native streaming
`/api/chat` at temperature zero, limited to 1–30 seconds and 1–64 output
tokens. It does not use cloud providers, remote redirects, API keys, personal
prompts or an automatic paid fallback.

The timestamped report records settings, streaming TTFT, total generation
elapsed time, reported token counts and tokens/second, exact-match quality,
and cryptographic hashes of prompt and response. TTFT is the first nonempty
generated text stream chunk; throughput is calculated from Ollama-reported
`eval_count` and `eval_duration`, labeled as such. The test measures a single
arithmetic response, not general quality or tool-use competence. Neither
prompt nor output bodies are stored. `status()` includes `last_qualification`
for observed model rows, while `qualification(model_id)` retrieves historical
evidence even after live observations have expired.

Persistent data lives in `HERMES_HOME/universal_models.sqlite3`, isolated per
Hermes profile. Only role assignments, policy, profile names, and short-lived
local metadata are stored. Credential values and upstream LiteLLM parameters
are never copied into this database or returned in status.

`assign(role, mode, model_id=None, fallback_ids=None)` accepts `pinned` or
`auto`; IDs must be discovered first. `profile(action, name=None, data=None)`
supports `list`, `save`, `import`, `export`, `activate` and `delete`. `save` with
no data copies the active role/policy set into a named profile, while `import`
validates a complete `{roles:[...], policy:{...}}` payload. The policy supports
`local_only`, `max_input_cost_per_million`, and
`max_output_cost_per_million`; unknown capability, context or cost data fails
the corresponding filter rather than being guessed. `resolve` keeps pinned
roles on their chosen primary without substituting fallback IDs. An empty auto
role selects only observed native local Ollama models in deterministic order,
and never automatically chooses a paid cloud model. An unassigned `main` role
retains Hermes's current explicitly configured model. A profile's `local_only`
policy recognizes only a live native Ollama
listing as evidence of local inference: even a localhost LiteLLM proxy may
forward to billable remote infrastructure.
