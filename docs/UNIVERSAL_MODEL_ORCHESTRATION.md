# Universal model discovery and orchestration

Hermes' native TUI offers `/orchestrate` (alias `/models`) to browse model
inventory, refresh local runtimes, pin models to agent roles, configure ordered
fallbacks, and save or activate routing profiles. The original `/model` picker
remains available for ordinary session model switching.

The subsystem uses Hermes' existing runtime provider resolution, AIAgent
execution, and delegated child lifecycle. There is no mandatory LiteLLM proxy.
The backend consists of `hermes_cli.universal_models.service.ModelService`,
`tui_gateway.methods_orchestration`, `tui_gateway.universal_model_runtime`, and
`tools.delegation_model_router`. Profile data lives in
`HERMES_HOME/universal_models.sqlite3` and does not contain credentials.

## Using the TUI

1. Launch the normal Hermes TUI and enter `/orchestrate`.
2. Refresh inventory to enumerate configured loopback Ollama and compatible
   runtime endpoints. An initial offline inventory also shows explicitly
   configured provider models, LiteLLM aliases, and any explicitly approved
   filesystem model roots.
3. Select a role (`main`, `delegation.leaf`, or `delegation.orchestrator`), then
   choose `auto` or pin a discovered model. Add fallbacks and order them in the
   panel. You can create additional named assignments, but only roles actually
   used by the execution integration affect current agent runtime behavior.
4. Save a routing profile or activate one from the profile tab. Changes to
   existing in-flight main-agent turns apply at a safe turn boundary. New
   subagents resolve their assignment when they are constructed.
5. Use **Tab** to move between roles, models, profiles and policy, **r** to
   refresh, **/** to filter models, and **Esc** to go back. The policy tab
   configures local-only routing and optional input/output cost ceilings in
   USD per million tokens. Every policy modification requires **y** confirmation.
6. On a directly observed Ollama model in the models tab, press **v** to view
   the bounded local qualification prompt. Confirm with **y**, or cancel with
   **n/Esc**. The dialog allows 16/32/64 output tokens and 10/20/30 seconds.
   A model must already be loaded in Ollama; qualification never starts a
   previously unloaded model. No cloud provider can be qualified this way.

## Discovery behavior and provenance

Normal status calls perform no network discovery. Explicit refresh probes
loopback endpoints including Ollama on port 11434 and compatible APIs on
ports 1234 and 8000, plus explicitly configured loopback endpoints. Probes are
bounded by HTTP timeouts, and results expire after 120 seconds. A name found
in a model catalog is **configured**, not proof of authenticated access.
An `on_disk` entry is not an executable model. A `reachable` entry means a
model-enumeration endpoint answered; it does not establish that text or tool
inference works, or that the endpoint offers free local inference.

LiteLLM aliases are parsed from approved configuration locations under
`HERMES_HOME`, never by scanning arbitrary directories. LiteLLM-configured
models need a matching Hermes provider configuration and credentials for
execution. The registry does not log or copy credentials. Existing provider
adapters handle authentication and provider-specific request formatting.

The native Ollama `/api/tags` response is the only loopback observation
treated as verified native local execution for strict local-only policy.
Generic OpenAI-compatible localhost endpoints may proxy paid cloud calls,
so they are not automatically allowed under a local-only profile.

## Routing and safety

Roles have stable provider- and endpoint-qualified model identities. Pinned
role selections and ordered fallbacks are stored per profile. Delegated leaf
and orchestrator agents resolve role selections separately, enabling concurrent
agents to use different configured providers. A locally selected child uses a
sentinel credential rather than inheriting the parent's cloud key. Declared
local-only and price-policy limits reject unsupported selections; unknown
pricing is never treated as zero cost. Explicit pinned models cannot silently
degrade to an unrelated parent/provider. Legacy delegation without a new
role policy retains existing behavior.

An explicitly saved `auto` assignment chooses from verified local Ollama
models in deterministic order when no model is pinned. Unconfigured/default
role labels do not change the previous provider behavior. A pinned model is
never silently replaced with its configured fallback before a real request.

New-session `main` pins use the same Hermes agent construction path. For live
native Ollama session switches, the gateway invokes the established
`agent.switch_model` method and installs only the policy-approved explicit
fallback chain; an empty chain cannot inherit a cloud fallback.
During an active turn, the switch queues for the next safe turn boundary.
The user can continue using `/model` for explicitly configured providers.

## Troubleshooting and limitations

If the model list is empty, check that the runtime is serving `/api/tags` or
`/v1/models`, then use refresh. If a provider is marked configured, verify
its existing Hermes provider authentication before assigning it. If a model
is absent after a period of inactivity, refresh its local observation.

The optional `ModelService.qualify(model_id, run_inference=True)` performs a
single bounded fixed arithmetic request over a native local Ollama stream,
only after the caller explicitly opts in. It checks `/api/tags` and `/api/ps`
again, requires the model to be already loaded, enforces a total timeout and
output limits, and persists timestamped TTFT, duration and Ollama-reported
tokens/sec with settings/provenance. It never sends a user's conversation or
provider credentials. The default `qualify(model_id)` remains a listing-only
check and does not claim inference was tested. A passing arithmetic probe is
not a broad reasoning, coding, tool use, context handling, or hardware fit
benchmark. Cross-provider fallback uses Hermes' existing completion retry
machinery, which is not a guarantee against provider billing or tool-side
effects.

The initial implementation still does not complete every item in the broader
engineering mission: full account-level live catalog enumeration for all
cloud providers, total spending/token budgets, independently verified
role-specific coding/tool-call benchmarks, GPU fit recommendations, and a
real paid-provider multi-agent test are outstanding. Generated text through
the TUI gateway-to-agent path *was* verified on a directly configured local
Ollama endpoint; a rendered interactive TUI keyboard test remains simulated.
Custom names outside Hermes' implemented `main` and delegation roles are
persisted but do not yet route a corresponding live task automatically.

### Real local qualification evidence (2026-10-23)

On the local Windows host, the already available `llama3.2:3b` model was
selected via `orchestration.assign`, created by the actual TUI gateway
`_make_agent` path, and completed `AIAgent.run_conversation` with the exact
response `TUI_ROUTE_OK` (one local API call). Ollama model metadata reported a
131,072-token context window, satisfying Hermes' 64K minimum. The separate opt-in native Ollama fixed
arithmetic probe returned `status=passed`, `inference_tested=true`, a 31 ms
observed first streamed-text interval and 185.151 tokens/sec calculated from
Ollama-reported eval count and duration. This is **one warm-model observation**
and does not establish performance for other prompts or hardware. A previous
attempt while the model was unloaded correctly returned
`model_not_already_loaded` without starting inference.

## Validation

Run the repository's canonical test wrapper for
`tests/universal_models/`, `tests/tools/test_delegation_model_router.py`,
`tests/tui_gateway/test_universal_model_routing.py`, and
`tests/tui_gateway/test_make_agent_provider.py`. Build and typecheck `ui-tui`,
and run its focused Vitest tests. A model enumeration response or a mocked
provider request must not be reported as a successful live inference trial.

The October 23 broader Windows run included 55 Python test files: 390 tests
passed and two failed. Both failures also reproduce against the unchanged
original Hermes checkout: the test subprocess assumes POSIX `signal.SIGPIPE`
exists on Windows, and a compute-host file-log concurrency assertion can
lose expected writes under this environment. The original-checkout run also
hit a compute-host drain timing assertion, supporting the classification of
these as existing platform/timing failures rather than orchestration changes.
The new model routing and qualification suites passed.
