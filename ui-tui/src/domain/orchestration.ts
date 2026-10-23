/** Public, non-credential inventory returned by the orchestration gateway. */
export interface OrchestrationModel {
  id: string
  provider: string
  model: string
  display_name?: string
  source?: string
  availability: string
  capabilities?: string[] | Record<string, unknown>
  context_window?: number | null
  endpoint?: string
  max_input_cost_per_million?: number | null
  max_output_cost_per_million?: number | null
}

/** Active profile guardrails; omitted caps mean no price limit, not zero cost. */
export interface OrchestrationPolicy {
  local_only?: boolean
  max_input_cost_per_million?: number
  max_output_cost_per_million?: number
}

export type PolicyCap = 'max_input_cost_per_million' | 'max_output_cost_per_million'

export interface OrchestrationPolicyResult {
  policy: OrchestrationPolicy
  active_profile?: string
}

export interface OrchestrationQualificationResult {
  model_id: string
  status?: string
  outcome?: string
  level?: string
  available?: boolean
  passed?: boolean
  inference_tested?: boolean
  reason?: string
  timestamp?: string
  quality?: { passed?: boolean }
  metrics?: { latency_ms?: number; elapsed_seconds?: number }
  ttft_ms?: number | null
  elapsed_ms?: number | null
  tokens_per_second?: number | null
  completion_tokens_reported?: number | null
  latency_ms?: number
  measured_at?: string
  warning?: string
  provenance?: Record<string, unknown>
}

/** Never derive a passing inference test from mere registry reachability. */
export function isVerifiedQualification(result: OrchestrationQualificationResult): boolean {
  return (
    result.inference_tested === true &&
    result.status === 'passed' &&
    result.quality?.passed !== false &&
    result.passed !== false
  )
}

/** Display only vetted provenance fields, not opaque response bodies or credentialed URLs. */
export function qualificationProvenance(result: OrchestrationQualificationResult): string {
  const provenance = result.provenance
  if (!provenance || typeof provenance !== 'object' || Array.isArray(provenance)) {
    return 'provenance not provided by gateway'
  }
  const visibleKeys = [
    'source',
    'probe',
    'method',
    'runner',
    'protocol',
    'backend',
    'model',
    'measured_at',
    'timestamp',
    'transport',
    'origin'
  ]
  const safe = visibleKeys.flatMap(key => {
    const value = provenance[key]
    return (typeof value === 'string' && /^[\w .:/+-]{1,80}$/.test(value)) || typeof value === 'boolean'
      ? [`${key}: ${value}`]
      : []
  })
  return safe.join(' · ') || 'provenance not supplied in safe display fields'
}

export interface OrchestrationRole {
  role: string
  mode: 'auto' | 'pinned'
  model_id?: string | null
  fallback_ids?: string[]
}

export interface OrchestrationProfile {
  name: string
  [key: string]: unknown
}

export interface OrchestrationStatus {
  models: OrchestrationModel[]
  roles: OrchestrationRole[]
  profiles: OrchestrationProfile[]
  active_profile?: string | null
  hardware?: unknown
  policy?: OrchestrationPolicy
  local_only?: boolean
}

/** Format amounts without implying that unknown catalog prices are zero. */
export function priceLimitLabel(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? `$${value.toLocaleString('en-US', { maximumFractionDigits: 6 })}/1M tokens`
    : 'no cap (price not guaranteed)'
}

export function policySummary(policy: OrchestrationPolicy | undefined): string {
  if (!policy) {
    return 'policy unavailable · refresh or update gateway'
  }

  return `local only: ${policy.local_only === true ? 'ON' : 'OFF'} · input cap: ${priceLimitLabel(policy.max_input_cost_per_million)} · output cap: ${priceLimitLabel(policy.max_output_cost_per_million)}`
}

/** A populated price field is metadata, not proof of an independently verified rate. */
export function modelCostLabel(model: OrchestrationModel): string {
  if (isDirectLocalModel(model)) {
    return 'direct local Ollama · no hosted token charge (uses local compute)'
  }

  const input = model.max_input_cost_per_million
  const output = model.max_output_cost_per_million
  if (typeof input === 'number' && Number.isFinite(input) && typeof output === 'number' && Number.isFinite(output)) {
    return `catalog price, not verified: input ${priceLimitLabel(input)}, output ${priceLimitLabel(output)}`
  }

  return 'price unverified · provider/proxy may charge for inference'
}

export function availabilityLabel(model: OrchestrationModel): string {
  switch (model.availability) {
    case 'reachable':
      return 'endpoint reachable (inference not verified)'
    case 'configured':
      return 'configured (live inference unverified)'
    case 'on_disk':
      return 'on disk (not confirmed serving)'
    case 'unconfigured':
      return 'unconfigured (not runnable)'
    case 'offline':
    case 'unavailable':
    case 'disconnected':
      return 'unavailable'
    default:
      return 'status unknown (not verified)'
  }
}

/** Only observed, direct, credential-free Ollama loopback is safe to probe. */
export function isDirectLocalModel(model: OrchestrationModel): boolean {
  if (model.source !== 'ollama-live' || model.provider !== 'custom' || model.availability !== 'reachable') {
    return false
  }

  try {
    const url = new URL(model.endpoint ?? '')
    return (
      url.protocol === 'http:' &&
      ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname.toLowerCase()) &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash &&
      url.pathname === '/v1'
    )
  } catch {
    return false
  }
}

export function qualificationRequest(model: OrchestrationModel, maxTokens: number, timeoutSeconds: number) {
  if (!isDirectLocalModel(model)) {
    throw new Error('Only observed direct local Ollama models can be qualified; refresh inventory first.')
  }
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 64) {
    throw new Error('Qualification token limit must be 1–64.')
  }
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 1 || timeoutSeconds > 30) {
    throw new Error('Qualification timeout must be 1–30 seconds.')
  }

  return { model_id: model.id, max_tokens: maxTokens, timeout_seconds: timeoutSeconds }
}

export function nextPolicy(
  policy: OrchestrationPolicy,
  field: 'local_only' | PolicyCap,
  value: boolean | number | null
) {
  const next: OrchestrationPolicy = { ...policy }
  if (field === 'local_only') {
    if (typeof value !== 'boolean') {
      throw new Error('Local-only setting must be ON or OFF.')
    }
    next.local_only = value
  } else if (value === null) {
    delete next[field]
  } else if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
    next[field] = value
  } else {
    throw new Error('Price cap must be a nonnegative finite USD amount per million tokens.')
  }

  return next
}

export function parsePriceCap(input: string): number | null {
  const trimmed = input.trim()
  if (!/^(?:\d+(?:\.\d{1,6})?|\.\d{1,6})$/.test(trimmed)) {
    return null
  }
  const value = Number(trimmed)
  return Number.isFinite(value) && value >= 0 && value <= 1_000_000 ? value : null
}

export function qualificationSummary(result: OrchestrationQualificationResult): string {
  const level = result.level || 'level unspecified'
  const outcome =
    result.inference_tested === true
      ? isVerifiedQualification(result)
        ? 'passed'
        : result.status || result.outcome || 'not confirmed passing'
      : `metadata-only (${result.status || result.outcome || 'not qualified'})`
  return `${outcome} · ${result.inference_tested === true ? 'inference executed' : 'inference NOT verified'} · ${level}`
}

export function qualificationReasonLabel(result: OrchestrationQualificationResult): string | null {
  switch (result.reason) {
    case 'model_not_already_loaded':
      return 'Model is not already loaded in Ollama. Load it manually, refresh inventory (r), then retry.'
    case 'model_not_listed_now':
      return 'Model is no longer listed by Ollama. Refresh inventory (r) before retrying.'
    case 'stream_failed_or_incomplete':
      return 'Ollama did not finish within the bounded probe. Check server logs or adjust the timeout.'
    case 'exact_match_failed':
      return 'Inference completed but did not pass the fixed arithmetic check; do not treat the model as qualified.'
    case 'invalid_model_identity':
      return 'Model identity was rejected. Refresh inventory.'
    case 'not_verified_native_ollama_loopback':
      return 'Endpoint cannot be verified as direct native Ollama loopback.'
    case 'exact_match':
      return 'Fixed arithmetic check passed; this is not a general quality benchmark.'
    default:
      return null
  }
}

export interface OrchestrationAssignResult {
  confirm_required?: boolean
  role?: string
  mode?: string
  model_id?: string | null
  deferred?: boolean
  warning?: string
}

/** The catalog is browsable even when a backend reports models as unavailable. */
export function filterOrchestrationModels(models: readonly OrchestrationModel[], query: string): OrchestrationModel[] {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean)

  return models.filter(m => {
    const haystack =
      `${m.display_name ?? ''} ${m.provider} ${m.model} ${m.source ?? ''} ${m.availability}`.toLowerCase()

    return terms.every(term => haystack.includes(term))
  })
}

export function modelLabel(model: OrchestrationModel): string {
  const name = model.display_name?.trim() || model.model || model.id
  const provider = model.provider.trim()

  return provider && !name.toLowerCase().includes(provider.toLowerCase()) ? `${provider} / ${name}` : name
}

/** Keep the same role highlighted when a refreshed inventory reorders assignments. */
export function roleIndexAfterRefresh(
  roles: readonly OrchestrationRole[],
  selectedRole: string | undefined,
  previousIndex: number
): number {
  const match = roles.findIndex(role => role.role === selectedRole)

  return match >= 0 ? match : Math.max(0, Math.min(previousIndex, roles.length - 1))
}

/** Preserve fallback priority; a primary model cannot also be a fallback. */
export function toggleOrchestrationFallback(
  existing: readonly string[],
  candidate: string,
  primary?: string | null
): string[] {
  const normalized = [...new Set(existing.filter(id => Boolean(id) && id !== primary))]

  if (!candidate || candidate === primary) {
    return normalized
  }

  return normalized.includes(candidate) ? normalized.filter(id => id !== candidate) : [...normalized, candidate]
}

/** Shift a selected fallback without changing the set of allowed candidates. */
export function moveOrchestrationFallback(existing: readonly string[], candidate: string, direction: -1 | 1): string[] {
  const ids = [...new Set(existing)]
  const from = ids.indexOf(candidate)
  const to = from + direction

  if (from < 0 || to < 0 || to >= ids.length) {
    return ids
  }

  ;[ids[from], ids[to]] = [ids[to]!, ids[from]!]

  return ids
}

export function buildOrchestrationAssignment(
  role: OrchestrationRole,
  mode: 'auto' | 'pinned',
  modelId?: string | null,
  fallbackIds?: readonly string[],
  sessionId?: string | null
) {
  const primary = mode === 'pinned' ? modelId : null

  return {
    role: role.role,
    mode,
    ...(primary ? { model_id: primary } : {}),
    fallback_ids: [...new Set((fallbackIds ?? role.fallback_ids ?? []).filter(id => Boolean(id) && id !== primary))],
    ...(sessionId ? { session_id: sessionId } : {})
  }
}

/** Avoid misrepresenting unexpected inventory data as a verified capability. */
export function capabilityNames(model: OrchestrationModel): string[] {
  if (Array.isArray(model.capabilities)) {
    return model.capabilities.filter(v => typeof v === 'string' && v.length > 0)
  }

  if (model.capabilities && typeof model.capabilities === 'object') {
    return Object.entries(model.capabilities)
      .filter(([, enabled]) => enabled === true)
      .map(([name]) => name)
  }

  return []
}

export function isOrchestrationStatus(value: unknown): value is OrchestrationStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const data = value as Partial<OrchestrationStatus>

  return Array.isArray(data.models) && Array.isArray(data.roles) && Array.isArray(data.profiles)
}
