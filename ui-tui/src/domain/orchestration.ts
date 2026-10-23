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
