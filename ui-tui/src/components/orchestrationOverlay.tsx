import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type { RpcMethods } from '@hermes/shared/gateway-events'
import { useEffect, useRef, useState } from 'react'

import type {
  OrchestrationAssignResult,
  OrchestrationModel,
  OrchestrationPolicy,
  OrchestrationProfile,
  OrchestrationQualificationResult,
  OrchestrationRole,
  OrchestrationStatus
} from '../domain/orchestration.js'
import {
  buildOrchestrationAssignment,
  availabilityLabel,
  capabilityNames,
  filterOrchestrationModels,
  isOrchestrationStatus,
  isDirectLocalModel,
  isVerifiedQualification,
  modelCostLabel,
  modelLabel,
  moveOrchestrationFallback,
  nextPolicy,
  parsePriceCap,
  policySummary,
  priceLimitLabel,
  qualificationProvenance,
  qualificationReasonLabel,
  qualificationRequest,
  qualificationSummary,
  roleIndexAfterRefresh,
  toggleOrchestrationFallback
} from '../domain/orchestration.js'
import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const MAX_WIDTH = 100
const VISIBLE = 10

type Stage =
  | 'roles'
  | 'models'
  | 'fallbacks'
  | 'profiles'
  | 'policy'
  | 'search'
  | 'save'
  | 'create-role'
  | 'cap-input'
  | 'policy-confirm'
  | 'cloud-confirm'
  | 'qualify-confirm'
const TABS: readonly Stage[] = ['roles', 'models', 'profiles', 'policy']
const POLICY_FIELDS = ['local_only', 'max_input_cost_per_million', 'max_output_cost_per_million'] as const
const QUALIFY_TOKEN_OPTIONS = [16, 32, 64]
const QUALIFY_TIMEOUT_OPTIONS = [10, 20, 30]

// The Pydantic gateway contract is the authority on wire names, params, and
// results. A missing/renamed method fails TypeScript compilation, not at click.
type OrchestrationRpcName = Extract<keyof RpcMethods, `orchestration.${string}`>
type OrchestrationMutationName = Extract<
  OrchestrationRpcName,
  'orchestration.assign' | 'orchestration.profile' | 'orchestration.policy'
>

interface OrchestrationOverlayProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  sessionId: null | string
  t: Theme
}

/** The backend owns persistence and routing; this panel is a view of its confirmed state. */
export function OrchestrationOverlay({ gw, maxWidth, onClose, sessionId, t }: OrchestrationOverlayProps) {
  const requestOrchestration = <M extends OrchestrationRpcName>(method: M, params: RpcMethods[M]['params']) =>
    // All six calls must read the selected conversation's profile scope. A
    // pooled Desktop gateway may serve several distinct homes at once.
    gw.request<RpcMethods[M]['result']>(
      method,
      { ...(params as Record<string, unknown>), ...(sessionId ? { session_id: sessionId } : {}) }
    )
  const [data, setData] = useState<OrchestrationStatus | null>(null)
  const [stage, setStage] = useState<Stage>('roles')
  const [roleIdx, setRoleIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [profileIdx, setProfileIdx] = useState(0)
  const [policyIdx, setPolicyIdx] = useState(0)
  const [draftCap, setDraftCap] = useState('')
  const [pendingPolicy, setPendingPolicy] = useState<OrchestrationPolicy | null>(null)
  const [pendingCloud, setPendingCloud] = useState<{
    mode: 'auto' | 'pinned'
    modelId?: string | null
    fallbackIds?: string[]
    onSaved?: () => void
    from: Stage
  } | null>(null)
  const [qualifyTarget, setQualifyTarget] = useState<OrchestrationModel | null>(null)
  const [qualifyTokens, setQualifyTokens] = useState(32)
  const [qualifyTimeout, setQualifyTimeout] = useState(20)
  const [qualification, setQualification] = useState<OrchestrationQualificationResult | null>(null)
  const [qualifying, setQualifying] = useState(false)
  const [query, setQuery] = useState('')
  const [draftName, setDraftName] = useState('')
  const [draftRole, setDraftRole] = useState('')
  const [pendingRole, setPendingRole] = useState('')
  const [fallbackDraft, setFallbackDraft] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const requestId = useRef(0)
  const { stdout } = useStdout()
  const width = clampOverlayWidth(Math.max(40, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6)), maxWidth)

  const load = (refresh = false) => {
    const request = ++requestId.current
    const selectedRole = activeRole?.role
    setLoading(!data)
    setError('')

    requestOrchestration(refresh ? 'orchestration.refresh' : 'orchestration.status', {})
      .then(result => {
        if (request !== requestId.current) {
          return
        }

        if (!isOrchestrationStatus(result)) {
          setError('invalid orchestration inventory from gateway')

          return
        }

        setData(result)
        const nextRoles: OrchestrationRole[] = [
          ...(!result.roles.some(role => role.role === 'main')
            ? [{ role: 'main', mode: 'auto' as const, fallback_ids: [] }]
            : []),
          ...result.roles,
          ...(pendingRole && pendingRole !== 'main' && !result.roles.some(role => role.role === pendingRole)
            ? [{ role: pendingRole, mode: 'auto' as const, fallback_ids: [] }]
            : [])
        ]
        setRoleIdx(i => roleIndexAfterRefresh(nextRoles, selectedRole, i))
        setProfileIdx(i => Math.min(i, Math.max(0, result.profiles.length - 1)))
      })
      .catch((reason: unknown) => {
        if (request === requestId.current) {
          setError(rpcErrorMessage(reason))
        }
      })
      .finally(() => {
        if (request === requestId.current) {
          setLoading(false)
        }
      })
  }

  useEffect(() => {
    load()

    return () => {
      requestId.current += 1
    }
    // The initial read runs once for this gateway; subsequent refreshes are explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw])

  const serverRoles = data?.roles ?? []
  // A new installation has no assignments; keep the main agent selectable.
  // Additional role names are user-created, never guessed from providers.
  const roles: OrchestrationRole[] = [
    ...(!serverRoles.some(role => role.role === 'main')
      ? [{ role: 'main', mode: 'auto' as const, fallback_ids: [] }]
      : []),
    ...serverRoles,
    ...(pendingRole && pendingRole !== 'main' && !serverRoles.some(role => role.role === pendingRole)
      ? [{ role: pendingRole, mode: 'auto' as const, fallback_ids: [] }]
      : [])
  ]
  const allModels = data?.models ?? []
  const models = filterOrchestrationModels(allModels, query)
  const profiles = data?.profiles ?? []
  const activeRole: OrchestrationRole | undefined = roles[Math.min(roleIdx, roles.length - 1)]
  const activeModel: OrchestrationModel | undefined = models[Math.min(modelIdx, models.length - 1)]
  const activeProfile: OrchestrationProfile | undefined = profiles[Math.min(profileIdx, profiles.length - 1)]
  const policy = data?.policy
  const policyField = POLICY_FIELDS[policyIdx] ?? POLICY_FIELDS[0]
  const selectedModelName = (id?: string | null): string => {
    if (!id) {
      return 'automatic'
    }

    const found = allModels.find(model => model.id === id)

    return found ? modelLabel(found) : `${id} (not in current inventory)`
  }

  const mutate = <M extends OrchestrationMutationName>(
    method: M,
    params: RpcMethods[M]['params'],
    success: string,
    onSaved?: () => void
  ) => {
    if (busy) {
      return
    }

    setBusy(true)
    setError('')
    setNotice('')
    requestOrchestration(method, params)
      .then(result => {
        const summary = result as OrchestrationAssignResult
        if (summary?.confirm_required) {
          setError(
            summary.warning || 'This provider requires price confirmation. Use /model to review before switching.'
          )

          return
        }

        setNotice(
          [success, summary?.deferred ? 'applies at next turn/spawn' : '', summary?.warning ?? '']
            .filter(Boolean)
            .join(' · ')
        )
        onSaved?.()
        // Always re-read server-authoritative state; failed writes never become local selections.
        load()
      })
      .catch((reason: unknown) => setError(rpcErrorMessage(reason)))
      .finally(() => setBusy(false))
  }

  const assign = (mode: 'auto' | 'pinned', modelId?: string | null, fallbackIds?: string[], onSaved?: () => void) => {
    if (!activeRole) {
      setError('select an agent role first')

      return
    }

    const assignment = buildOrchestrationAssignment(activeRole, mode, modelId, fallbackIds, sessionId)
    // A remote configured name or localhost proxy can incur cloud charges.
    // Get a deliberate keyboard confirmation before persisting a deferred
    // main/delegation route (which may execute much later without a prompt).
    const selectedIds = [...(assignment.model_id ? [assignment.model_id] : []), ...assignment.fallback_ids]
    if (selectedIds.some(id => {
      const row = allModels.find(model => model.id === id)
      return row && !isDirectLocalModel(row)
    })) {
      setPendingCloud({ mode, modelId, fallbackIds, onSaved, from: stage })
      setError('')
      setStage('cloud-confirm')
      return
    }
    mutate(
      'orchestration.assign',
      assignment,
      `${activeRole.role}: ${mode === 'auto' ? 'automatic' : selectedModelName(modelId)}`,
      onSaved
    )
  }

  const proposePolicy = (field: (typeof POLICY_FIELDS)[number], value: boolean | number | null) => {
    if (!policy) {
      setError('Policy is not available from the gateway. Refresh with r and check the backend version.')

      return
    }

    try {
      setPendingPolicy(nextPolicy(policy, field, value))
      setError('')
      setStage('policy-confirm')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Invalid policy value.')
    }
  }

  const qualify = (model: OrchestrationModel) => {
    if (busy || !isDirectLocalModel(model)) {
      setError(
        'Qualification is restricted to live direct local Ollama models. Refresh with r; other proxies might incur cloud costs.'
      )

      return
    }

    let params: ReturnType<typeof qualificationRequest>
    try {
      params = qualificationRequest(model, qualifyTokens, qualifyTimeout)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Invalid qualification limits.')

      return
    }

    setBusy(true)
    setQualifying(true)
    setError('')
    setNotice('')
    setQualification(null)
    requestOrchestration('orchestration.qualify', params)
      .then(result => {
        if (!result || result.model_id !== model.id) {
          setError('Qualification response did not identify the requested model; no result accepted.')

          return
        }

        const evidence = result as unknown as OrchestrationQualificationResult
        setQualification(evidence)
        if (isVerifiedQualification(evidence)) {
          setNotice(`Local inference probe passed for ${modelLabel(model)}.`)
        } else if (result.inference_tested !== true) {
          setError(
            'Only metadata or reachability was checked; inference is NOT qualified. Verify model loading and backend probe support.'
          )
        } else {
          setError(
            'Inference was attempted, but no passing result was confirmed. Inspect the result and local server before retrying.'
          )
        }
      })
      .catch((reason: unknown) =>
        setError(`Local qualification failed: ${rpcErrorMessage(reason)}. Check Ollama and try r to refresh.`)
      )
      .finally(() => {
        setBusy(false)
        setQualifying(false)
        setQualifyTarget(null)
        setStage('models')
      })
  }

  const back = () => {
    if (stage === 'search') {
      setStage('models')

      return
    }

    if (stage === 'save') {
      setStage('profiles')
      setDraftName('')

      return
    }

    if (stage === 'create-role') {
      setStage('roles')
      setDraftRole('')

      return
    }

    if (stage === 'cap-input' || stage === 'policy-confirm') {
      setDraftCap('')
      setPendingPolicy(null)
      setStage('policy')

      return
    }

    if (stage === 'qualify-confirm') {
      setQualifyTarget(null)
      setStage('models')

      return
    }

    if (stage === 'cloud-confirm') {
      setStage(pendingCloud?.from ?? 'roles')
      setPendingCloud(null)
      return
    }

    if (stage === 'models' || stage === 'fallbacks') {
      setStage('roles')
      setQuery('')
      setModelIdx(0)

      return
    }

    onClose()
  }

  useInput((ch, key) => {
    if (busy) {
      return
    }

    if (stage === 'cloud-confirm') {
      if (key.escape || ch === 'n') {
        back()
      } else if (ch === 'y' && pendingCloud && activeRole) {
        mutate(
          'orchestration.assign',
          { ...buildOrchestrationAssignment(activeRole, pendingCloud.mode,
              pendingCloud.modelId, pendingCloud.fallbackIds, sessionId),
            confirm_cloud: true },
          `${activeRole.role}: confirmed remote model assignment`,
          () => {
            pendingCloud.onSaved?.()
            setPendingCloud(null)
          }
        )
      }
      return
    }

    if (stage === 'policy-confirm') {
      if (key.escape || ch === 'n') {
        back()
      } else if (ch === 'y' && pendingPolicy) {
        mutate('orchestration.policy', { action: 'set', policy: pendingPolicy }, 'saved active profile policy', () => {
          setPendingPolicy(null)
          setStage('policy')
        })
      }

      return
    }

    if (stage === 'qualify-confirm') {
      if (key.escape || ch === 'n') {
        back()
      } else if (ch === '[' || ch === ']') {
        setQualifyTokens(current => {
          const index = QUALIFY_TOKEN_OPTIONS.indexOf(current)
          return (
            QUALIFY_TOKEN_OPTIONS[
              Math.max(0, Math.min(QUALIFY_TOKEN_OPTIONS.length - 1, index + (ch === '[' ? -1 : 1)))
            ] ?? current
          )
        })
      } else if (ch === '-' || ch === '+') {
        setQualifyTimeout(current => {
          const index = QUALIFY_TIMEOUT_OPTIONS.indexOf(current)
          return (
            QUALIFY_TIMEOUT_OPTIONS[
              Math.max(0, Math.min(QUALIFY_TIMEOUT_OPTIONS.length - 1, index + (ch === '-' ? -1 : 1)))
            ] ?? current
          )
        })
      } else if (ch === 'y' && qualifyTarget) {
        qualify(qualifyTarget)
      }

      return
    }

    if (stage === 'cap-input') {
      if (key.escape) {
        back()
      } else if (key.return) {
        const value = parsePriceCap(draftCap)
        if (value === null) {
          setError(
            'Enter a nonnegative USD cap per million tokens (up to $1,000,000; max 6 decimals), or Esc and c to clear.'
          )
        } else if (policyField !== 'local_only') {
          proposePolicy(policyField, value)
        }
      } else if (key.backspace || key.delete) {
        setDraftCap(value => value.slice(0, -1))
      } else if (/^[0-9.]$/.test(ch) && draftCap.length < 24 && !key.ctrl && !key.meta) {
        setDraftCap(value => value + ch)
      }

      return
    }

    if (stage === 'search' || stage === 'save' || stage === 'create-role') {
      if (key.escape) {
        back()
      } else if (key.return) {
        if (stage === 'save' && draftName.trim()) {
          mutate(
            'orchestration.profile',
            { action: 'save', name: draftName.trim() },
            `saved profile ${draftName.trim()}`,
            () => {
              setDraftName('')
              setStage('profiles')
            }
          )
        }
        if (stage === 'create-role') {
          const candidate = draftRole.trim()

          if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$/.test(candidate)) {
            setError('role must be 1–100 letters, numbers, periods, dashes or underscores')

            return
          }

          const existing = roles.findIndex(role => role.role === candidate)
          setRoleIdx(existing < 0 ? roles.length : existing)
          setPendingRole(candidate)
          setDraftRole('')
          setError('')
          setStage('roles')
        }
        if (stage === 'search') {
          setStage('models')
        }
      } else if (key.backspace || key.delete) {
        if (stage === 'search') {
          setQuery(value => value.slice(0, -1))
        } else if (stage === 'create-role') {
          setDraftRole(value => value.slice(0, -1))
        } else {
          setDraftName(value => value.slice(0, -1))
        }
      } else if (ch && !key.ctrl && !key.meta) {
        if (stage === 'search') {
          setQuery(value => value + ch)
          setModelIdx(0)
        } else if (stage === 'create-role') {
          setDraftRole(value => value + ch)
        } else {
          setDraftName(value => value + ch)
        }
      }

      return
    }

    if (key.escape) {
      back()

      return
    }

    if (ch === 'q') {
      onClose()

      return
    }

    if (key.tab) {
      const current = TABS.indexOf(stage)
      setStage(TABS[(current + 1 + TABS.length) % TABS.length]!)
      setError('')

      return
    }

    if ((ch === 'r' || (key.ctrl && ch === 'r')) && !loading) {
      load(true)

      return
    }

    const length =
      stage === 'roles'
        ? roles.length
        : stage === 'profiles'
          ? profiles.length
          : stage === 'policy'
            ? POLICY_FIELDS.length
            : models.length
    const idx =
      stage === 'roles' ? roleIdx : stage === 'profiles' ? profileIdx : stage === 'policy' ? policyIdx : modelIdx
    const updateIdx =
      stage === 'roles'
        ? setRoleIdx
        : stage === 'profiles'
          ? setProfileIdx
          : stage === 'policy'
            ? setPolicyIdx
            : setModelIdx

    if (stage === 'fallbacks' && key.shift && (key.upArrow || key.downArrow) && activeModel) {
      setFallbackDraft(existing => moveOrchestrationFallback(existing, activeModel.id, key.upArrow ? -1 : 1))

      return
    }

    if (key.upArrow || ch === 'k') {
      updateIdx(Math.max(0, idx - 1))

      return
    }

    if (key.downArrow || ch === 'j') {
      updateIdx(Math.min(length - 1, idx + 1))

      return
    }

    if (stage === 'roles') {
      if (ch === 'a') {
        assign('auto')
      } else if (ch === 'n') {
        setDraftRole('')
        setStage('create-role')
      } else if (ch === 'f' && activeRole) {
        setFallbackDraft([...new Set((activeRole.fallback_ids ?? []).filter(id => id !== activeRole.model_id))])
        setModelIdx(0)
        setQuery('')
        setStage('fallbacks')
      } else if ((key.return || ch === 'm') && activeRole) {
        setModelIdx(
          Math.max(
            0,
            allModels.findIndex(model => model.id === activeRole.model_id)
          )
        )
        setQuery('')
        setStage('models')
      }

      return
    }

    if (stage === 'profiles') {
      if (ch === 'n') {
        setDraftName('')
        setStage('save')
      } else if (key.return && activeProfile) {
        mutate(
          'orchestration.profile',
          { action: 'activate', name: activeProfile.name },
          `activated ${activeProfile.name}`
        )
      }

      return
    }

    if (stage === 'policy') {
      if (key.return || ch === ' ') {
        if (policyField === 'local_only') {
          proposePolicy('local_only', policy?.local_only !== true)
        } else if (policy) {
          const existing = policy[policyField]
          setDraftCap(existing === undefined ? '' : String(existing))
          setError('')
          setStage('cap-input')
        } else {
          setError('Policy unavailable; refresh with r or update the gateway.')
        }
      } else if (ch === 'c' && policyField !== 'local_only') {
        proposePolicy(policyField, null)
      }

      return
    }

    if (ch === '/') {
      setStage('search')

      return
    }

    if (stage === 'models') {
      if (key.return && activeModel) {
        assign('pinned', activeModel.id, undefined, () => setStage('roles'))
      } else if (ch === 'v' && activeModel) {
        if (!isDirectLocalModel(activeModel)) {
          setError(
            'Not a confirmed direct local Ollama model. Qualification is disabled for cloud providers, proxies, disk-only or stale models.'
          )
        } else {
          setQualifyTarget(activeModel)
          setQualification(null)
          setError('')
          setStage('qualify-confirm')
        }
      }

      return
    }

    if (stage === 'fallbacks') {
      if (ch === ' ' && activeModel) {
        setFallbackDraft(existing => toggleOrchestrationFallback(existing, activeModel.id, activeRole?.model_id))
      } else if (key.return && activeRole) {
        assign(activeRole.mode, activeRole.model_id, fallbackDraft, () => setStage('roles'))
      }
    }
  })

  const tabName =
    stage === 'search'
      ? 'models'
      : stage === 'save'
        ? 'profiles'
        : stage === 'fallbacks' || stage === 'create-role' || stage === 'cloud-confirm'
          ? 'roles'
          : stage === 'cap-input' || stage === 'policy-confirm'
            ? 'policy'
            : stage === 'qualify-confirm'
              ? 'models'
              : stage
  const title = `[${TABS.map(tab => (tab === tabName ? tab.toUpperCase() : tab)).join(' · ')}]`
  const list =
    stage === 'roles' || stage === 'create-role'
      ? roles
      : stage === 'profiles' || stage === 'save'
        ? profiles
        : stage === 'policy' || stage === 'cap-input' || stage === 'policy-confirm'
          ? [...POLICY_FIELDS]
          : models
  const cursor = Math.min(
    stage === 'roles' || stage === 'create-role'
      ? roleIdx
      : stage === 'profiles' || stage === 'save'
        ? profileIdx
        : stage === 'policy' || stage === 'cap-input' || stage === 'policy-confirm'
          ? policyIdx
          : modelIdx,
    Math.max(0, list.length - 1)
  )
  const { items, offset } = windowItems([...list], cursor, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Agent Model Orchestration
      </Text>
      <Text color={t.color.muted}>{title}</Text>
      {data && (
        <Text color={t.color.label} wrap="truncate-end">
          {allModels.length} models · {roles.length} roles · active profile: {data.active_profile || 'default'}
        </Text>
      )}
      {data && (
        <Text color={t.color.label} wrap="truncate-end">
          Policy: local only {policy?.local_only === true ? 'ON' : policy ? 'OFF' : 'unavailable'} · caps: input{' '}
          {priceLimitLabel(policy?.max_input_cost_per_million)}, output{' '}
          {priceLimitLabel(policy?.max_output_cost_per_million)}
        </Text>
      )}
      {loading && <Text color={t.color.muted}>loading model inventory…</Text>}
      {error && (
        <Text color={t.color.error} wrap="truncate-end">
          error: {error}
        </Text>
      )}
      {notice && (
        <Text color={t.color.ok} wrap="truncate-end">
          {notice}
        </Text>
      )}
      {busy && (
        <Text color={t.color.accent}>
          {qualifying ? 'running opt-in local inference probe…' : 'saving configuration…'}
        </Text>
      )}
      {stage === 'save' && <Text color={t.color.text}>Save current assignments as profile: {draftName || '▌'}</Text>}
      {stage === 'create-role' && <Text color={t.color.text}>Agent role name: {draftRole || '▌'}</Text>}
      {stage === 'cap-input' && (
        <Text color={t.color.text}>
          USD per 1M {policyField === 'max_input_cost_per_million' ? 'input' : 'output'} tokens: {draftCap || '▌'}
        </Text>
      )}
      {stage === 'policy' && (
        <Text color={t.color.label}>Active profile routing guardrails · {policySummary(policy)}</Text>
      )}
      {stage === 'policy' && (
        <Text color={t.color.muted}>
          Unverified model prices cannot satisfy a price cap; limits may block routes even for local models.
        </Text>
      )}
      {stage === 'policy-confirm' && pendingPolicy && (
        <Box flexDirection="column">
          <Text bold color={t.color.accent}>
            Confirm active profile policy change?
          </Text>
          <Text color={t.color.label}>Proposed: {policySummary(pendingPolicy)}</Text>
          <Text color={t.color.muted}>
            Changing these limits can permit billable model routing. y save · n/Esc cancel
          </Text>
        </Box>
      )}
      {stage === 'cloud-confirm' && pendingCloud && (
        <Box flexDirection="column">
          <Text bold color={t.color.accent}>
            Confirm remote model or fallback assignment?
          </Text>
          <Text color={t.color.label}>
            Configured providers and local proxies may call billable cloud APIs; model prices are not verified.
          </Text>
          <Text color={t.color.muted}>y confirm and save · n/Esc cancel without saving</Text>
        </Box>
      )}
      {stage === 'qualify-confirm' && qualifyTarget && (
        <Box flexDirection="column">
          <Text bold color={t.color.accent}>
            Run a real local inference qualification for {modelLabel(qualifyTarget)}?
          </Text>
          <Text color={t.color.label}>Model: {qualifyTarget.id} · directly observed Ollama only</Text>
          <Text color={t.color.label}>
            Bounded probe: {qualifyTokens} max output tokens · {qualifyTimeout}s timeout
          </Text>
          <Text color={t.color.muted}>
            Requires an already-loaded model. Uses local compute only; no model pull or cloud request. y run · n/Esc
            cancel
          </Text>
        </Box>
      )}
      {stage === 'search' && <Text color={t.color.text}>Search providers/models: {query || '▌'}</Text>}
      {(stage === 'models' || stage === 'fallbacks' || stage === 'search') && activeRole && (
        <Text color={t.color.label} wrap="truncate-end">
          {stage === 'fallbacks' ? 'Fallbacks for' : 'Primary model for'} {activeRole.role} ·{' '}
          {selectedModelName(activeRole.model_id)}
        </Text>
      )}
      {stage === 'fallbacks' && (
        <Text color={t.color.muted} wrap="truncate-end">
          Priority: {fallbackDraft.length ? fallbackDraft.map(selectedModelName).join(' → ') : 'none'}
        </Text>
      )}
      {qualification && (
        <Box flexDirection="column">
          <Text color={isVerifiedQualification(qualification) ? t.color.ok : t.color.muted}>
            Qualification: {qualificationSummary(qualification)}
          </Text>
          <Text color={t.color.muted} wrap="truncate-end">
            Provenance: {qualificationProvenance(qualification)}
          </Text>
          {qualificationReasonLabel(qualification) && (
            <Text color={t.color.muted}>Probe: {qualificationReasonLabel(qualification)}</Text>
          )}
          {qualification.inference_tested === true && (
            <Text color={t.color.muted}>
              Single fixed arithmetic check:{' '}
              {qualification.quality?.passed === true ? 'exact match' : 'not passed or not reported'} (not broad model
              quality)
            </Text>
          )}
          {typeof qualification.latency_ms === 'number' && Number.isFinite(qualification.latency_ms) && (
            <Text color={t.color.muted}>Measured latency: {Math.round(qualification.latency_ms)}ms</Text>
          )}
          {typeof qualification.elapsed_ms === 'number' && Number.isFinite(qualification.elapsed_ms) && (
            <Text color={t.color.muted}>
              Elapsed: {Math.round(qualification.elapsed_ms)}ms
              {typeof qualification.ttft_ms === 'number' && Number.isFinite(qualification.ttft_ms)
                ? ` · first text: ${Math.round(qualification.ttft_ms)}ms`
                : ''}
              {typeof qualification.tokens_per_second === 'number' && Number.isFinite(qualification.tokens_per_second)
                ? ` · ${qualification.tokens_per_second.toFixed(1)} tokens/s (Ollama reported)`
                : ''}
            </Text>
          )}
        </Box>
      )}
      {!loading && !list.length && stage !== 'save' && (
        <Text color={t.color.muted}>
          {stage === 'search' || stage === 'models' || stage === 'fallbacks'
            ? 'no models match the current filter'
            : stage === 'profiles'
              ? 'no saved profiles · press n to save'
              : 'no agent roles available'}
        </Text>
      )}
      {offset > 0 && <Text color={t.color.muted}>↑ {offset} more</Text>}
      {items.map((item, index) => {
        const selected = cursor === offset + index
        let label = ''

        if (stage === 'roles' || stage === 'create-role') {
          const role = item as OrchestrationRole
          label = `${role.role} · ${role.mode === 'auto' ? 'auto' : selectedModelName(role.model_id)} · ${role.fallback_ids?.length ?? 0} fallbacks`
        } else if (stage === 'profiles' || stage === 'save') {
          const profile = item as OrchestrationProfile
          label = `${profile.name}${profile.name === data?.active_profile ? ' ✓ active' : ''}`
        } else if (stage === 'policy' || stage === 'cap-input' || stage === 'policy-confirm') {
          const field = item as (typeof POLICY_FIELDS)[number]
          label =
            field === 'local_only'
              ? `Local-only routing: ${policy?.local_only === true ? 'ON (observed direct Ollama only)' : policy ? 'OFF (cloud possible)' : 'unavailable'}`
              : `${field === 'max_input_cost_per_million' ? 'Input' : 'Output'} USD cap: ${priceLimitLabel(policy?.[field])}`
        } else {
          const model = item as OrchestrationModel
          const selectedFallback = fallbackDraft.indexOf(model.id)
          const fallbackMark =
            stage === 'fallbacks' ? (selectedFallback >= 0 ? `[${selectedFallback + 1}] ` : '[ ] ') : ''
          label = `${fallbackMark}${modelLabel(model)} · ${availabilityLabel(model)}${model.source ? ` · ${model.source}` : ''}`
        }

        return (
          <Text
            {...chipRowProps(t, selected)}
            key={
              stage === 'roles' || stage === 'create-role'
                ? (item as OrchestrationRole).role
                : stage === 'profiles' || stage === 'save'
                  ? (item as OrchestrationProfile).name
                  : stage === 'policy' || stage === 'cap-input' || stage === 'policy-confirm'
                    ? (item as string)
                    : (item as OrchestrationModel).id
            }
            wrap="truncate-end"
          >
            {selected ? '▸ ' : '  '}
            {label}
          </Text>
        )
      })}
      {offset + VISIBLE < list.length && <Text color={t.color.muted}>↓ {list.length - offset - VISIBLE} more</Text>}
      {(stage === 'models' || stage === 'search' || stage === 'fallbacks') && activeModel && (
        <Box flexDirection="column">
          <Text color={t.color.muted} wrap="truncate-end">
            {activeModel.context_window ? `reported context ${activeModel.context_window.toLocaleString()} · ` : ''}
            {capabilityNames(activeModel).join(', ') || 'capabilities not verified'}
          </Text>
          <Text color={t.color.muted} wrap="truncate-end">
            Cost: {modelCostLabel(activeModel)}
          </Text>
          <Text color={t.color.muted}>
            {isDirectLocalModel(activeModel)
              ? 'v = explicitly request local inference probe (never automatic)'
              : 'Qualification disabled: not observed direct local Ollama'}
          </Text>
        </Box>
      )}
      <OverlayHint t={t}>
        {stage === 'roles'
          ? '↑↓ role · Enter models · a auto · f fallbacks · n new role · Tab section · r refresh'
          : stage === 'models'
            ? '↑↓ choose model · Enter pin · v verify local (confirm) · / filter · Tab section · Esc roles'
            : stage === 'fallbacks'
              ? '↑↓ model · Space toggle · Shift+↑↓ reorder · Enter save · Esc cancel'
              : stage === 'profiles'
                ? '↑↓ choose profile · Enter activate · n save · Tab section · Esc/q close'
                : stage === 'policy'
                  ? '↑↓ policy field · Enter edit/toggle · c clear selected cap · Tab section · Esc close'
                  : stage === 'cap-input'
                    ? 'type nonnegative USD cap per 1M tokens · Enter review · Esc cancel'
                    : stage === 'policy-confirm'
                      ? 'y confirm policy write · n/Esc cancel'
                  : stage === 'cloud-confirm'
                    ? 'y confirm possible cloud charges · n/Esc cancel'
                      : stage === 'qualify-confirm'
                        ? 'y run local-only probe · [ ] token limit · - + timeout · n/Esc cancel'
                        : stage === 'save'
                          ? 'type profile name · Enter save · Esc cancel'
                          : stage === 'create-role'
                            ? 'type role name · Enter choose · Esc cancel'
                            : 'type to filter · Enter finish · Esc finish'}
      </OverlayHint>
    </Box>
  )
}
