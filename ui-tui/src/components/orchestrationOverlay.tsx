import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useRef, useState } from 'react'

import type {
  OrchestrationAssignResult,
  OrchestrationModel,
  OrchestrationProfile,
  OrchestrationRole,
  OrchestrationStatus
} from '../domain/orchestration.js'
import {
  buildOrchestrationAssignment,
  capabilityNames,
  filterOrchestrationModels,
  isOrchestrationStatus,
  modelLabel,
  moveOrchestrationFallback,
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

type Stage = 'roles' | 'models' | 'fallbacks' | 'profiles' | 'search' | 'save' | 'create-role'
const TABS: readonly Stage[] = ['roles', 'models', 'profiles']

interface OrchestrationOverlayProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  sessionId: null | string
  t: Theme
}

/** The backend owns persistence and routing; this panel is a view of its confirmed state. */
export function OrchestrationOverlay({ gw, maxWidth, onClose, sessionId, t }: OrchestrationOverlayProps) {
  const [data, setData] = useState<OrchestrationStatus | null>(null)
  const [stage, setStage] = useState<Stage>('roles')
  const [roleIdx, setRoleIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [profileIdx, setProfileIdx] = useState(0)
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

    gw.request<OrchestrationStatus>(refresh ? 'orchestration.refresh' : 'orchestration.status', refresh ? {} : {})
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
  const selectedModelName = (id?: string | null): string => {
    if (!id) {
      return 'automatic'
    }

    const found = allModels.find(model => model.id === id)

    return found ? modelLabel(found) : `${id} (not in current inventory)`
  }

  const mutate = (method: string, params: Record<string, unknown>, success: string, onSaved?: () => void) => {
    if (busy) {
      return
    }

    setBusy(true)
    setError('')
    setNotice('')
    gw.request<OrchestrationAssignResult>(method, params)
      .then(result => {
        if (result?.confirm_required) {
          setError(
            result.warning || 'This provider requires price confirmation. Use /model to review before switching.'
          )

          return
        }

        setNotice(
          [success, result?.deferred ? 'applies at next turn/spawn' : '', result?.warning ?? '']
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

    mutate(
      'orchestration.assign',
      buildOrchestrationAssignment(activeRole, mode, modelId, fallbackIds, sessionId),
      `${activeRole.role}: ${mode === 'auto' ? 'automatic' : selectedModelName(modelId)}`,
      onSaved
    )
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

    const length = stage === 'roles' ? roles.length : stage === 'profiles' ? profiles.length : models.length
    const idx = stage === 'roles' ? roleIdx : stage === 'profiles' ? profileIdx : modelIdx
    const updateIdx = stage === 'roles' ? setRoleIdx : stage === 'profiles' ? setProfileIdx : setModelIdx

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

    if (ch === '/') {
      setStage('search')

      return
    }

    if (stage === 'models') {
      if (key.return && activeModel) {
        assign('pinned', activeModel.id, undefined, () => setStage('roles'))
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
        : stage === 'fallbacks' || stage === 'create-role'
          ? 'roles'
          : stage
  const title = `[${TABS.map(tab => (tab === tabName ? tab.toUpperCase() : tab)).join(' · ')}]`
  const list = stage === 'roles' ? roles : stage === 'profiles' || stage === 'save' ? profiles : models
  const cursor = Math.min(
    stage === 'roles' ? roleIdx : stage === 'profiles' ? profileIdx : modelIdx,
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
      {busy && <Text color={t.color.accent}>saving configuration…</Text>}
      {stage === 'save' && <Text color={t.color.text}>Save current assignments as profile: {draftName || '▌'}</Text>}
      {stage === 'create-role' && <Text color={t.color.text}>Agent role name: {draftRole || '▌'}</Text>}
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

        if (stage === 'roles') {
          const role = item as OrchestrationRole
          label = `${role.role} · ${role.mode === 'auto' ? 'auto' : selectedModelName(role.model_id)} · ${role.fallback_ids?.length ?? 0} fallbacks`
        } else if (stage === 'profiles' || stage === 'save') {
          const profile = item as OrchestrationProfile
          label = `${profile.name}${profile.name === data?.active_profile ? ' ✓ active' : ''}`
        } else {
          const model = item as OrchestrationModel
          const selectedFallback = fallbackDraft.indexOf(model.id)
          const fallbackMark =
            stage === 'fallbacks' ? (selectedFallback >= 0 ? `[${selectedFallback + 1}] ` : '[ ] ') : ''
          label = `${fallbackMark}${modelLabel(model)} · ${model.availability || 'unknown'}${model.source ? ` · ${model.source}` : ''}`
        }

        return (
          <Text
            {...chipRowProps(t, selected)}
            key={
              stage === 'roles'
                ? (item as OrchestrationRole).role
                : stage === 'profiles' || stage === 'save'
                  ? (item as OrchestrationProfile).name
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
        <Text color={t.color.muted} wrap="truncate-end">
          {activeModel.context_window ? `context ${activeModel.context_window.toLocaleString()} · ` : ''}
          {capabilityNames(activeModel).join(', ') || 'capabilities not reported'}
        </Text>
      )}
      <OverlayHint t={t}>
        {stage === 'roles'
          ? '↑↓ role · Enter models · a auto · f fallbacks · n new role · Tab section · r refresh'
          : stage === 'models'
            ? '↑↓ choose model · Enter pin · / filter · Tab section · Esc roles · q close'
            : stage === 'fallbacks'
              ? '↑↓ model · Space toggle · Shift+↑↓ reorder · Enter save · Esc cancel'
              : stage === 'profiles'
                ? '↑↓ choose profile · Enter activate · n save · Tab section · Esc/q close'
                : stage === 'save'
                  ? 'type profile name · Enter save · Esc cancel'
                  : stage === 'create-role'
                    ? 'type role name · Enter choose · Esc cancel'
                    : 'type to filter · Enter finish · Esc finish'}
      </OverlayHint>
    </Box>
  )
}
