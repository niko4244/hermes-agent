import { describe, expect, it } from 'vitest'

import {
  buildOrchestrationAssignment,
  capabilityNames,
  filterOrchestrationModels,
  isOrchestrationStatus,
  moveOrchestrationFallback,
  roleIndexAfterRefresh,
  toggleOrchestrationFallback,
  type OrchestrationModel,
  type OrchestrationRole
} from '../domain/orchestration.js'

const models: OrchestrationModel[] = [
  {
    id: 'ollama/qwen',
    provider: 'Ollama',
    model: 'qwen3.8',
    display_name: 'Qwen Coder',
    availability: 'loaded',
    source: 'local',
    capabilities: { tools: true, vision: false }
  },
  {
    id: 'litellm/opus',
    provider: 'LiteLLM',
    model: 'claude-opus',
    availability: 'unavailable',
    source: 'gateway',
    capabilities: ['tools', 'reasoning']
  }
]

const coder: OrchestrationRole = {
  role: 'coder',
  mode: 'pinned',
  model_id: 'ollama/qwen',
  fallback_ids: ['litellm/opus']
}

describe('orchestration TUI inventory', () => {
  it('can search across provider, model, display name, source and availability without hiding offline choices', () => {
    expect(filterOrchestrationModels(models, 'LOCAL coder').map(model => model.id)).toEqual(['ollama/qwen'])
    expect(filterOrchestrationModels(models, 'litellm unavailable').map(model => model.id)).toEqual(['litellm/opus'])
    expect(filterOrchestrationModels(models, 'missing')).toEqual([])
    expect(capabilityNames(models[0]!)).toEqual(['tools'])
    expect(capabilityNames(models[1]!)).toEqual(['tools', 'reasoning'])
  })

  it('rejects incomplete RPC inventory instead of displaying a blank success state', () => {
    expect(isOrchestrationStatus({ models, roles: [coder], profiles: [] })).toBe(true)
    expect(isOrchestrationStatus({ models, roles: [coder] })).toBe(false)
    expect(isOrchestrationStatus({ models: null, roles: [], profiles: [] })).toBe(false)
    expect(isOrchestrationStatus([])).toBe(false)
  })

  it('keeps the selected agent after a refreshed status reorders the role list', () => {
    expect(
      roleIndexAfterRefresh(
        [
          { role: 'main', mode: 'auto' },
          { role: 'coder', mode: 'pinned' },
          { role: 'reviewer', mode: 'auto' }
        ],
        'reviewer',
        1
      )
    ).toBe(2)
    expect(roleIndexAfterRefresh([{ role: 'main', mode: 'auto' }], 'reviewer', 2)).toBe(0)
  })
})

describe('orchestration assignment protocol', () => {
  it('respects session scope and removes the selected primary from ordered fallback candidates', () => {
    expect(
      buildOrchestrationAssignment(
        coder,
        'pinned',
        'litellm/opus',
        ['ollama/qwen', 'litellm/opus', 'ollama/qwen'],
        'session-1'
      )
    ).toEqual({
      role: 'coder',
      mode: 'pinned',
      model_id: 'litellm/opus',
      fallback_ids: ['ollama/qwen'],
      session_id: 'session-1'
    })
  })

  it('switches to auto without accidentally pinning the previous primary', () => {
    expect(buildOrchestrationAssignment(coder, 'auto', coder.model_id, undefined, null)).toEqual({
      role: 'coder',
      mode: 'auto',
      fallback_ids: ['litellm/opus']
    })
  })

  it('preserves deterministic fallback priority and cannot add the primary', () => {
    expect(toggleOrchestrationFallback(['litellm/opus'], 'ollama/qwen', coder.model_id)).toEqual(['litellm/opus'])
    expect(toggleOrchestrationFallback(['litellm/opus'], 'other', coder.model_id)).toEqual(['litellm/opus', 'other'])
    expect(toggleOrchestrationFallback(['litellm/opus', 'other'], 'litellm/opus', coder.model_id)).toEqual(['other'])
    expect(moveOrchestrationFallback(['first', 'second', 'third'], 'third', -1)).toEqual(['first', 'third', 'second'])
    expect(moveOrchestrationFallback(['first', 'second'], 'first', -1)).toEqual(['first', 'second'])
  })
})
