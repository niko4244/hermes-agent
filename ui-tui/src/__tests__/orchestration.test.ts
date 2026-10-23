import { describe, expect, it } from 'vitest'

import {
  buildOrchestrationAssignment,
  availabilityLabel,
  capabilityNames,
  filterOrchestrationModels,
  isOrchestrationStatus,
  isDirectLocalModel,
  isVerifiedQualification,
  modelCostLabel,
  moveOrchestrationFallback,
  nextPolicy,
  parsePriceCap,
  policySummary,
  qualificationProvenance,
  qualificationRequest,
  qualificationSummary,
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

describe('orchestration policy controls and cost disclosure', () => {
  it('shows uncapped and unknown pricing honestly; preserves other policy caps on edits', () => {
    expect(policySummary({ local_only: false })).toContain('no cap (price not guaranteed)')
    const original = { local_only: true, max_input_cost_per_million: 1.25, max_output_cost_per_million: 5 }
    expect(nextPolicy(original, 'local_only', false)).toEqual({
      local_only: false,
      max_input_cost_per_million: 1.25,
      max_output_cost_per_million: 5
    })
    expect(nextPolicy(original, 'max_output_cost_per_million', null)).toEqual({
      local_only: true,
      max_input_cost_per_million: 1.25
    })
    expect(original.local_only).toBe(true)
    expect(original.max_output_cost_per_million).toBe(5)
    expect(() => nextPolicy(original, 'max_output_cost_per_million', Number.NaN)).toThrow(/nonnegative/)
    expect(modelCostLabel(models[1]!)).toContain('price unverified')
  })

  it('validates price cap strings without scientific notation, negative values or infinity', () => {
    expect(parsePriceCap('0')).toBe(0)
    expect(parsePriceCap('0.000001')).toBe(0.000001)
    expect(parsePriceCap('12.5')).toBe(12.5)
    for (const invalid of ['', 'unlimited', '-2', '1e9', 'NaN', '.0000001', '1.2.3', '1000001']) {
      expect(parsePriceCap(invalid)).toBeNull()
    }
  })

  it('does not turn catalog discovery or disk presence into inference qualification', () => {
    expect(availabilityLabel(models[1]!)).toBe('unavailable')
    expect(availabilityLabel({ ...models[0]!, availability: 'on_disk' })).toContain('not confirmed serving')
    expect(availabilityLabel({ ...models[0]!, availability: 'configured' })).toContain('unverified')
    expect(availabilityLabel({ ...models[0]!, availability: 'reachable' })).toContain('inference not verified')
  })
})

describe('explicit local-only qualification gate', () => {
  const local: OrchestrationModel = {
    id: 'custom@abc::qwen',
    provider: 'custom',
    model: 'qwen',
    availability: 'reachable',
    source: 'ollama-live',
    endpoint: 'http://127.0.0.1:11434/v1'
  }

  it('permits only observed direct Ollama on loopback with bounded probe arguments', () => {
    expect(isDirectLocalModel(local)).toBe(true)
    expect(isDirectLocalModel({ ...local, endpoint: 'http://[::1]:11434/v1' })).toBe(true)
    expect(qualificationRequest(local, 64, 20)).toEqual({
      model_id: local.id,
      max_tokens: 64,
      timeout_seconds: 20
    })
    for (const changes of [
      { source: 'openai-compatible-live' },
      { availability: 'on_disk' },
      { provider: 'openrouter' },
      { endpoint: 'https://example.com/v1' },
      { endpoint: 'http://localhost.evil.example/v1' },
      { endpoint: 'http://user:password@127.0.0.1/v1' },
      { endpoint: 'http://127.0.0.1:11434/v1?api_key=secret' }
    ]) {
      const notLocal = { ...local, ...changes }
      expect(isDirectLocalModel(notLocal)).toBe(false)
      expect(() => qualificationRequest(notLocal, 64, 20)).toThrow(/Only observed/)
    }
    expect(() => qualificationRequest(local, 257, 20)).toThrow(/token limit/)
    expect(() => qualificationRequest(local, 64, 61)).toThrow(/timeout/)
  })

  it('never reports pass for metadata-only or failed inference and omits unsafe provenance fields', () => {
    const metadataOnly = { model_id: local.id, status: 'passed', inference_tested: false, level: 'model-list-only' }
    expect(isVerifiedQualification(metadataOnly)).toBe(false)
    expect(qualificationSummary(metadataOnly)).toContain('inference NOT verified')
    expect(
      isVerifiedQualification({ model_id: local.id, inference_tested: true, passed: false, status: 'passed' })
    ).toBe(false)
    expect(isVerifiedQualification({ model_id: local.id, inference_tested: true, passed: true })).toBe(false)
    expect(
      isVerifiedQualification({
        model_id: local.id,
        status: 'passed',
        inference_tested: true,
        quality: { passed: true }
      })
    ).toBe(true)
    expect(
      qualificationProvenance({
        model_id: local.id,
        provenance: {
          source: 'ollama-live',
          api_key: 'must-not-display',
          method: 'inference',
          endpoint: 'http://unsafe.example/key'
        }
      })
    ).toBe('source: ollama-live · method: inference')
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
