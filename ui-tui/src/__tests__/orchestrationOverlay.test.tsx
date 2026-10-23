import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { act } from 'react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

const inputHarness = vi.hoisted(() => ({
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void)
}))

// Preserve actual Ink layout but drive keys manually without a raw-mode terminal.
vi.mock('@hermes/ink', async importOriginal => {
  const ink = await importOriginal<typeof import('@hermes/ink')>()
  return {
    ...ink,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHarness.handler = handler
    }
  }
})

import { OrchestrationOverlay } from '../components/orchestrationOverlay.js'
import type { OrchestrationStatus } from '../domain/orchestration.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const local = {
  id: 'custom@abc::qwen',
  provider: 'custom',
  model: 'qwen',
  availability: 'reachable',
  source: 'ollama-live',
  endpoint: 'http://127.0.0.1:11434/v1'
}
const proxy = {
  id: 'custom@def::proxy',
  provider: 'custom',
  model: 'proxy',
  availability: 'reachable',
  source: 'openai-compatible-live',
  endpoint: 'http://localhost:4000/v1'
}

const initialStatus = (): OrchestrationStatus => ({
  models: [local, proxy],
  roles: [{ role: 'main', mode: 'auto', fallback_ids: [] }],
  profiles: [{ name: 'default' }],
  active_profile: 'default',
  policy: { local_only: true, max_input_cost_per_million: 1.5, max_output_cost_per_million: 3 }
})

const mounted: Array<() => void> = []
const environment = globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
const previousActEnvironment = environment.IS_REACT_ACT_ENVIRONMENT
beforeAll(() => {
  environment.IS_REACT_ACT_ENVIRONMENT = true
})
afterAll(() => {
  environment.IS_REACT_ACT_ENVIRONMENT = previousActEnvironment
})
afterEach(() => {
  for (const cleanup of mounted.splice(0)) cleanup()
  inputHarness.handler = undefined
})

async function mount() {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''
  Object.assign(stdout, { columns: 180, rows: 55, isTTY: false })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  let status = initialStatus()
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'orchestration.status' || method === 'orchestration.refresh') return status
    if (method === 'orchestration.policy') {
      status = { ...status, policy: params.policy as OrchestrationStatus['policy'] }
      return { policy: status.policy, active_profile: 'default' }
    }
    if (method === 'orchestration.qualify') {
      return {
        model_id: params.model_id,
        status: 'passed',
        passed: true,
        inference_tested: true,
        level: 'inference-probe',
        provenance: { source: 'ollama-live', api_key: 'private-must-not-render' }
      }
    }
    throw new Error(`unexpected method ${method}`)
  })
  const element = React.createElement(OrchestrationOverlay, {
    gw: { request } as unknown as GatewayClient,
    maxWidth: 175,
    onClose: () => {},
    sessionId: 'sess',
    t: DEFAULT_THEME
  })
  let instance!: ReturnType<typeof renderSync>
  await act(async () => {
    instance = renderSync(element, {
      patchConsole: false,
      stdout: stdout as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stderr: stderr as NodeJS.WriteStream
    })
    await Promise.resolve()
  })
  mounted.push(() => {
    act(() => {
      instance.unmount()
      instance.cleanup()
    })
  })
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    instance.rerender(element)
  })
  return {
    request,
    output: () => stripAnsi(output),
    press: async (key: string, modifiers: Record<string, boolean> = {}) => {
      await act(async () => {
        expect(inputHarness.handler).toBeTypeOf('function')
        inputHarness.handler?.(key, modifiers)
        await Promise.resolve()
        await Promise.resolve()
        instance.rerender(element)
      })
    }
  }
}

describe('/orchestrate interactive policy and local qualification', () => {
  it('never probes during browsing; requires explicit confirm and shows provenance', async () => {
    const ui = await mount()
    expect(ui.output()).toContain('local only ON')
    expect(ui.request).toHaveBeenCalledWith('orchestration.status', {})
    await ui.press('', { tab: true }) // roles → models
    expect(ui.output()).toContain('inference not verified')
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.qualify')).toBe(false)
    await ui.press('v')
    expect(ui.output()).toContain('Run a real local inference qualification')
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.qualify')).toBe(false)
    await ui.press('[') // reduce 32 → 16 tokens
    await ui.press('+') // increase 20 → 30 seconds
    await ui.press('y')
    expect(ui.request).toHaveBeenCalledWith('orchestration.qualify', {
      model_id: local.id,
      max_tokens: 16,
      timeout_seconds: 30
    })
    expect(ui.output()).toContain('inference executed')
    expect(ui.output()).toContain('source: ollama-live')
    expect(ui.output()).not.toContain('private-must-not-render')
  })

  it('rejects a reachable localhost proxy and cancellation never starts inference', async () => {
    const ui = await mount()
    await ui.press('', { tab: true })
    await ui.press('v')
    await ui.press('n')
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.qualify')).toBe(false)
    await ui.press('', { downArrow: true })
    await ui.press('v')
    expect(ui.output()).toContain('Qualification disabled')
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.qualify')).toBe(false)
  })

  it('requires confirmation before replacing active profile policy, preserving its caps', async () => {
    const ui = await mount()
    await ui.press('', { tab: true })
    await ui.press('', { tab: true })
    await ui.press('', { tab: true }) // policy tab
    expect(ui.output()).toContain('Active profile routing guardrails')
    await ui.press(' ') // propose local_only off
    expect(ui.output()).toContain('Confirm active profile policy change?')
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.policy')).toBe(false)
    await ui.press('n') // reject
    expect(ui.request.mock.calls.some(([name]) => name === 'orchestration.policy')).toBe(false)
    await ui.press(' ') // propose again
    await ui.press('y') // apply
    expect(ui.request).toHaveBeenCalledWith('orchestration.policy', {
      action: 'set',
      policy: { local_only: false, max_input_cost_per_million: 1.5, max_output_cost_per_million: 3 }
    })

    await ui.press('', { downArrow: true }) // input cap
    await ui.press('', { return: true })
    expect(ui.output()).toContain('USD per 1M input tokens')
    for (let i = 0; i < 3; i++) await ui.press('', { backspace: true })
    await ui.press('0') // zero is a real cap, distinct from clearing it
    await ui.press('', { return: true })
    expect(ui.request.mock.calls.filter(([name]) => name === 'orchestration.policy')).toHaveLength(1)
    await ui.press('y')
    expect(ui.request).toHaveBeenCalledWith('orchestration.policy', {
      action: 'set',
      policy: { local_only: false, max_input_cost_per_million: 0, max_output_cost_per_million: 3 }
    })
  })
})
