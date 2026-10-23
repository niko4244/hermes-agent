import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'

export const orchestrationCommands: SlashCommand[] = [
  {
    aliases: ['models'],
    help: 'browse available models and assign agents, fallbacks, and profiles',
    name: 'orchestrate',
    run: () => patchOverlayState({ orchestration: true, modelPicker: false })
  }
]
