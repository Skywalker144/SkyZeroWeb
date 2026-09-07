const SESSION_GAME_KEY = 'skz_scotland_active_game_id';

export function gameCreationRequest(selection, seed) {
  return {
    pursuer_strategy: selection.pursuer_strategy,
    evader_strategy: selection.evader_strategy,
    pursuer_action_mode: selection.pursuer_action_mode,
    az_simulations: selection.az_simulations,
    ...(selection.ruleset ? {ruleset: selection.ruleset} : {}),
    seed,
  };
}

function browserSessionStorage() {
  try {
    return globalThis.sessionStorage;
  } catch (_) {
    return null;
  }
}

export class GameSession {
  constructor() {
    this.gameId = null;
    this.gameMetadata = null;
    this.state = null;
    this.strategyData = null;
    this.catalogReady = false;
  }

  rememberGameId(value, storage = browserSessionStorage()) {
    this.gameId = value || null;
    if (!storage) return;
    try {
      if (this.gameId) storage.setItem(SESSION_GAME_KEY, this.gameId);
      else storage.removeItem(SESSION_GAME_KEY);
    } catch (_) {}
  }

  rememberedGameId(storage = browserSessionStorage()) {
    if (!storage) return null;
    try {
      return storage.getItem(SESSION_GAME_KEY);
    } catch (_) {
      return null;
    }
  }
}

export const gameSession = new GameSession();
