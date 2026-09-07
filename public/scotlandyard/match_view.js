function controllerName(source, policyId = '') {
  const kind = source?.kind;
  if (kind === 'onnx') {
    const name = String(source.name || policyId);
    if (name.startsWith('AZ')) return 'AZ';
    if (name.startsWith('PPO')) return 'PPO';
    return name.replace('（默认）', '');
  }
  if (kind === 'human') return '玩家';
  if (kind === 'random') return 'Random';
  if (kind === 'ppo') return 'PPO';
  if (kind === 'psro') return 'PSRO';
  if (kind === 'az_selfplay') return 'AZ Selfplay';
  if (kind === 'az' || kind === 'az_model') return 'AZ';
  if (kind === 'heuristic') {
    const name = String(source.name || policyId || 'Heuristic');
    return name.charAt(0).toLocaleUpperCase() + name.slice(1);
  }
  return policyId || kind || '未知';
}

export function matchPresentation(state, metadata) {
  return {
    pursuer: `${controllerName(metadata?.pursuer, state?.pursuer_policy_id)} Pursuer`,
    evader: `${controllerName(metadata?.evader, state?.evader_policy_id)} Evader`,
  };
}

export function winnerPresentation(state, metadata) {
  if (!state?.terminal) return null;
  const match = matchPresentation(state, metadata);
  if (state.winner === 'evader') return match.evader;
  if (state.winner === 'pursuers') return match.pursuer;
  return null;
}
