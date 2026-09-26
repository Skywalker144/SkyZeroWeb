export function buildTimelineItems(events, ticketLabels) {
  const ticketLabel = ticket => ticketLabels[ticket] || ticket;
  return (events || []).map(event => {
    const evaderText = `E ${ticketLabel(event.evader.ticket)} ${event.evader.origin}→${event.evader.destination}`;
    const pursuersText = [...(event.pursuers || [])]
      .sort((left, right) => (left.sequence || 999) - (right.sequence || 999))
      .map(move => `P${move.agent} ${ticketLabel(move.ticket)} ${move.origin}→${move.destination}`)
      .join(' · ');
    return {
      round: event.round,
      evaderText,
      pursuersText: pursuersText || '本轮未执行 Pursuer 行动',
    };
  });
}

export function buildTurnTrack(state, completedStepsOverride = null) {
  const maxRounds = Math.max(0, Number(state?.max_rounds) || 0);
  const totalSteps = maxRounds * 2;
  const observedCompletedSteps = Math.min(
    totalSteps,
    (state?.evader_history?.length || 0) + (state?.round_history?.length || 0),
  );
  const completedSteps = completedStepsOverride == null
    ? observedCompletedSteps
    : Math.min(observedCompletedSteps, Math.max(0, Number(completedStepsOverride) || 0));
  const terminal = Boolean(state?.terminal && completedSteps === observedCompletedSteps);
  const currentStep = totalSteps === 0
    ? 0
    : Math.min(totalSteps, Math.max(1, terminal ? completedSteps : completedSteps + 1));
  const scheduledRounds = new Set(state?.reveal_rounds || []);
  const exposedByRound = new Map(
    (state?.exposed_positions || []).map(item => [item.round, item.destination]),
  );
  const steps = Array.from({length: totalSteps}, (_, index) => {
    const step = index + 1;
    const round = Math.ceil(step / 2);
    const role = step % 2 === 1 ? 'evader' : 'pursuer';
    const scheduled = role === 'evader' && scheduledRounds.has(round);
    const exposedAt = role === 'evader' && step <= completedSteps ? exposedByRound.get(round) : undefined;
    return {
      step,
      round,
      role,
      played: step <= completedSteps,
      current: step === currentStep,
      scheduled,
      exposedAt,
      label: `第 ${round} 轮 · ${role === 'evader' ? 'Evader' : 'Pursuer'} 行动`,
    };
  });
  return {
    totalSteps,
    completedSteps,
    currentStep,
    terminal,
    steps,
  };
}

export function revealPresentation(state) {
  const schedule = state?.reveal_rounds || [];
  const latest = [...(state?.exposed_positions || [])].sort((a,b) => b.round-a.round)[0] || null;
  const revealsNow = Boolean(state && !state.terminal && state.phase === 'evader' && schedule.includes(state.round));
  const nextRound = state?.terminal ? null : schedule.find(r => r > (state?.round || 0)
    || (r === state?.round && state.phase === 'evader')) ?? null;
  return {current:latest?.round === state?.round ? latest : null,revealsNow,nextRound};
}
