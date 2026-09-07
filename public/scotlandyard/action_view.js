export function legalActionRows(moves, ticketOrder) {
  const ticketRank = new Map(ticketOrder.map((ticket, index) => [ticket, index]));
  const grouped = new Map();
  for (const move of moves || []) {
    const destination = Number(move.destination);
    if (!grouped.has(destination)) grouped.set(destination, []);
    grouped.get(destination).push(move);
  }
  return [...grouped.entries()]
    .sort(([left], [right]) => left - right)
    .map(([destination, destinationMoves]) => ({
      destination,
      moves: [...destinationMoves].sort((left, right) => (
        (ticketRank.get(left.ticket) ?? ticketOrder.length)
        - (ticketRank.get(right.ticket) ?? ticketOrder.length)
      )),
    }));
}

export function ticketCountPresentation(value, initialTickets) {
  const effectivelyUnlimited = Number(initialTickets) >= 999_999;
  return effectivelyUnlimited || Number(value) >= 999_999 ? '∞' : String(value);
}

export function pursuerOrderPresentation(order) {
  const identities = (order || []).map(value => Number(value) + 1);
  if (!identities.length) {
    return {
      display: '等待行动',
      label: 'Pursuer 尚未行动',
    };
  }
  return {
    display: identities.join('›'),
    label: `Pursuer 行动顺序：${identities.map(identity => `P${identity}`).join('、')}`,
  };
}

export function undoAvailable(state) {
  return Boolean(state?.evader_mode === 'human' && state.undo_available);
}
