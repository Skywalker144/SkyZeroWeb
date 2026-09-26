export const TICKETS = ['taxi', 'bus', 'underground', 'black'];
const clone = value => structuredClone(value);
const inventory = values => Object.fromEntries(values.map((v, i) => [TICKETS[i], v]));

function randomStream(seed) {
  let state = ((seed >>> 0) ^ Math.imul(Math.floor(seed / 4294967296), 0x9e3779b9)) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export class ScotlandGame {
  constructor(board, config, seed, opening = null) {
    if (!Number.isSafeInteger(seed) || seed < 0) throw new Error('seed 必须是非负安全整数');
    this.board = board;
    this.config = config;
    this.seed = seed;
    this.undoStack = [];
    const rng = randomStream(seed);
    const starts = [...board.start_nodes];
    for (let i = starts.length - 1; i > 0; --i) {
      const j = Math.floor(rng() * (i + 1));
      [starts[i], starts[j]] = [starts[j], starts[i]];
    }
    const evader = opening?.evader ?? starts[0];
    const pursuers = opening?.pursuers ?? starts.slice(1, config.num_pursuers + 1);
    if (pursuers.length !== config.num_pursuers || new Set([evader, ...pursuers]).size !== pursuers.length + 1
        || [evader, ...pursuers].some(n => !board.actions[n])) throw new Error('无效起点');
    this.state = {
      evader_mode: 'human', evader_strategy_id: 'human', evader_policy_id: null,
      evader_action_mode: 'human', pursuer_action_mode: 'deterministic_argmax',
      phase: 'evader', round: 1, max_rounds: config.max_rounds,
      reveal_rounds: config.reveal_rounds, evader_position: evader,
      pursuer_positions: [...pursuers],
      possible_locations: board.start_nodes.filter(n => !pursuers.includes(n)),
      evader_tickets: inventory(config.resolved_ruleset.evader_initial_tickets),
      pursuer_tickets: pursuers.map(() => inventory(config.resolved_ruleset.pursuer_initial_tickets)),
      evader_history: [], exposed_positions: [], round_history: [],
      last_evader_move: null, last_pursuer_moves: [], last_pursuer_order: [],
      terminal: false, winner: null, terminal_reason: null, last_az_search: {},
    };
    if (!this.hasPursuerMove()) this.finish('pursuers_stuck', 'evader');
    else if (!this.legalEvader().length) this.finish('evader_stuck', 'pursuers');
  }
  legalEvader() {
    const s = this.state;
    return this.board.evader_actions[s.evader_position].filter(m => s.evader_tickets[m.ticket] > 0 && !s.pursuer_positions.includes(m.destination));
  }
  candidateRows() {
    return this.state.pursuer_positions.map((n, i) => this.board.actions[n].filter(m => m.ticket !== 'black' && this.state.pursuer_tickets[i][m.ticket] > 0));
  }
  hasPursuerMove() {
    return this.candidateRows().some(row => row.some(m => !this.state.pursuer_positions.includes(m.destination)));
  }
  orderMask(rows) {
    const positions = this.state.pursuer_positions;
    const destinations = rows.map(row => new Set(row.map(m => m.destination)));
    const interactions = [];
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        if (destinations[i].has(positions[j]) || destinations[j].has(positions[i])
            || [...destinations[i]].some(n => destinations[j].has(n))) interactions.push([i,j]);
      }
    }
    const rank = positions.map(n => [...positions].sort((a,b) => a-b).indexOf(n));
    const representatives = new Map();
    this.board.pursuer_orders.forEach((order,id) => {
      const signature = interactions.map(([i,j]) => Number(order.indexOf(i) < order.indexOf(j))).join('');
      const ranked = order.map(i => rank[i]).join('');
      const previous = representatives.get(signature);
      if (!previous || ranked < previous.ranked) representatives.set(signature, {ranked,id});
    });
    const selected = new Set([...representatives.values()].map(v => v.id));
    return this.board.pursuer_orders.map((_,id) => Number(selected.has(id)));
  }
  snapshot() {
    return clone({...this.state, possible_location_count: this.state.possible_locations.length,
      legal_evader_moves: this.state.phase === 'evader' && !this.state.terminal ? this.legalEvader() : [],
      pursuer_moves: this.candidateRows(), undo_available: this.undoStack.length > 0});
  }
  finish(reason, winner) {
    Object.assign(this.state, {terminal: true, terminal_reason: reason, winner, phase: 'terminal'});
  }
  evaderMove(move) {
    const s = this.state;
    if (s.terminal || s.phase !== 'evader') throw new Error('当前不是 Evader 回合');
    const legal = this.legalEvader().find(m => m.destination === move.destination && m.ticket === move.ticket);
    if (!legal) throw new Error('非法移动');
    this.undoStack.push(clone(s));
    this.pending = {round: s.round, evader: {origin: s.evader_position, ...legal}, origins: [...s.pursuer_positions]};
    s.evader_position = legal.destination;
    s.evader_tickets[legal.ticket] -= 1;
    s.evader_history.push({round: s.round, ...legal});
    s.last_evader_move = clone(legal);
    s.last_pursuer_moves = [];
    s.last_pursuer_order = [];
    if (s.reveal_rounds.includes(s.round)) {
      s.possible_locations = [legal.destination];
      s.exposed_positions.push({round: s.round, ...legal});
    } else {
      s.possible_locations = [...new Set(s.possible_locations
        .filter(n => !s.pursuer_positions.includes(n))
        .flatMap(n => this.board.evader_actions[n])
        .filter(m => m.ticket === legal.ticket && !s.pursuer_positions.includes(m.destination))
        .map(m => m.destination))].sort((a,b) => a-b);
    }
    s.phase = 'pursuer';
  }
  pursuerMove(actions, order) {
    const s = this.state;
    if (s.terminal || s.phase !== 'pursuer' || !this.pending) throw new Error('当前不是 Pursuer 回合');
    const count = s.pursuer_positions.length;
    if (actions.length !== count || order.length !== count || new Set(order).size !== count
        || order.some(i => !Number.isInteger(i) || i < 0 || i >= count)) throw new Error('无效网络动作顺序');
    const rows = this.candidateRows();
    const positions = [...s.pursuer_positions];
    const moves = Array(count).fill(null);
    for (const id of order) {
      const occupied = positions.filter((_,i) => i !== id);
      const valid = rows[id].map((m,i) => occupied.includes(m.destination) ? -1 : i).filter(i => i >= 0);
      const a = actions[id];
      if (!valid.length) {
        if (a !== -1) throw new Error('该 Pursuer 必须停留');
      } else {
        if (!valid.includes(a)) throw new Error('网络选择了被占用或无效的目的地');
        moves[id] = rows[id][a];
        positions[id] = moves[id].destination;
      }
    }
    s.pursuer_positions = positions;
    moves.forEach((m,i) => { if (m) s.pursuer_tickets[i][m.ticket] -= 1; });
    s.last_pursuer_order = [...order];
    s.last_pursuer_moves = moves.map((m,i) => m || {destination:positions[i],ticket:'forced_pass'});
    s.round_history.push({round:this.pending.round, evader:this.pending.evader,
      pursuer_order: order.map(i => i+1),
      pursuers: s.last_pursuer_moves.map((m,i) => ({agent:i+1,sequence:order.indexOf(i)+1,origin:this.pending.origins[i],...m}))});
    this.pending = null;
    if (positions.includes(s.evader_position)) this.finish('capture', 'pursuers');
    else {
      s.possible_locations = s.possible_locations.filter(n => !positions.includes(n));
      if (!this.hasPursuerMove()) this.finish('pursuers_stuck', 'evader');
      else if (s.round >= s.max_rounds) this.finish('max_rounds', 'evader');
      else {
        s.round += 1;
        s.phase = 'evader';
        if (!this.legalEvader().length) this.finish('evader_stuck', 'pursuers');
      }
    }
  }
  undo() {
    if (!this.undoStack.length) throw new Error('没有可撤回的回合');
    this.state = this.undoStack.pop();
    this.pending = null;
  }
  publicInputs() {
    const s = this.state, board = this.board, config = this.config;
    const pad = (a, n) => [...a, ...Array(Math.max(0, n - a.length)).fill(0)];
    const rows = this.candidateRows(), width = board.max_candidates;
    return {
      history_tickets: pad(s.evader_history.map(m => TICKETS.indexOf(m.ticket)+1), config.max_rounds),
      reveal_positions: pad(s.exposed_positions.map(m => m.destination), config.reveal_rounds.length),
      pursuer_positions: [...s.pursuer_positions],
      pursuer_tickets: s.pursuer_tickets.flatMap(t => TICKETS.slice(0,3).map((k,i) => t[k]/Math.max(1,config.resolved_ruleset.pursuer_initial_tickets[i]))),
      evader_tickets: TICKETS.map((k,i) => s.evader_tickets[k]/Math.max(1,config.resolved_ruleset.evader_initial_tickets[i])),
      possible_mask: Array.from({length:board.node_vocab_size}, (_,n) => Number(s.possible_locations.includes(n))),
      round_scalar: [s.round/config.max_rounds], search_context: [0,0],
      candidate_destinations: rows.flatMap(row => pad(row.map(m => m.destination),width)),
      candidate_tickets: rows.flatMap(row => pad(row.map(m => TICKETS.indexOf(m.ticket)+1),width)),
      candidate_mask: rows.flatMap(row => pad(row.map(() => 1),width)),
      order_mask: this.orderMask(rows),
    };
  }
  serialize() { return {version:2,environment_identity:this.config.environment_identity,seed:this.seed,state:this.state,pending:this.pending,undoStack:this.undoStack}; }
  static restore(board, config, data) {
    if (data.version !== 2 || data.environment_identity !== config.environment_identity) throw new Error('保存的对局版本不兼容');
    const game = new ScotlandGame(board, config, data.seed);
    game.state = data.state;
    game.pending = data.pending;
    game.undoStack = data.undoStack;
    return game;
  }
}
