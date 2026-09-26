import {ScotlandGame} from './engine.js?v=v016-pilot-1';

export class PilotChallenge {
  constructor(board, config, seed) {
    this.game = new ScotlandGame(board, config, seed);
    this.undoCount = 0;
    this.result = 'in_progress';
    this.settleVictory();
  }
  get state() { return this.game.state; }
  get gameLength() { return this.state.evader_history.length; }
  get settled() { return this.result !== 'in_progress'; }
  ensureOpen() {
    if (this.settled) throw new Error('本局已经结算，请开始新对局。');
  }
  settleVictory() {
    if (this.state.terminal && this.state.winner === 'evader') this.result = 'win';
  }
  evaderMove(move) {
    this.ensureOpen();
    this.game.evaderMove(move);
    this.settleVictory();
  }
  pursuerMove(actions, order) {
    this.ensureOpen();
    this.game.pursuerMove(actions, order);
    this.settleVictory();
  }
  undo() {
    this.ensureOpen();
    this.game.undo();
    this.undoCount += 1;
  }
  rewind() {
    this.ensureOpen();
    if (!this.game.undoStack.length) throw new Error('当前已经在开局，没有可悔棋的回合。');
    while (this.game.undoStack.length) this.undo();
  }
  giveUp() {
    this.ensureOpen();
    this.result = 'give_up';
  }
  publicInputs() { return this.game.publicInputs(); }
  snapshot() {
    const state = this.game.snapshot();
    if (this.settled) {
      state.undo_available = false;
      state.legal_evader_moves = [];
    }
    if (this.result === 'give_up') Object.assign(state, {
      terminal:true,phase:'terminal',winner:'pursuers',terminal_reason:'give_up',
    });
    return {...state,challenge:{result:this.result,undo_num:this.undoCount,game_len:this.gameLength}};
  }
  serialize() {
    return {version:1,game:this.game.serialize(),undo_num:this.undoCount,result:this.result};
  }
  static restore(board, config, data) {
    if (data.version !== 1 || !Number.isSafeInteger(data.undo_num) || data.undo_num < 0
        || !['in_progress','win','give_up'].includes(data.result)) throw new Error('挑战记录格式不兼容');
    const challenge = new PilotChallenge(board, config, data.game.seed);
    challenge.game = ScotlandGame.restore(board, config, data.game);
    challenge.undoCount = data.undo_num;
    challenge.result = data.result;
    if ((data.result === 'win') !== (challenge.state.terminal && challenge.state.winner === 'evader'))
      throw new Error('挑战结果与棋局状态不一致');
    return challenge;
  }
}
