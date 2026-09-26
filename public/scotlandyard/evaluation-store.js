export const EVALUATION_STUDY = 'v016-belief-blind-20-v1';
export const EVALUATION_STORAGE_KEY = 'skz_scotland_v016_blind_20_v1';
export const EVALUATION_QUOTAS = Object.freeze({2:2,8:5,16:7,32:4,118:2});
const totalGames = Object.values(EVALUATION_QUOTAS).reduce((a,b) => a+b,0);
const copy = value => structuredClone(value);
export const modelKey = model => `${model.id}:${model.sha256}:${model.game.environment_identity}`;
const settled = game => game.result === 'win' || game.result === 'give_up';

export class EvaluationStore {
  constructor(storage) {
    this.storage = storage;
    this.error = null;
    this.blocked = false;
    this.savedRaw = null;
    this.data = {format_version:3,study:EVALUATION_STUDY,participant_id:crypto.randomUUID(),models:{},batches:[],active:null};
    try {
      this.savedRaw = storage?.getItem(EVALUATION_STORAGE_KEY) ?? null;
      if (this.savedRaw) {
        const saved = JSON.parse(this.savedRaw);
        if (saved.format_version !== 3 || saved.study !== EVALUATION_STUDY || !Array.isArray(saved.batches)
            || !saved.models || typeof saved.participant_id !== 'string'
            || saved.batches.some(b => !Array.isArray(b.games) || b.games.length !== totalGames)) throw new Error('格式不兼容');
        this.data = saved;
      }
    } catch (_) {
      this.blocked = true;
      this.error = '无法读取已保存的测评。请保留浏览器数据并联系维护者。';
    }
  }
  assertWritable() {
    if (this.blocked) throw new Error(this.error);
    if (this.storage && this.storage.getItem(EVALUATION_STORAGE_KEY) !== this.savedRaw)
      throw new Error('另一个页面已更新测评，请刷新当前页面后继续。');
  }
  persist() {
    try {
      if (!this.storage) throw new Error('storage unavailable');
      const raw = JSON.stringify(this.data);
      this.storage.setItem(EVALUATION_STORAGE_KEY,raw);
      this.savedRaw = raw;
      this.error = null;
    } catch (_) {
      this.error = '进度与成绩未能保存，请现在导出 JSON，避免关闭页面后丢失。';
    }
  }
  ensureBatch(models) {
    this.assertWritable();
    if (this.data.batches.length && !this.progress().complete) return;
    const identities = {}, games = [];
    for (const [checkpoint,count] of Object.entries(EVALUATION_QUOTAS)) {
      const matches = models.filter(m => m.checkpoint === Number(checkpoint));
      if (matches.length !== 1) throw new Error('测评资源不完整，请联系维护者。');
      const model = matches[0], key = modelKey(model);
      identities[key] = {checkpoint:model.checkpoint,id:model.id,onnx_sha256:model.sha256,
        source_sha256:model.source_sha256,source_version:model.source_version,run_id:model.run_id,
        environment_identity:model.game.environment_identity,decision_protocol:copy(model.decision_protocol)};
      for (let i=0;i<count;i++) games.push({game_id:crypto.randomUUID(),model_id:key,
        seed:crypto.getRandomValues(new Uint32Array(1))[0],result:'not_started',game_len:0,undo_num:0});
    }
    for (let i=games.length-1;i>0;i--) {
      const j = Math.floor(crypto.getRandomValues(new Uint32Array(1))[0]/4294967296*(i+1));
      [games[i],games[j]] = [games[j],games[i]];
    }
    Object.assign(this.data.models,identities);
    this.data.batches.push({batch_id:crypto.randomUUID(),batch_number:this.data.batches.length+1,
      created_at:new Date().toISOString(),completed_at:null,games});
    this.data.active = null;
    this.persist();
  }
  nextSlot() {
    const batch = this.data.batches.at(-1);
    const index = batch?.games.findIndex(g => !settled(g)) ?? -1;
    return index < 0 ? null : copy({...batch.games[index],batch_id:batch.batch_id,game_number:index+1});
  }
  save(active) {
    this.assertWritable();
    const batch = this.data.batches.at(-1);
    const index = batch?.games.findIndex(g => g.game_id === active.id) ?? -1;
    const slot = batch?.games[index];
    if (!slot || slot.model_id !== modelKey(active.model) || slot.seed !== active.game.game.seed)
      throw new Error('对局与测评安排不一致');
    const updated = {...slot,result:active.game.result,game_len:active.game.gameLength,undo_num:active.game.undoCount};
    if (settled(slot)) {
      if (JSON.stringify(slot) !== JSON.stringify(updated)) throw new Error('已结算的挑战不能更改');
      return;
    }
    if (this.nextSlot()?.game_id !== active.id) throw new Error('请先完成当前对局');
    batch.games[index] = updated;
    this.data.active = {id:active.id,model_id:slot.model_id,game:active.game.serialize()};
    if (batch.games.every(settled)) batch.completed_at = new Date().toISOString();
    this.persist();
  }
  active() { return copy(this.data.active); }
  progress() {
    const batch = this.data.batches.at(-1);
    const games = batch?.games || [];
    const completed = games.filter(settled).length;
    const current = games.findIndex(g => g.game_id === this.data.active?.id);
    return {batch_id:batch?.batch_id ?? null,batch_number:batch?.batch_number ?? 0,total:totalGames,
      completed,wins:games.filter(g => g.result === 'win').length,complete:completed === totalGames,
      game_number:current < 0 ? completed+1 : current+1,active_game_id:this.data.active?.id ?? null,
      can_next:current >= 0 && settled(games[current]) && completed < totalGames,
      num_batches:this.data.batches.length,num_completed_batches:this.data.batches.filter(b => b.completed_at).length};
  }
  summary(games = this.data.batches.flatMap(b => b.games)) {
    return Object.entries(this.data.models).map(([key,model]) => {
      const attempts = games.filter(g => g.model_id === key && g.result !== 'not_started');
      const completed = attempts.filter(settled), wins = completed.filter(g => g.result === 'win');
      const length = completed.reduce((n,g) => n+g.game_len,0), undos = completed.reduce((n,g) => n+g.undo_num,0);
      return {model_id:key,checkpoint:model.checkpoint,num_games:completed.length,num_wins:wins.length,
        num_giveups:completed.length-wins.length,win_rate:completed.length ? wins.length/completed.length : null,
        avg_gamelen:completed.length ? length/completed.length : null,
        avg_undo_num:completed.length ? undos/completed.length : null,
        avg_undo_num_wins:wins.length ? wins.reduce((n,g) => n+g.undo_num,0)/wins.length : null,
        total_gamelen:length,total_undo_num:undos,num_in_progress:attempts.length-completed.length};
    });
  }
  export() {
    const {format_version,study,participant_id,models} = this.data;
    return copy({format_version,study,participant_id,exported_at:new Date().toISOString(),
      quotas:EVALUATION_QUOTAS,models,summary:this.summary(),batches:this.data.batches.map(batch => ({
        batch_id:batch.batch_id,batch_number:batch.batch_number,created_at:batch.created_at,
        completed_at:batch.completed_at,status:batch.completed_at ? 'completed' : 'in_progress',
        target_games:totalGames,num_games:batch.games.filter(settled).length,summary:this.summary(batch.games),
        games:batch.games.flatMap((g,i) => g.result === 'not_started' ? [] : [{game_id:g.game_id,
          game_number:i+1,model_id:g.model_id,checkpoint:models[g.model_id].checkpoint,
          result:g.result,game_len:g.game_len,undo_num:g.undo_num}]),
      }))});
  }
}
