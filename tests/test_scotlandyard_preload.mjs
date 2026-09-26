import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import vm from 'node:vm';

const base = new URL('../public/scotlandyard/', import.meta.url);
const {pursuer: models} = JSON.parse(await readFile(new URL('models.json', base)));
const source = await readFile(new URL('inference-worker.js', base), 'utf8');
const files = new Map(await Promise.all(models.map(async model => [model.url, await readFile(new URL(model.url, base))])));
const entries = models.map(model => ({model, url:model.url}));

function harness(intercept = () => null) {
  const requests = [], messages = [], sessions = [];
  const context = vm.createContext({
    self:{location:{search:''}, postMessage:message => messages.push(message)},
    importScripts() {},
    SkyZeroOnnxRuntime:{loadOnnxRuntime() {}},
    crypto:webcrypto, Uint8Array, Map,
    fetch:async url => {
      requests.push(url);
      return intercept(url) || new Response(files.get(url));
    },
    ort:{InferenceSession:{create:async bytes => {
      const session = {bytes, released:false, async release() { this.released = true; }};
      sessions.push(session);
      return session;
    }}},
  });
  vm.runInContext(source, context);
  return {requests, sessions, async call(message) {
    messages.length = 0;
    await context.self.onmessage({data:{id:1,...message}});
    const response = messages.find(message => !message.progress);
    assert.ok(response, 'worker must reply');
    return {response, progress:messages.filter(message => message.progress).map(message => message.progress)};
  }};
}

test('all weights are verified up front and every opponent reuses them without another download', async () => {
  const worker = harness();
  const {response, progress} = await worker.call({kind:'prepare', models:entries});
  assert.equal(response.result, true);
  assert.equal(worker.requests.length, models.length);
  const total = models.reduce((sum, model) => sum + model.bytes, 0);
  assert.ok(progress.length > models.length);
  assert.ok(progress.every(value => value.total === total));
  assert.equal(progress.at(-1).loaded, total);
  assert.ok(progress.every((value, i) => !i || value.loaded >= progress[i-1].loaded));
  for (const model of [...models, ...models]) {
    assert.equal((await worker.call({kind:'load', model})).response.result, true);
    assert.deepEqual(new Uint8Array(worker.sessions.at(-1).bytes), new Uint8Array(files.get(model.url)));
    assert.equal(worker.sessions.filter(session => !session.released).length, 1);
  }
  assert.equal((await worker.call({kind:'prepare', models:entries})).response.result, true);
  assert.equal(worker.requests.length, models.length);
});

test('failed preparation retries only missing weights', async () => {
  let fail = true;
  const worker = harness(url => fail && url === models[2].url ? new Response('', {status:503}) : null);
  assert.match((await worker.call({kind:'prepare', models:entries})).response.error, /503/);
  assert.equal(worker.requests.length, 3);
  assert.equal(worker.sessions.length, 0);
  fail = false;
  const {response, progress} = await worker.call({kind:'prepare', models:entries});
  assert.equal(response.result, true);
  assert.equal(worker.requests.length, models.length + 1);
  assert.equal(progress[0].loaded, models[0].bytes + models[1].bytes);
});

for (const corruption of ['size', 'hash']) {
  test(`rejects ${corruption} mismatch and never caches corrupt weights`, async () => {
    let corrupt = true;
    const original = files.get(models[0].url);
    const damaged = corruption === 'size' ? original.subarray(1) : Uint8Array.from(original, (value, i) => i ? value : value ^ 1);
    const worker = harness(() => corrupt ? new Response(damaged) : null);
    assert.match((await worker.call({kind:'prepare', models:entries.slice(0,1)})).response.error, /大小不符|校验失败/);
    assert.match((await worker.call({kind:'load', model:models[0]})).response.error, /准备|下载/);
    assert.equal(worker.sessions.length, 0);
    corrupt = false;
    assert.equal((await worker.call({kind:'prepare', models:entries.slice(0,1)})).response.result, true);
    assert.equal(worker.requests.length, 2);
    assert.equal((await worker.call({kind:'load', model:models[0]})).response.result, true);
  });
}
