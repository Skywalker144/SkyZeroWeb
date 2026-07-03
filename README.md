# SkyZeroWeb

Static webpage that runs SkyZero self-trained AI models in the browser via
`onnxruntime-web` — gomoku (`SkyZero_V7.1`), 2048 (`SkyZero_2048_V2`), and a
dodge game (`DodgeSAC`). No server, no C++ engine — everything runs client-side.

## Quick start

### Local dev

```bash
cd /home/sky/RL/SkyZeroWeb
python3 -m http.server 8000
# Open http://localhost:8000
```

(`file://` won't work — Worker `importScripts` and `fetch('models/...')`
need an HTTP server.)

### Tests

```bash
export PATH=/home/sky/.nvm/versions/node/v24.16.0/bin:$PATH
npm test
```

Runs the Node 18+ builtin test runner against `gomoku.js` / `mcts.js` /
`ai2048.js` / `mcts2048.js` (pure-logic units; UI is verified manually).
Should report `tests 58 | pass 58`.

### Deploy

This repo is set up for [Cloudflare Pages](https://pages.cloudflare.com/)
with no build step. Connect via git, point to repo root.

## Adding / updating a model

```bash
# In the `pytorch` conda env (has torch/onnx/onnxscript installed)
python tools/export_onnx.py \
    --ckpt /path/to/SkyZero_V7.1/.../nets/b5c128/model_iter_NNNNNN.pt \
    --out  models/levelN.onnx \
    --num-blocks 5 --num-channels 128

# Or use a TorchScript anchor file (the script handles both formats):
python tools/export_onnx.py \
    --ckpt /path/to/SkyZero_V7.1/anchors/b5c128iterNN.pt \
    --out  models/level3.onnx \
    --num-blocks 5 --num-channels 128

# Then edit models/manifest.json — add or update the entry's elo / label / file
# git add + commit + push → Cloudflare auto-deploys
```

The 5-tier ELO catalog is hand-curated. Each tier ships one ONNX
(~3.45 MB for the current `b5c128` nets).

### Python prerequisites for export

The export script needs these packages (in addition to torch):

```bash
pip install onnx onnxscript onnxruntime
```

## Adding / updating the 2048 model

The 2048 page runs the `SkyZero_2048_V2` value net in the browser. The current
network is `b5c96`; the export script eats the traced TorchScript directly (no
`--net` needed) and bakes the value transform into the ONNX graph:

```bash
# In the `pytorch` conda env (has torch/onnx/onnxruntime installed)
python tools/export_onnx_2048.py \
    --ckpt ../SkyZero/SkyZero_2048_V2/data2048/nets/b5c96/latest.pt \
    --out models/2048.onnx --value-scale 30 --value-transform
```

`--value-scale 30` and `--value-transform` come from the net's `latest.meta.json`
(V2's value lives in h-space at scale 30); the head is rescaled to raw 2048
points inside the ONNX graph. Then bump `AI_MODEL_VERSION` in `2048.html` to
cache-bust. See `CLAUDE.md` for the authoritative, step-by-step flow.

The browser AI (`ai2048.js`) plays a **1-ply expectimax** over the value head —
`Q(a) = reward(a) + γ·E_spawn[V(next)]` — rather than the full Gumbel MCTS used
during self-play, so it's lighter but weaker than the engine's searched
strength. `tests/test_ai2048.mjs` cross-checks the JS slide/spawn/encode logic
against a fixture generated from `SkyZero_2048/python/game.py`.

## Architecture

- `index.html` — landing page (game picker → `/gomoku`, `/2048`)
- `gomoku.html` / `style.css` — five-in-a-row UI (ported from `play_web.py`)
- `2048.html` — 2048 game UI + in-page AI controls (AI 走子 / AI 托管)
- `main.js` — UI controller, canvas rendering, worker plumbing
- `worker.js` — runs `ort.InferenceSession` + MCTS in a Web Worker (gomoku)
- `worker2048.js` — runs the 2048 value net (ONNX) + expectimax off-thread
- `mcts.js` — Sequential MCTS with variance-scaled cPUCT + Gumbel halving
- `ai2048.js` — 2048 afterstate logic + 1-ply value-net expectimax
- `gomoku.js` — RENJU game logic with multi-board-size + V7.1 5-plane encoding
- `tools/export_onnx.py` — V7.1 `.pt` → `.onnx` (gomoku, drops UI-unused heads)
- `tools/export_onnx_2048.py` — SkyZero_2048 `.pt` → `models/2048.onnx`
- `models/manifest.json` — gomoku 5-tier ELO catalog

## Gomoku search loop

The gomoku engine alternates between two modes, **reusing the search tree across
every move** (`worker.js applyMove` re-roots the tree at the child for the move
just played). `main.js triggerAISearch()` picks the path via `isPonderTurn()`:

- **Ponder** (your turn in play mode, or any move in analysis mode): fixed
  `ANALYSIS_CHUNK = 96`-sim PUCT chunks, re-fired (reusing the tree) after each
  result until cumulative root visits reach `ANALYSIS_CAP_MIN = 2000`, then it
  idles. Runs quietly on your turn but keeps the candidate list / win-rate /
  heatmaps live; placing a stone aborts the in-flight chunk via `searchId`.
- **Move-search** (the AI's own turn, play mode): a single anytime-PUCT search
  that runs for `thinkMs` (toolbar "thinking time", default 3000ms) **or** until
  cumulative root visits hit `SEARCH_VISIT_CAP` (`worker.js`, = 2000, kept equal
  to the ponder cap) — whichever comes first — then plays the most-visited move.
  `thinkMs` only governs the AI's own move; it does not deepen the your-turn
  ponder (that is always the 96-chunk → 2000 cap, independent of `thinkMs`).

One full turn: page `ready` → `newGame` → ponder your turn; you move → `move`
(tree reuse) → move-search the AI's reply → AI moves → `move` (tree reuse) →
ponder your turn again. Both caps count **cumulative** root visits across tree
reuse, so in the midgame/endgame the search often tops out before spending the
full time / chunk budget. The two caps are deliberately kept equal — change one,
change the other.

## Differences from V5 `play_web.py`

These simplifications are intentional (browser constraints):

- No 8-fold symmetry ensemble (single forward pass)
- No stochastic transform
- No parallel MCTS (Worker is single-threaded)
- No root symmetry pruning toggle
- UI only exposes RENJU / FREESTYLE toggle buttons (STANDARD is trained — see
  `models/manifest.json`'s per-model `rules` — and playable via `gomoku.js`,
  but has no button in `main.js`/`gomoku.html`)
- `value_td` and intermediate heads dropped from ONNX export
