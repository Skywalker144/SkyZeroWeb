# SkyZeroWeb

Static webpage that runs SkyZero self-trained AI models in the browser via
`onnxruntime-web` — gomoku (`SkyZero_V7.19`), 2048 (`SkyZero_2048_V2`), and a
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
`ai2048.js` / `mcts2048.js` (pure-logic units; UI is verified in a browser).

### Deploy

This repo is set up for [Cloudflare Pages](https://pages.cloudflare.com/)
with no build step. Connect via git, point to repo root.

## Adding / updating a model

```bash
# In the `pytorch` conda env (has torch/onnx/onnxscript installed)
conda run -n pytorch python tools/export_v719_onnx.py \
    --ckpt ../SkyZero/SkyZero_V7.19/data/web/models/b9c96tflrs_iter1189.pt \
    --out models/level6.onnx

# Then edit models/manifest.json — add or update order / label / file / params
# git add + commit + push → Cloudflare auto-deploys
```

The six tiers are V7.19 `b9c96tflrs` checkpoints at iter
206/300/419/654/1000/1189. They are ordered by training iteration; the UI does
not present the old V7.1 ELO numbers as if they were comparable. The exporter
checks rebuilt-PyTorch ↔ TorchScript and ONNX Runtime parity before succeeding.

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
- `mcts.js` — single-thread V7.19 PUCT, weighted backup, SVB and LCB selection
- `ai2048.js` — 2048 afterstate logic + 1-ply value-net expectimax
- `gomoku.js` — V7.19 Renju/Standard/Freestyle rules and 5+7 input protocol
- `tools/export_v719_onnx.py` — V7.19 TorchScript → parity-checked ONNX
- `tools/export_onnx_2048.py` — SkyZero_2048 `.pt` → `models/2048.onnx`
- `models/manifest.json` — gomoku six-tier V7.19 catalog

## Gomoku search loop

The gomoku engine alternates between two modes, **reusing the search tree across
every move** (`worker.js applyMove` re-roots the tree at the child for the move
just played). `main.js triggerAISearch()` picks the path via `isPonderTurn()`:

- **Ponder** (your turn in play mode, or any move in analysis mode): fixed
  `ANALYSIS_CHUNK = 128`-sim PUCT chunks, re-fired (reusing the tree) after each
  result until cumulative root visits reach `ANALYSIS_CAP_MIN = 2000`, then it
  idles. Runs quietly on your turn but keeps the candidate list / win-rate /
  heatmaps live; placing a stone aborts the in-flight chunk via `searchId`.
- **Move-search** (the AI's own turn, play mode): a single anytime-PUCT search
  that runs for `thinkMs` (toolbar "thinking time", default 3000ms) **or** until
  cumulative root visits hit `SEARCH_VISIT_CAP` (`worker.js`, = 2000, kept equal
  to the ponder cap) — whichever comes first — then applies V7.19 retrospective
  root weighting and LCB selection.
  `thinkMs` only governs the AI's own move; it does not deepen the your-turn
  ponder (that is always the 128-chunk → 2000 cap, independent of `thinkMs`).

One full turn: page `ready` → `newGame` → ponder your turn; you move → `move`
(tree reuse) → move-search the AI's reply → AI moves → `move` (tree reuse) →
ponder your turn again. Both caps count **cumulative** root visits across tree
reuse, so in the midgame/endgame the search often tops out before spending the
full time / chunk budget. The two caps are deliberately kept equal — change one,
change the other.

## V7.19 browser search contract

The gomoku page uses the active V7.19 play package: 5 spatial + 7 global
features on a fixed 15x15 network canvas (runtime boards 11x11 through 15x15),
main/opponent/optimistic policy heads, uncertainty-weighted recompute backup,
value-weighted child aggregation, subtree value bias, variance-scaled PUCT,
root/non-root FPU, stochastic D4 inference, empty-board center restriction,
root symmetry pruning, tree reuse, retrospective play-selection weights and
LCB move selection. Renju forbidden black moves remain playable white-win
terminals. Standard is exact-five for both colors; Freestyle is five-or-more.
The analysis drawer shows current, optimistic, opponent, and LCB-adjusted
play-selection heatmaps by default; raw visits remain an expandable diagnostic.
The two future-position heatmaps are intentionally not displayed.

Browser adaptation: inference and tree descent are deliberately single-threaded
and one leaf at a time. Each cache miss samples one D4 transform rather than
forming an 8-way ensemble. RNNM (`ROOT_NONNEIGHBOUR_MASK_RADIUS`) is explicitly
not implemented; empty-board center-only and symmetry pruning are independent
root restrictions and remain enabled. Training-only `value_td` is omitted from
the browser ONNX graph.
