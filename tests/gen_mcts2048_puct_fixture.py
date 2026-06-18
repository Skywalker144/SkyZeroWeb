#!/usr/bin/env python3
"""Generate a cross-check fixture for the PUCT-root path of mcts2048.js against
SkyZero_2048_V2/python/mcts.py (root_algo="puct").

Same idea as gen_mcts2048_fixture.py (the gumbel fixture), but with the classic
AlphaZero root: PUCT selection at the root, most-visited move, visit-count
policy. Uses the SAME fixed synthetic evaluator so mcts2048.js reproduces the
whole search bit-for-bit:
    value(state)     = sum over cells of 2**exp (exp > 0)   [raw points]
    logits(state)[a] = apply_move(state, a).reward * 0.01   [0 if illegal]

Run from the SkyZeroWeb dir with the torch-env python (numpy):
    $PY tests/gen_mcts2048_puct_fixture.py
Writes tests/mcts2048_puct_fixture.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# The PUCT root path lives in the V2 repo. Try the known layouts (CLAUDE.md:
# 2048 comes from ../SkyZero/SkyZero_2048_V2); allow $SKYZERO_2048_V2 to override.
import os  # noqa: E402

_RL = Path(__file__).resolve().parents[2]   # .../RL
_CANDIDATES = [
    Path(os.environ["SKYZERO_2048_V2"]) / "python" if os.environ.get("SKYZERO_2048_V2") else None,
    _RL / "SkyZero" / "SkyZero_2048_V2" / "python",
    _RL / "SkyZero_2048_V2" / "python",
]
_2048_PY = next((p for p in _CANDIDATES if p and p.is_dir()), None)
if _2048_PY is None:
    raise SystemExit(f"Could not find SkyZero_2048_V2/python; tried {_CANDIDATES}")
sys.path.insert(0, str(_2048_PY))

import game as G          # noqa: E402
import mcts as M          # noqa: E402
from model_config import Config  # noqa: E402

HERE = Path(__file__).resolve().parent
NUM_STATES = 60
NUM_SIMS = 64


def make_eval():
    def _eval(states):
        b = len(states)
        logits = np.zeros((b, 4), dtype=np.float64)
        values = np.zeros(b, dtype=np.float64)
        for k, s in enumerate(states):
            s = np.asarray(s, dtype=np.int8)
            values[k] = float(sum((1 << int(e)) for e in s if e > 0))
            for a in range(4):
                _after, reward, _changed = G.apply_move(s, a)
                logits[k, a] = reward * 0.01
        return logits, values
    return _eval


def main() -> int:
    cases = json.loads((HERE / "ai2048_fixture.json").read_text())
    states = [c["state"] for c in cases if not c["terminal"]][:NUM_STATES]

    cfg = Config(num_simulations=NUM_SIMS, gamma=0.999, c_puct=1.25,
                 gumbel_c_visit=50.0, gumbel_c_scale=1.0, gumbel_noise=False,
                 root_algo="puct")
    eval_fn = make_eval()

    out = []
    for st in states:
        arr = np.asarray(st, dtype=np.int8)
        gs = M.GameSearch(arr, cfg, np.random.default_rng(0))
        M.batch_search([gs], eval_fn, cfg)
        out.append({
            "state": st,
            "best_action": int(gs.best_action()),
            "visits": [int(x) for x in gs.visit_counts()],
            "improved_policy": [float(x) for x in gs.improved_policy()],
            "root_value": float(gs.root_value()),
            "nn_policy": [float(x) for x in gs.nn_policy()],
        })

    (HERE / "mcts2048_puct_fixture.json").write_text(json.dumps(out))
    print(f"wrote {HERE/'mcts2048_puct_fixture.json'}  ({len(out)} states, "
          f"sims={NUM_SIMS}, root_algo=puct, gumbel_noise=False)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
