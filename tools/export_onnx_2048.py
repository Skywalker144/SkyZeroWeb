#!/usr/bin/env python3
"""Export a SkyZero_2048 Net2048 checkpoint to ONNX for browser inference.

The 2048 net (python/nets.py:Net2048) takes (B, 16, 4, 4) one-hot exponent
planes and returns (policy_logits[B,4], value[B]) where `value` is in SCALED
units (raw expected-discounted-score / value_scale). The browser AI wants raw
points so its expectimax Q = reward + gamma*V stays in one unit, so this
wrapper folds the value_scale multiply into the graph: the ONNX `value` output
is already in raw 2048 points.

Two checkpoint shapes are accepted:
  - an already-traced TorchScript module (V1 scripted_iter_*.pt, V2
    server_models/*.pt) — exported AS-IS, so no --net is needed and no V1/V2
    architecture rebuild can drift;
  - a bare/dict state_dict — rebuilt via --net <b{blocks}c{channels}>.

Usage:
    # V2 server model (TorchScript): VALUE_SCALE=30, value transform always on
    python tools/export_onnx_2048.py \\
        --ckpt ../SkyZero/SkyZero_2048_V2/server_models/model_iter406.pt \\
        --out models/2048.onnx --value-scale 30 --value-transform

    # bare state_dict: --net parses to blocks/channels, other Config fields default
    python tools/export_onnx_2048.py \\
        --ckpt .../nets/b3c64/model_latest.pt --net b3c64 --out models/2048.onnx

Mirrors tools/export_onnx.py's gotchas: opset 18 (Mish), dynamo=False (so the
weights inline into a single self-contained .onnx rather than a sibling
.onnx.data file that onnxruntime-web can't follow), dynamic batch axis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import onnx


# 2048 is encoded as 16 one-hot exponent planes (game.py:encode_state); a traced
# net takes (B, 16, 4, 4) and the browser builds the same.
NUM_PLANES_2048 = 16


def _import_net_builders():
    """Lazily import the SkyZero_2048 net definition. Only needed to rebuild a
    bare/dict state_dict; an already-traced TorchScript checkpoint (V1
    scripted_iter_*.pt, V2 server_models/*.pt) is exported as-is without it."""
    py = Path(__file__).resolve().parents[2] / "SkyZero_2048" / "python"
    if not py.is_dir():
        raise SystemExit(f"Expected SkyZero_2048 python at {py}")
    sys.path.insert(0, str(py))
    from nets import build_net               # noqa: E402
    from model_config import config_from_name  # noqa: E402
    return build_net, config_from_name


class ExportWrapper(torch.nn.Module):
    """Wraps Net2048 and rescales the value head back to raw 2048 points.

    Two value-target conventions (must match the run.cfg the ckpt was trained
    with), so the ONNX `value` output is always raw 2048 points:
      - linear (value_transform=False): head regresses raw/value_scale, so we
        just multiply by value_scale.
      - MuZero h() (value_transform=True): head regresses h(raw)/value_scale, so
        we multiply by value_scale to recover h(raw) then apply h_inv to get raw
        points. h/h_inv mirror python/value_transform.py (eps = 1e-3).
    """

    def __init__(self, model: torch.nn.Module, value_scale: float,
                 value_transform: bool = False) -> None:
        super().__init__()
        self.model = model
        self.value_scale = float(value_scale)
        self.value_transform = bool(value_transform)
        self.eps = 1e-3   # instance attr (not a class const) so jit.script can read it

    def _h_inv(self, y: torch.Tensor) -> torch.Tensor:
        """h-space -> raw points (exact inverse of value_transform.to_h)."""
        eps = self.eps
        z = (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(y) + 1.0 + eps)) - 1.0) / (2.0 * eps)
        return torch.sign(y) * (z * z - 1.0)

    def forward(self, x: torch.Tensor):
        policy_logits, value_scaled = self.model(x)
        value_points = value_scaled * self.value_scale          # raw or h-space
        if self.value_transform:
            value_points = self._h_inv(value_points)            # h-space -> raw
        # Force a (B, 1) shape so onnxruntime-web sees a 2-D tensor regardless
        # of how the scalar head squeezes.
        return policy_logits, value_points.reshape(-1, 1)


def load_state(ckpt_path: Path) -> dict:
    """Extract a state_dict from a bare/dict SkyZero_2048 .pt file (a TorchScript
    module is exported directly by export_one, not rebuilt, so it never reaches here).

      - {"model": state_dict} or {"model_state_dict": ...} wrapper      -> inner
      - bare state_dict (OrderedDict of name -> Tensor)                 -> as-is
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                print(f"[export] using '{key}' from {ckpt_path.name}")
                return state[key]
        if state and all(isinstance(v, torch.Tensor) for v in state.values()):
            print(f"[export] loading bare state_dict from {ckpt_path.name}")
            return state
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def export_one(ckpt: Path, out: Path, net: str | None,
               value_scale: float | None, value_transform: bool, opset: int) -> None:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.jit.ScriptModule):
        # Already-traced net (V1 scripted_iter_*.pt / V2 server_models/*.pt): export
        # it as-is. Rebuilding from --net would risk silently loading the weights
        # into a mismatched architecture (strict=False), so we skip build_net.
        if value_scale is None:
            raise SystemExit("--value-scale is required for a TorchScript checkpoint "
                             "(the traced graph carries no config)")
        print(f"[export] TorchScript module {ckpt.name} -> exporting directly "
              f"(value_scale={value_scale} value_transform={value_transform})")
        model = obj.eval()
        num_planes = NUM_PLANES_2048
    else:
        if net is None:
            raise SystemExit("--net is required for a bare/dict state_dict checkpoint")
        build_net, config_from_name = _import_net_builders()
        cfg = config_from_name(net)
        if value_scale is not None:
            cfg.value_scale = value_scale
        cfg.value_transform = value_transform
        print(f"[export] net={net} blocks={cfg.blocks} channels={cfg.channels} "
              f"c_mid={cfg.c_mid} c_gpool={cfg.c_gpool} value_scale={cfg.value_scale} "
              f"value_transform={cfg.value_transform}")
        model = build_net(cfg)
        missing, unexpected = model.load_state_dict(load_state(ckpt), strict=False)
        if missing:
            print(f"[export] missing keys: {missing}")
        if unexpected:
            print(f"[export] unexpected keys: {unexpected}")
        model.eval()
        value_scale, num_planes = cfg.value_scale, cfg.num_planes

    wrapper = ExportWrapper(model, value_scale, value_transform).eval()
    if isinstance(model, torch.jit.ScriptModule):
        # The wrapper nests an already-traced module; scripting the wrapper makes
        # torch.onnx use the script graph instead of tracing, which otherwise dies
        # with "module ... not part of the active trace".
        wrapper = torch.jit.script(wrapper)

    dummy = torch.zeros(1, num_planes, 4, 4, dtype=torch.float32)
    with torch.no_grad():
        p, v = wrapper(dummy)
        print(f"[export] policy: {tuple(p.shape)}, value: {tuple(v.shape)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "input":         {0: "B"},
            "policy_logits": {0: "B"},
            "value":         {0: "B"},
        },
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))
    print(f"[export] wrote {out}  (check passed)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="source checkpoint (.pt)")
    ap.add_argument("--out", type=Path, required=True, help="output .onnx path")
    ap.add_argument("--net", type=str, default=None,
                    help="net name, e.g. b3c64 — required only for a bare/dict "
                         "state_dict (a TorchScript .pt is exported as-is)")
    ap.add_argument("--value-scale", type=float, default=None,
                    help="override Config.value_scale (default: net's config)")
    ap.add_argument("--value-transform", action="store_true",
                    help="ckpt trained with VALUE_TRANSFORM=1 (h-space target); "
                         "fold the MuZero h_inv into the graph so value stays raw")
    # opset 18 is the minimum that supports Mish; onnxruntime-web 1.17 has it.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        ap.error(f"checkpoint not found: {args.ckpt}")
    export_one(args.ckpt, args.out, args.net, args.value_scale,
               args.value_transform, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
