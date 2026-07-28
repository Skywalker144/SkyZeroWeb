#!/usr/bin/env python3
"""Export a SkyZero V7.19 TorchScript checkpoint for SkyZeroWeb."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import onnx
import onnxruntime
import torch


OUTPUT_NAMES = (
    "policy_logits",
    "value_wdl_logits",
    "value_futurepos_pretanh",
    "value_st_error_sq",
)


class WebWrapper(torch.nn.Module):
    """Expose every V7.19 head consumed by the browser UI and search."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_spatial: torch.Tensor, input_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(input_spatial, input_global)
        return (
            output["policy"],
            output["value_wdl"],
            output["value_futurepos"],
            output["value_st_error"],
        )


def maximum_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--network", default="b9c96tflrs")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=here.parents[1] / "SkyZero" / "SkyZero_V7.19",
    )
    args = parser.parse_args()

    checkpoint = args.ckpt.resolve()
    output = args.out.resolve()
    source_root = args.source_root.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not (source_root / "python").is_dir():
        raise SystemExit(f"V7.19 source root not found: {source_root}")

    sys.path.insert(0, str(source_root / "python"))
    from full_nets import RMSNorm  # noqa: PLC0415
    from model_config import net_config_from_name  # noqa: PLC0415
    from nets import build_model  # noqa: PLC0415

    scripted = torch.jit.load(str(checkpoint), map_location="cpu").eval()
    model = build_model(net_config_from_name(args.network))
    incompatible = model.load_state_dict(scripted.state_dict(), strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith(".pos_x") and not key.endswith(".pos_y")
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "checkpoint/source topology mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.include_intermediate_outputs = False
    model.set_norm_scales()
    model.eval()

    # The legacy ONNX exporter does not implement aten::rms_norm.
    def export_rms_norm(
        module: torch.nn.Module, value: torch.Tensor
    ) -> torch.Tensor:
        mean_square = value.square().mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(mean_square + module.eps) * module.weight

    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = export_rms_norm.__get__(module, type(module))
    wrapper = WebWrapper(model).eval()

    spatial = torch.zeros(1, 5, 15, 15, dtype=torch.float32)
    spatial[:, 0] = 1.0
    global_features = torch.zeros(1, 7, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        reference_dict = scripted(spatial, global_features)
        reference = (
            reference_dict["policy"],
            reference_dict["value_wdl"],
            reference_dict["value_futurepos"],
            reference_dict["value_st_error"],
        )
        rebuilt = wrapper(spatial, global_features)
        rebuilt_errors = [
            maximum_error(actual, expected)
            for actual, expected in zip(rebuilt, reference)
        ]
        if max(rebuilt_errors) > 1e-4:
            raise RuntimeError(
                "rebuilt model differs from TorchScript: "
                + ", ".join(
                    f"{name}={error:.3g}"
                    for name, error in zip(OUTPUT_NAMES, rebuilt_errors)
                )
            )
        print(
            "TorchScript parity: "
            + ", ".join(
                f"{name}={error:.3g}"
                for name, error in zip(OUTPUT_NAMES, rebuilt_errors)
            )
        )
        torch.onnx.export(
            wrapper,
            (spatial, global_features),
            str(output),
            input_names=["input_spatial", "input_global"],
            output_names=list(OUTPUT_NAMES),
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    ort_session = onnxruntime.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    ort_values = ort_session.run(
        list(OUTPUT_NAMES),
        {
            "input_spatial": spatial.numpy(),
            "input_global": global_features.numpy(),
        },
    )
    ort_errors = [
        maximum_error(torch.from_numpy(actual), expected)
        for actual, expected in zip(ort_values, rebuilt)
    ]
    if max(ort_errors) > 1e-4:
        raise RuntimeError(
            "ONNX differs from rebuilt model: "
            + ", ".join(
                f"{name}={error:.3g}"
                for name, error in zip(OUTPUT_NAMES, ort_errors)
            )
        )
    print(
        "ONNX parity: "
        + ", ".join(
            f"{name}={error:.3g}"
            for name, error in zip(OUTPUT_NAMES, ort_errors)
        )
    )
    print(f"exported {checkpoint} -> {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
