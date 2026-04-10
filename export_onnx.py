import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from alphazero import AlphaZero
from envs.gomoku import Gomoku
from gomoku.gomoku_train import train_args
from nets import ResNet
import onnx


class BNToAffine(nn.Module):
    """Replace BatchNorm2d with a fixed affine transform (eval-mode equivalent).

    BN eval computes: y = (x - mean) / sqrt(var + eps) * weight + bias
    which is just:    y = x * scale + shift
    This eliminates the BatchNormalization ONNX op entirely, avoiding
    potential bugs in certain onnxruntime-web WASM backends.
    """

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale
        self.register_buffer("scale", scale.reshape(1, -1, 1, 1))
        self.register_buffer("shift", shift.reshape(1, -1, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


def fold_bn(module: nn.Module):
    """Recursively replace every BatchNorm2d with BNToAffine in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, BNToAffine(child))
        elif isinstance(child, nn.Sequential):
            new_layers = []
            for layer in child:
                if isinstance(layer, nn.BatchNorm2d):
                    new_layers.append(BNToAffine(layer))
                else:
                    fold_bn(layer)
                    new_layers.append(layer)
            setattr(module, name, nn.Sequential(*new_layers))
        else:
            fold_bn(child)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_args["device"] = device

    game = Gomoku(board_size=train_args["board_size"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZero(game, model, optimizer, train_args)
    weight_file = "gomoku_model_from_cpp_state_dict.pth"
    alphazero.load_model(weight_file)
    print(f"Loaded model weights: {weight_file}")

    model.eval()

    # Fold all BatchNorm2d into fixed affine transforms (Mul + Add) so that
    # the exported ONNX graph contains zero BatchNormalization nodes.
    # This works around onnxruntime-web WASM backends that mishandle BN
    # eval mode (training_mode=0), producing near-random outputs.
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    fold_bn(model)
    bn_remaining = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    affine_count = sum(1 for m in model.modules() if isinstance(m, BNToAffine))
    print(f"Folded {bn_count} BatchNorm2d -> {affine_count} BNToAffine ({bn_remaining} BN remaining)")

    dummy_input = torch.randn(1, game.num_planes, game.board_size, game.board_size).to(train_args["device"])
    
    onnx_model_name = "model.onnx"

    output_names = ["policy_logits", "opponent_policy_logits", "value_logits"]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "opponent_policy_logits": {0: "batch_size"},
            "value_logits": {0: "batch_size"}
        },
        dynamo=False 
    )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")
