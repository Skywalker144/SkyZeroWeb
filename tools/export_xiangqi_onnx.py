import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ['policy_logits', 'wdl_logits', 'optimistic_logits', 'value_stdev']


def export(run, destination):
    source = next(parent for parent in run.parents if (parent / 'python/etazero').is_dir())
    sys.path.insert(0, str(source / 'python'))
    from etazero.network import make_network, inference_network
    from etazero.symmetry import xiangqi_maps
    config = json.loads((run / '.internal/run.json').read_text())['config']
    info = json.loads((run / '.internal/state.json').read_text())['model']
    if config['agent']['algorithm'] != 'alphazero' or info['game'] != 'xiangqi':
        raise ValueError('Expected an AlphaZero Xiangqi model')
    torch.set_num_threads(2)
    original = run / info['path']
    reference = torch.jit.load(str(original), map_location='cpu').eval()
    model = inference_network(make_network(config).eval()).eval()
    weights = reference.state_dict()
    indices = weights.pop('model.policy_head.action_indices', model.model.policy_head.action_indices)
    torch.testing.assert_close(indices, model.model.policy_head.action_indices, rtol=0, atol=0)
    model.load_state_dict(weights, strict=True)
    fixture = json.loads((ROOT / 'tests/xiangqi-rules.json').read_text())
    rows = fixture['cases'][-1]['states']
    probes = []
    spatial_maps, action_maps = xiangqi_maps(10, torch.device('cpu'))
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / '.xiangqi.onnx'
    spatial = torch.zeros(1, 113, 10, 10)
    spatial[:, 0, :, :9] = 1
    globals_ = torch.tensor([[1, 0, 0, 0, 0, .125, 0, 0]], dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(model, (spatial, globals_), str(temporary), input_names=['input_spatial', 'input_global'],
                          output_names=OUTPUTS, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(temporary)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
    errors = dict.fromkeys(OUTPUTS, 0.)
    for turn in (0, 1, 7, 16, 32, 60):
        row = rows[turn]
        for mirror in (0, 1):
            spatial = np.zeros(11300, np.float32)
            spatial[row['spatial']] = 1
            choice = mirror + 2 * (row['player'] == -1)
            spatial = spatial.reshape(1, 113, 100)[:, :, spatial_maps[choice].numpy()]
            spatial = spatial.reshape(1, 113, 10, 10).copy()
            globals_ = np.array([row['globals']], np.float32)
            with torch.inference_mode():
                expected = reference(torch.from_numpy(spatial), torch.from_numpy(globals_))
            actual = session.run(OUTPUTS, dict(input_spatial=spatial, input_global=globals_))
            for name, a, b in zip(OUTPUTS, actual, expected):
                np.testing.assert_allclose(a, b.numpy(), rtol=2e-4, atol=2e-4)
                errors[name] = max(errors[name], float(np.abs(a - b.numpy()).max()))
            physical = torch.full((info['action_count'],), -torch.inf)
            physical[row['legal']] = torch.from_numpy(actual[0][0])[action_maps[choice][row['legal']]]
            probes.append(dict(actions=fixture['cases'][-1]['actions'][:turn], mirror=mirror,
                               policy=physical.softmax(0).tolist(), wdl=torch.from_numpy(actual[1][0]).softmax(0).tolist(),
                               outputs={name: value.flatten().tolist() for name, value in zip(OUTPUTS, actual)}))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    filename = f'xiangqi-iter{info["checkpoint"]["iteration"]:06d}-{digest[:12]}.onnx'
    temporary.replace(destination / filename)
    manifest_path = destination / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = dict(version=f'etazero-v3-xiangqi-{digest[:12]}', source_run=str(run), network=config['network'],
                    model=dict(file=filename, sha256=digest, bytes=(destination / filename).stat().st_size,
                               source=info, source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                               onnx_max_absolute_error=errors), rules=previous.get('rules', {}))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'tests/xiangqi-inference.json').write_text(json.dumps(dict(model=filename, cases=probes), separators=(',', ':')) + '\n')
    if previous.get('model', {}).get('file', filename) != filename:
        (destination / previous['model']['file']).unlink()
    print(f'{filename}: {errors}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT.parent / 'EtaZero-lab/EtaZero_V3/data/az_go_xiangqi_5h/az_xiangqi')
    parser.add_argument('--out', type=Path, default=ROOT / 'public/xiangqi/models')
    args = parser.parse_args()
    export(args.run.resolve(), args.out.resolve())
