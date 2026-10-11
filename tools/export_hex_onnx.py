from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch


OUTPUTS = ['policy_logits', 'wdl_logits', 'optimistic_logits', 'value_stdev']


def export_run(run: Path, destination: Path) -> None:
    torch.set_num_threads(2)
    state = json.loads((run / '.internal/state.json').read_text())
    config = json.loads((run / '.internal/run.json').read_text())['config']
    sys.path.insert(0, str(run.parents[1] / 'python'))
    from etazero.network import make_network, inference_network
    from etazero.engine_config import load_match_hex_opening_config
    if config['agent']['algorithm'] != 'alphazero' or config['environment']['rules'] != 'hex':
        raise ValueError('Expected an AlphaZero Hex run')
    rows = [json.loads((run / 'logs/iterations' / f'{i:06d}.json').read_text())
            for i in range(state['iteration'])]
    if rows[-1]['elapsed_seconds'] != state['elapsed_seconds']:
        raise ValueError('Committed timing does not match state')
    rows = [row for row in rows if row['model']]
    choices = [('5h', 18000), ('6h', 21600), ('last', None)]
    previous_path = destination / 'manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {'models': []}
    opening_profile = run.parents[1] / 'configs/baseline_hex'
    opening = load_match_hex_opening_config(opening_profile, environ={})
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        'version': 'etazero-v1.4-hex-20261009', 'source_run': str(run.resolve()),
        'network': config['network'],
        'hex_opening': opening, 'opening_profile': str(opening_profile / 'match.cfg'),
        'sizes': [int(size) for size in config['environment']['sizes'].split(',')],
        'timing_basis': rows[-1]['timing_basis'],
        'selection': 'nearest committed elapsed_seconds to target; last is latest committed model',
        'default': 'last', 'models': [],
    }
    rng = np.random.default_rng(20261008)
    for name, target in choices:
        row = rows[-1] if target is None else min(rows, key=lambda r: abs(r['elapsed_seconds'] - target))
        info = row['model']
        source = run / info['path']
        if hashlib.sha256(source.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Source hash mismatch: {source}')
        reference = torch.jit.load(str(source), map_location='cpu').eval()
        canvas, contract = reference.metadata()
        if canvas != config['network']['canvas'] or contract != info['contract']:
            raise ValueError('Model metadata mismatch')
        model = inference_network(make_network(config).eval()).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        spatial = torch.zeros(1, 5, canvas, canvas)
        spatial[:, 0] = 1
        globals_ = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)
        temporary = destination / f'.{name}.onnx'
        with torch.inference_mode():
            torch.onnx.export(model, (spatial, globals_), str(temporary),
                              input_names=['input_spatial', 'input_global'],
                              output_names=OUTPUTS, opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(temporary)))
        session = ort.InferenceSession(str(temporary), providers=['CPUExecutionProvider'])
        probes = []
        errors = {key: 0.0 for key in OUTPUTS}
        for size in manifest['sizes']:
            for player in (1, -1):
                board = np.zeros((size, size), np.int8)
                count = min(18 + (player == -1), size * size - 1)
                for ply, action in enumerate(rng.choice(size * size, count, replace=False)):
                    board.flat[action] = 1 if ply % 2 == 0 else -1
                for rotation in (0, 1):
                    obs = np.zeros((1, 5, canvas, canvas), np.float32)
                    obs[0, 0, :size, :size] = 1
                    obs[0, 1, :size, :size] = board == player
                    obs[0, 2, :size, :size] = board == -player
                    global_values = np.array([[0, 0, 0, 0, 0, 0, 1, player == -1]], np.float32)
                    if rotation:
                        obs = obs[:, :, ::-1, ::-1]
                    if player == -1:
                        obs = obs.transpose(0, 1, 3, 2)
                    obs = obs.copy()
                    with torch.inference_mode():
                        expected = reference(torch.from_numpy(obs), torch.from_numpy(global_values))
                    actual = session.run(OUTPUTS, {'input_spatial': obs, 'input_global': global_values})
                    for key, a, b in zip(OUTPUTS, actual, expected):
                        np.testing.assert_allclose(a, b.numpy(), rtol=2e-4, atol=2e-4)
                        errors[key] = max(errors[key], float(np.abs(a - b.numpy()).max()))
                    if size in (9, 15):
                        probes.append({'size': size, 'player': player, 'rotation': rotation,
                                       'board': board.flatten().tolist(),
                                       'outputs': {key: value.tolist() for key, value in zip(OUTPUTS, actual)}})
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        filename = f'hex-{name}-iter{row["iteration"]:06d}-{digest[:12]}.onnx'
        temporary.replace(destination / filename)
        manifest['models'].append({
            'id': name, 'file': filename, 'iteration': row['iteration'],
            'elapsed_seconds': row['elapsed_seconds'], 'target_seconds': target,
            'sha256': digest, 'bytes': (destination / filename).stat().st_size,
            'source': info, 'onnx_max_absolute_error': errors,
        })
        fixture = {'canvas': canvas, 'model': filename, 'cases': probes}
        (destination.parents[2] / 'tests' / f'hex-{name}-inference.json').write_text(json.dumps(fixture))
        print(f'{name}: iteration {row["iteration"]}, {row["elapsed_seconds"]:.3f}s, {filename}, errors={errors}', flush=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    retained_files = {model['file'] for model in manifest['models']}
    retained_ids = {model['id'] for model in manifest['models']}
    for old in previous['models']:
        if old['file'] not in retained_files:
            (destination / old['file']).unlink()
        if old['id'] not in retained_ids:
            (destination.parents[2] / 'tests' / f'hex-{old["id"]}-inference.json').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--run', type=Path, default=root.parent / 'EtaZero-lab/EtaZero_V1.4/data/baseline_hex')
    parser.add_argument('--out', type=Path, default=root / 'public/hex/models')
    args = parser.parse_args()
    export_run(args.run.resolve(), args.out.resolve())
