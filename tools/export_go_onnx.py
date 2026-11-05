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

class PlayNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_spatial, input_global):
        policy, value, *_ = self.model(input_spatial, input_global)[0]
        return policy[:, 0], value


def export(run, destination):
    source = next(p for p in run.parents if (p / 'python/katago').is_dir())
    sys.path.insert(0, str(source / 'python'))
    from katago.train.load_model import load_model
    state = json.loads((run / 'state.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if config['game'] != 'go' or config['publish_weights'] != 'swa':
        raise ValueError('Expected published Go SWA weights')
    name = Path(state['accepted_model']).name
    checkpoint = run / 'torchmodels_toexport' / name / 'model.ckpt'
    torch.set_num_threads(2)
    _, swa, _ = load_model(str(checkpoint), use_swa=True, device='cpu', pos_len=19)
    model = PlayNetwork(swa.eval()).eval()
    fixture = json.loads((ROOT / 'tests/go-rules.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / '.go.onnx'
    spatial = torch.zeros(1,22,19,19)
    spatial[:,0] = 1
    globals_ = torch.zeros(1,19)
    with torch.inference_mode():
        torch.onnx.export(model,(spatial,globals_),str(temporary),input_names=['input_spatial','input_global'],
            output_names=['policy_logits','wdl_logits'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(temporary)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
    errors = dict(policy_logits=0.,wdl_logits=0.)
    probes = []
    for row in fixture['cases']:
        spatial = np.zeros(22*361,np.float32)
        for i,v in row['spatial']: spatial[i] = v
        spatial = spatial.reshape(1,22,19,19)
        globals_ = np.array([row['globals']],np.float32)
        with torch.inference_mode(): expected = model(torch.from_numpy(spatial),torch.from_numpy(globals_))
        actual = session.run(None,dict(input_spatial=spatial,input_global=globals_))
        for key,a,b in zip(errors,actual,expected):
            np.testing.assert_allclose(a,b.numpy(),rtol=3e-4,atol=3e-4)
            errors[key] = max(errors[key],float(np.abs(a-b.numpy()).max()))
        if not row['finished']:
            probes.append(dict(size=row['size'],actions=row['actions'],outputs={k:v.flatten().tolist() for k,v in zip(errors,actual)}))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    filename = f'go-round{state["completed_rounds"]:06d}-{digest[:12]}.onnx'
    temporary.replace(destination / filename)
    path = destination / 'manifest.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    manifest = dict(version=f'etazero-v4-go-{digest[:12]}',source_run=str(run),sizes=[9,13,19],default_size=19,
        network=dict(canvas=19,spatial=22,globals=19,config=swa.config),
        game=dict(rules='chinese',komi=7.5),model=dict(file=filename,sha256=digest,bytes=(destination / filename).stat().st_size,
        round=state['completed_rounds'],published=name,weights='swa',source=str(checkpoint),
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),onnx_max_absolute_error=errors),rules=old.get('rules',{}))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT / 'tests/go-inference.json').write_text(json.dumps(dict(model=filename,cases=probes),separators=(',',':'))+'\n')
    if old.get('model',{}).get('file',filename) != filename: (destination / old['model']['file']).unlink()
    print(filename, errors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT.parent / 'EtaZero-lab/EtaZero_V4/data/az_pcr_go/runs/no_pcr_225')
    parser.add_argument('--out',type=Path,default=ROOT / 'public/go/models')
    args = parser.parse_args()
    export(args.run.resolve(), args.out.resolve())
