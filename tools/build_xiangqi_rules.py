import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools/xiangqi'


def build(args):
    source = MODULE / 'source'
    provenance = MODULE / 'source.json'
    if args.sync_source:
        origin = args.sync_source.resolve()
        paths = list((origin / 'cpp/external/pikafish').rglob('*'))
        paths += [origin / 'cpp/src/game/xiangqi.cpp']
        paths += [origin / 'cpp/include/etazero' / name for name in
                  ['rule_state.h', 'outcome.h', 'game_spec.h', 'rules.h', 'geometry.h', 'subtree_value_bias_key.h']]
        paths += [origin / 'licenses/Pikafish.txt']
        shutil.rmtree(source, ignore_errors=True)
        hashes = {}
        for path in paths:
            if not path.is_file():
                continue
            relative = path.relative_to(origin)
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        sys.path.insert(0, str(origin / 'python'))
        from etazero.schema import generate_header
        generate_header(source / 'generated/etazero/schema.h')
        provenance.write_text(json.dumps({'source': str(origin), 'files': hashes,
            'schema_inputs': {name: hashlib.sha256((origin / 'python/etazero' / name).read_bytes()).hexdigest()
                              for name in ['schema.py', 'games.py', 'outcomes.py']}}, indent=2) + '\n')
    destination = ROOT / 'public/xiangqi'
    destination.mkdir(parents=True, exist_ok=True)
    exports = ['xq_create', 'xq_clone', 'xq_delete', 'xq_play', 'xq_snapshot', 'xq_observation', 'xq_moves', 'xq_error']
    files = ['bitboard.cpp', 'attacks.cpp', 'movegen.cpp', 'position.cpp', 'nnue/features/k6r2nc2.cpp']
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'rules.js'
        subprocess.run([args.emxx, '-std=c++20', '-O3', '-DNDEBUG', '-DIS_64BIT', '-DNO_PREFETCH',
            '-Wno-deprecated-enum-enum-conversion',
            '-I' + str(source / 'cpp/include'), '-I' + str(source / 'generated'),
            str(MODULE / 'bridge.cpp'), str(source / 'cpp/src/game/xiangqi.cpp'),
            *[str(source / 'cpp/external/pikafish' / file) for file in files],
            '-sINITIAL_MEMORY=67108864', '-sALLOW_MEMORY_GROWTH=1', '-sMODULARIZE=1', '-sEXPORT_NAME=createXiangqiRules',
            '-sENVIRONMENT=web,worker,node', '-sFILESYSTEM=0', '-sDISABLE_EXCEPTION_CATCHING=0',
            '-sEXPORTED_FUNCTIONS=' + json.dumps(['_' + name for name in exports]),
            '-sEXPORTED_RUNTIME_METHODS=' + json.dumps(['cwrap', 'UTF8ToString']), '-o', str(output)], check=True)
        rules = {}
        for suffix, key in [('js', 'loader'), ('wasm', 'wasm')]:
            file = output.with_suffix('.' + suffix)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            name = f'rules-{digest[:12]}.{suffix}'
            shutil.copyfile(file, destination / name)
            rules[key] = name
            rules[key + '_sha256'] = digest
        archive = Path(temporary) / 'rules-source.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
            for file in sorted(MODULE.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts:
                    package.write(file, file.relative_to(ROOT))
            package.write(Path(__file__), Path(__file__).relative_to(ROOT))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        rules['source'] = f'rules-source-{digest[:12]}.zip'
        shutil.copyfile(archive, destination / rules['source'])
        shutil.copyfile(source / 'licenses/Pikafish.txt', destination / 'LICENSE.txt')
    manifest_path = destination / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version': '20261023'}
    previous = manifest.get('rules', {})
    manifest['rules'] = rules
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for key in ('loader', 'wasm', 'source'):
        if key in previous and previous[key] != rules[key]:
            (destination / previous[key]).unlink()
    print(json.dumps(rules, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-source', type=Path)
    parser.add_argument('--emxx', default='em++')
    build(parser.parse_args())
