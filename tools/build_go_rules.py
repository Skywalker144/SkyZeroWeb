import argparse
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools/go'
FILES = ['core/global.cpp', 'core/hash.cpp', 'core/rand.cpp', 'core/sha2.cpp',
         'core/md5.cpp', 'game/board.cpp', 'game/boardhistory.cpp',
         'game/rules.cpp', 'game/placement.cpp', 'game/xiangqi.cpp', 'neuralnet/nninputs.cpp',
         'neuralnet/modelversion.cpp']
FILES += ['external/pikafish/' + name for name in
          ['bitboard.cpp', 'attacks.cpp', 'movegen.cpp', 'position.cpp', 'nnue/features/k6r2nc2.cpp']]


def build(args):
    source = MODULE / 'source'
    if args.sync_source:
        origin = args.sync_source.resolve()
        dependencies = subprocess.run(['g++', '-std=c++20', '-DIS_64BIT', '-DNO_PREFETCH',
            '-I'+str(origin / 'cpp'), '-MM', str(MODULE / 'bridge.cpp'),
            *[str(origin / 'cpp' / name) for name in FILES]], check=True, capture_output=True, text=True).stdout
        paths = [Path(token).resolve() for token in shlex.split(dependencies.replace('\\\n', ' '))
                 if token.startswith(str(origin)+'/')]
        paths += [origin / 'LICENSE', origin / 'licenses/Pikafish.txt']
        shutil.rmtree(source, ignore_errors=True)
        hashes = {}
        for path in sorted(set(paths)):
            relative = path.relative_to(origin)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (MODULE / 'source.json').write_text(json.dumps(dict(source=str(origin), files=hashes), indent=2)+'\n')
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-DIS_64BIT', '-DNO_PREFETCH', '-Wno-deprecated-enum-enum-conversion',
             '-ffunction-sections', '-fdata-sections', '-I'+str(source / 'cpp')]
    files = [str(source / 'cpp' / file) for file in FILES]
    destination = ROOT / 'public/go'
    with tempfile.TemporaryDirectory() as tmp:
        if args.native:
            exe = Path(tmp) / 'reference'
            subprocess.run(['g++', *flags, str(MODULE / 'reference.cpp'), *files, '-Wl,--gc-sections', '-o',str(exe)], check=True)
            result = subprocess.run([str(exe)], check=True, capture_output=True)
            (ROOT / 'tests/go-rules.json').write_bytes(result.stdout)
            return
        output = Path(tmp) / 'rules.js'
        exports = ['go_create', 'go_clone', 'go_delete', 'go_play', 'go_snapshot', 'go_score', 'go_observation', 'go_error']
        subprocess.run([args.emxx, *flags, str(MODULE / 'bridge.cpp'), str(MODULE / 'platform.cpp'), *files,
            '-Wl,--wrap=quick_exit', '-sINITIAL_MEMORY=67108864', '-sALLOW_MEMORY_GROWTH=1', '-sMODULARIZE=1', '-sEXPORT_NAME=createGoRules',
            '-sENVIRONMENT=web,worker,node', '-sFILESYSTEM=0', '-sDISABLE_EXCEPTION_CATCHING=0',
            '-sEXPORTED_FUNCTIONS='+json.dumps(['_'+name for name in exports]),
            '-sEXPORTED_RUNTIME_METHODS='+json.dumps(['cwrap','UTF8ToString']), '-o',str(output)], check=True)
        rules = {}
        for suffix,key in [('js','loader'),('wasm','wasm')]:
            path = output.with_suffix('.'+suffix)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = f'rules-{digest[:12]}.{suffix}'
            shutil.copyfile(path, destination / name)
            rules[key] = name
            rules[key+'_sha256'] = digest
        archive = Path(tmp) / 'source.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as package:
            for file in sorted(MODULE.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts:
                    package.write(file,file.relative_to(ROOT))
            package.write(Path(__file__),Path(__file__).relative_to(ROOT))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        rules['source'] = f'rules-source-{digest[:12]}.zip'
        shutil.copyfile(archive,destination / rules['source'])
        shutil.copyfile(source / 'LICENSE',destination / 'LICENSE.txt')
        shutil.copyfile(source / 'licenses/Pikafish.txt',destination / 'PIKAFISH-LICENSE.txt')
        path = destination / 'models/manifest.json'
        manifest = json.loads(path.read_text()) if path.exists() else {}
        previous = manifest.get('rules',{})
        manifest['rules'] = rules
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        for key in ['loader','wasm','source']:
            if key in previous and previous[key] != rules[key]: (destination / previous[key]).unlink()
        print(json.dumps(rules,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-source',type=Path)
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--emxx',default='em++')
    build(parser.parse_args())
