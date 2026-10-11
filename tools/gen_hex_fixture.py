import json
from pathlib import Path
import random
import subprocess

root = Path(__file__).resolve().parents[1]
source = root.parent / 'EtaZero-lab/EtaZero_V1.4'
binary = root / '.cache/hex-reference'
binary.parent.mkdir(exist_ok=True)
subprocess.run(['g++', '-std=c++17', '-O2', '-I' + str(source / 'cpp/include'),
                '-I' + str(source / 'build/generated'), str(root / 'tools/hex_reference.cpp'),
                str(source / 'cpp/src/game/rules.cpp'), '-o', str(binary)], check=True)
inputs = []
rng = random.Random(20261008)
for size in (5, 9, 10, 11, 12, 13, 14, 15):
    for count in (0, 1, 2, 17, 18):
        moves = rng.sample(range(size * size), count)
        inputs.append(f'{size} {count} ' + ' '.join(map(str, moves)))
inputs.extend(['5 9 0 4 5 9 10 14 15 19 20', '5 10 24 0 20 1 21 2 22 3 19 4'])
result = subprocess.run([str(binary)], input='\n'.join(inputs), text=True, capture_output=True, check=True)
fixture = {'canvas': 15, 'source': 'EtaZero_V1.4 native Game',
           'cases': [json.loads(line) for line in result.stdout.splitlines()]}
(root / 'tests/hex-rules.json').write_text(json.dumps(fixture) + '\n')
print(f'{len(fixture["cases"])} native cases')
