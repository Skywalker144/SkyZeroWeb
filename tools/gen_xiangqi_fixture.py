import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def generate(source):
    sys.path.insert(0, str(source / 'python'))
    from etazero.games import XIANGQI_INDEX
    binary = source / 'build/rules_replay'

    def replay(actions, fen=''):
        request = dict(size=10, canvas=10, rule='xiangqi', actions=actions, options=dict(fen=fen))
        result = subprocess.run([str(binary)], input=json.dumps(request) + '\n',
                                capture_output=True, text=True, check=True)
        return [json.loads(line) for line in result.stdout.splitlines()]

    cases = []
    sequences = [
        ('neutral-repetition', '', [(1, 0, 2, 2), (1, 9, 2, 7), (2, 2, 1, 0), (2, 7, 1, 9)] * 2),
        ('perpetual-check', '5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1',
         [(4, 8, 5, 8), (5, 9, 4, 9), (5, 8, 4, 8), (4, 9, 5, 9)] * 2),
        ('insufficient-material', '5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1', []),
        ('move-limit', 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 119 61', [(1, 0, 2, 2)]),
        ('mate', '4k4/3R1R3/4P4/9/9/9/9/9/9/4K4 b - - 0 1', []),
    ]
    for name, fen, moves in sequences:
        actions = [XIANGQI_INDEX[move] for move in moves]
        cases.append(dict(name=name, fen=fen, actions=actions, states=replay(actions, fen)))
    rng = random.Random(20261023)
    for number in range(2):
        actions = []
        while len(actions) < 65:
            row = replay(actions)[-1]
            if row['finished']:
                break
            actions.append(rng.choice(row['legal']))
        cases.append(dict(name=f'random-{number}', fen='', actions=actions, states=replay(actions)))
    pieces = {'r': 1, 'a': 2, 'c': 3, 'p': 4, 'n': 5, 'b': 6, 'k': 7}
    for case in cases:
        for row in case['states']:
            board = [0] * 100
            for line, rank in enumerate(row['fen'].split()[0].split('/')):
                x = 0
                for char in rank:
                    if char.isdigit():
                        x += int(char)
                    else:
                        board[(9 - line) * 10 + x] = pieces[char.lower()] * (1 if char.isupper() else -1)
                        x += 1
            observation = row.pop('observation')
            row.update(board=board, spatial=[i for i, value in enumerate(observation[:11300]) if value],
                       globals=observation[11300:])
            del row['fen']
    fixture = dict(source=str(source), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), cases=cases)
    (ROOT / 'tests/xiangqi-rules.json').write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
    print(f'{len(cases)} trajectories, {sum(len(c["states"]) for c in cases)} positions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'EtaZero-lab/EtaZero_V3')
    generate(parser.parse_args().source.resolve())
