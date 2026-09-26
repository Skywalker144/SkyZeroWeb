import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0,str(args.source.resolve()))
from shared_game_kernel import load_standard_board
from web.unified_web.policy import build_specs
from web.unified_web.controllers import RandomPursuerController
from web.unified_web.game import InteractiveGame
from az_selfplay.alphazero.tensors import public_tensor_batch, pursuer_candidate_batch
root=Path(__file__).resolve().parents[1]
from az_selfplay.alphazero.config import load_config
from az_selfplay.alphazero.pursuer_action_space import pursuer_order_mask
config=load_config(args.source.resolve()/'az_selfplay/configs/evader_location_belief_backup').game
assert config.gameplay_payload() == json.loads((root/'public/scotlandyard/models.json').read_text())['default_game']
board=load_standard_board()
es,ps=build_specs(board,config)
keys=['phase','round','evader_position','pursuer_positions','possible_locations',
      'evader_tickets','pursuer_tickets','exposed_positions','evader_history',
      'last_evader_move','last_pursuer_moves','last_pursuer_order','round_history',
      'terminal','winner','terminal_reason','legal_evader_moves','pursuer_moves','undo_available']
names=['history_tickets','reveal_positions','pursuer_positions','pursuer_tickets',
       'evader_tickets','possible_mask','round_scalar','search_context',
       'candidate_destinations','candidate_tickets','candidate_mask','order_mask']
fixtures=[]
for seed in range(40):
    game=InteractiveGame(config,RandomPursuerController(seed=seed),None,evader_spec=es,pursuer_spec=ps,seed=seed)
    initial=game.snapshot()
    fixture={'seed':seed,'opening':{'evader':initial['evader_position'],'pursuers':initial['pursuer_positions']},'steps':[]}
    rng=np.random.default_rng(seed)
    while game.result is None:
        legal=game.legal_evader_moves()
        move=legal[int(rng.integers(len(legal)))]
        game.apply_evader_move(move)
        snap=game.snapshot()
        public=public_tensor_batch([game.state.public],game.rules)
        candidates=pursuer_candidate_batch([game.state.public],game.rules)
        order_mask=pursuer_order_mask(game.state.public.pursuer_positions,game.rules.pursuer_candidate_rows(game.state.public),True)
        tensors=(*public.network_args(),candidates.destinations,candidates.tickets,candidates.mask,torch.tensor([order_mask],dtype=torch.bool))
        inputs={k:v.numpy().reshape(-1).tolist() for k,v in zip(names,tensors)}
        fixture['steps'].append({'kind':'evader','move':{'destination':move.destination,'ticket':move.ticket.name.lower()},
            'state':{k:snap[k] for k in keys},'inputs':inputs})
        policy_snapshot=game.pursuer_policy.snapshot()
        actions,order=game.pursuer_policy.act(game.observations.pursuer_frame(game.rules.pursuer_candidate_rows(game.state.public)))
        game.pursuer_policy.restore(policy_snapshot)
        game.advance_pursuers()
        snap=game.snapshot()
        fixture['steps'].append({'kind':'pursuer','actions':actions.tolist(),'order':order.tolist(),'state':{k:snap[k] for k in keys}})
    fixtures.append(fixture)
with tempfile.TemporaryDirectory(prefix='scotland-parity-') as temp:
    path=Path(temp)/'games.json';path.write_text(json.dumps(fixtures))
    subprocess.run(['node',str(root/'tests/scotlandyard-parity.mjs'),str(path)],check=True)
