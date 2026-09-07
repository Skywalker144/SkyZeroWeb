"""Export V0.11 Pursuer checkpoints; no critic/private Evader inputs or search.
Run: conda run -n pytorch python tools/export_scotlandyard.py --source ../ScotlandYardRL/ScotlandYardRL_V0.11
"""
import argparse
import configparser
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort

INPUTS = ['history_tickets', 'reveal_positions', 'pursuer_positions', 'pursuer_tickets',
          'evader_tickets', 'possible_mask', 'round_scalar', 'search_context',
          'candidate_destinations', 'candidate_tickets', 'candidate_mask']

class PursuerExport(nn.Module):
    """Tensor-only greedy decoder; avoid bool-scalar scatter in legacy ONNX tracing."""
    def __init__(self, model, az):
        super().__init__()
        self.model, self.az = model, az
    def forward(self, history_tickets, reveal_positions, pursuer_positions, pursuer_tickets,
                evader_tickets, possible_mask, round_scalar, search_context,
                candidate_destinations, candidate_tickets, candidate_mask):
        m = self.model
        if self.az:
            latent, features, nodes = m._state_features(history_tickets, reveal_positions,
                pursuer_positions, pursuer_tickets, evader_tickets, possible_mask,
                round_scalar, search_context)
        else:
            from shared_models.graph import gather_nodes
            planes, global_features = m.actor_graph.tensor_planes(
                pursuer_positions.unsqueeze(-1), pursuer_tickets, reveal_positions,
                history_tickets, evader_tickets, possible_mask, round_scalar,
                pursuer_positions[:, :0], search_context[:, :0])
            latent, nodes = m.actor_graph(planes, global_features)
            features = gather_nodes(nodes, pursuer_positions)
        order_ids = m.order_head(latent).argmax(-1)
        orders = m.permutation_table[order_ids]
        if self.az:
            hidden = m.recurrent_initial(torch.cat((latent, m.order_embedding(order_ids)), -1))
            previous = m.begin_action.unsqueeze(0)
        else:
            hidden, previous = m.autoregressive_decoder.initial(latent, order_ids)
        # Batch size one is intentional for interactive browser inference.
        positions = pursuer_positions
        actions = torch.full_like(positions, -1)
        identities_grid = torch.arange(3).unsqueeze(0)
        for step in range(3):
            identity = orders[:, step]
            origin = pursuer_positions[0, identity]
            destinations = candidate_destinations[0, identity]
            tickets = candidate_tickets[0, identity]
            occupied = (destinations.unsqueeze(-1) == positions.unsqueeze(1)) & (identities_grid != identity[:, None]).unsqueeze(1)
            valid = candidate_mask[0, identity] & ~occupied.any(-1)
            has_move = valid.any(-1)
            safe = valid | ((~has_move).unsqueeze(1) & (torch.arange(valid.shape[-1]) == 0))
            if self.az:
                hidden, logits = m._step(hidden, previous, features[0, identity],
                    m._candidate_features(destinations, tickets, nodes), safe)
            else:
                hidden = m.autoregressive_decoder.step(hidden, previous, features[0, identity])
                logits = m.autoregressive_action_head(hidden, destinations, tickets, safe, nodes)
            selected = logits.argmax(-1)
            destination = destinations.gather(1, selected[:, None]).squeeze(1)
            ticket = tickets.gather(1, selected[:, None]).squeeze(1)
            actions = torch.where(identities_grid == identity[:, None],
                torch.where(has_move, selected, -torch.ones_like(selected))[:, None], actions)
            positions = torch.where(identities_grid == identity[:, None],
                torch.where(has_move, destination, origin)[:, None], positions)
            if self.az:
                previous = m._next_previous_action(origin, destination, ticket, has_move, nodes)
            else:
                decoder = m.autoregressive_decoder
                previous = decoder.next_previous_action(decoder.selected_action(origin, destination, ticket, nodes), has_move)
        return actions, orders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from shared_game_kernel import load_standard_board, GameRules
    from shared_models.policy_value.networks import PursuerPolicyValueNetwork
    from shared_models.policy_value.config import NetworkConfig
    from shared_models.pursuer import ActorCriticModel
    from ppo_pursuer.settings import config_from_parser
    from web.unified_web.policy import build_specs
    from web.unified_web.map_data import map_payload
    from az.alphazero.tensors import public_tensor_batch, pursuer_candidate_batch
    torch.set_num_threads(1)
    board = load_standard_board()
    az_path = args.source/'az_selfplay/data/finite_tickets_p3/checkpoints/checkpoint_000360.pth'
    ppo_path = args.source/'ppo_pursuer/data/finite_tickets_p3_gnn_external/checkpoints/update_000500.pt'
    manifest = json.loads((az_path.parent.parent/'run_manifest.json').read_text())
    az_checkpoint = torch.load(az_path, map_location='cpu', weights_only=False)
    ppo_checkpoint = torch.load(ppo_path, map_location='cpu', weights_only=False)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_string(ppo_checkpoint['metadata']['resolved_config'])
    config = config_from_parser(cfg_parser)
    rules = GameRules(config.game.resolved_ruleset, board)
    assert rules.ruleset.payload() == manifest['execution_fingerprint']['resolved_ruleset']
    _, spec = build_specs(board, config.game)
    az = PursuerPolicyValueNetwork(board.node_vocab_size, num_pursuers=3,
        max_candidates=board.max_pursuer_actions,
        config=NetworkConfig(**manifest['execution_fingerprint']['network']))
    az.load_state_dict(az_checkpoint['pursuer_model'], strict=True)
    ppo = ActorCriticModel.from_config(spec, config.model)
    ppo.load_state_dict(ppo_checkpoint['model'], strict=True)
    wrappers = [PursuerExport(az, True).eval(), PursuerExport(ppo, False).eval()]
    paths = [az_path, ppo_path]
    ids = ['az-selfplay-000360', 'ppo-gnn-000500']
    labels = ['AZ Selfplay · 360（默认）', 'PPO GNN · 500']
    out = args.output
    static = out/'public'
    (static/'scotlandyard/models').mkdir(parents=True, exist_ok=True)
    (static/'scotlandyard').mkdir(parents=True, exist_ok=True)
    catalog = []
    rng = np.random.default_rng(7301)
    def inputs(state):
        public = public_tensor_batch([state.public], rules, device=torch.device('cpu'))
        candidates = pursuer_candidate_batch([state.public], rules, device=torch.device('cpu'))
        return (*public.network_args(), candidates.destinations, candidates.tickets, candidates.mask)
    state = rules.initial_state(rng)
    state = rules.apply_evader(state, rules.legal_evader_moves(state.evader_private())[0])
    sample = inputs(state)
    for wrapper, source, key, label in zip(wrappers, paths, ids, labels):
        target = static/f'scotlandyard/models/{key}.onnx'
        with torch.inference_mode():
            torch.onnx.export(wrapper, sample, target, input_names=INPUTS,
                output_names=['actions', 'order'], opset_version=17, dynamo=False)
        onnx.checker.check_model(str(target))
        session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
        names = {x.name for x in session.get_inputs()}
        fixtures = []
        count = 0
        for game_index in range(15):
            state = rules.initial_state(rng)
            if game_index >= 12:
                from dataclasses import replace
                tickets = tuple((0, 0, 0) if i <= game_index - 12 else row
                                for i, row in enumerate(state.public.pursuer_tickets))
                state = replace(state, public=replace(state.public, pursuer_tickets=tickets))
            while state.terminal_reason is None:
                legal = rules.legal_evader_moves(state.evader_private())
                move = legal[int(rng.integers(len(legal)))]
                state = rules.apply_evader(state, move)
                tensors = inputs(state)
                with torch.inference_mode():
                    expected = [x.numpy() for x in wrapper(*tensors)]
                    if wrapper.az:
                        native_ids, native_actions, _ = wrapper.model.greedy_joint_actions(*tensors)
                        native_order = wrapper.model.permutation_table[native_ids.long()]
                    else:
                        from web.unified_web.observations import WebObservationAdapter
                        from psro.game.env import FrameBatch
                        es, ps = build_specs(board, config.game)
                        obs = WebObservationAdapter(board, config.game, es, ps)
                        obs.reset(state)
                        native = wrapper.model.sample_team_actions(FrameBatch.stack([obs.pursuer_frame(rules.pursuer_candidate_rows(state.public))]), torch.device('cpu'), deterministic=True)
                        native_actions, native_order = native.actions_by_identity, native.orders
                    np.testing.assert_array_equal(expected[0], native_actions.numpy())
                    np.testing.assert_array_equal(expected[1], native_order.numpy())
                feed = {k:v.numpy() for k,v in zip(INPUTS,tensors) if k in names}
                actual = session.run(None, feed)
                for a,b in zip(expected,actual):
                    np.testing.assert_array_equal(a,b)
                from shared_game_kernel import permutation_id
                joint = rules.build_joint_action(state.public, permutation_id(tuple(actual[1][0])), actual[0][0])
                if len(fixtures) < 3 or (game_index >= 12 and state.public.round_number == 1):
                    fixtures.append({'inputs': {k: {'data':v.reshape(-1).tolist(),'dims':list(v.shape),'type':str(v.dtype)} for k,v in feed.items()},
                                     'actions': actual[0][0].tolist(), 'order': actual[1][0].tolist()})
                state = rules.apply_pursuers(state,joint)
                count += 1
        (out/f'tests/{key}-fixtures.json').write_text(json.dumps(fixtures))
        catalog.append({'id':key, 'label':label, 'kind':'onnx', 'game':config.game.gameplay_payload(),
            'url':f'./models/{key}.onnx', 'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source':str(source.relative_to(args.source)), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'inputs':[{'name':v.name,'type':v.type,'dims':v.shape} for v in session.get_inputs()],
            'parity_decisions':count})
        print(f'{key}: {target.stat().st_size} bytes; {count} PyTorch/ONNX decisions matched', flush=True)
    payload = map_payload(board)
    payload['background'] = './assets/scotland_yard.jpg'
    payload['actions'] = {str(k):[{'destination':m.destination,'ticket':m.ticket.name.lower()} for m in v] for k,v in board.actions.items()}
    payload['start_nodes'] = list(board.start_nodes)
    payload['node_vocab_size'] = board.node_vocab_size
    payload['max_candidates'] = board.max_pursuer_actions
    (static/'scotlandyard/board.json').write_text(json.dumps(payload))
    (static/'scotlandyard/models.json').write_text(json.dumps({'format_version':1,'pursuer':catalog,'evader':[],
        'default_game':config.game.gameplay_payload(), 'rulesets':{'finite_tickets_v0':config.game.gameplay_payload()}},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
