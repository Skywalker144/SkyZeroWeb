import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

INPUTS = ['history_tickets', 'reveal_positions', 'pursuer_positions', 'pursuer_tickets',
          'evader_tickets', 'possible_mask', 'round_scalar', 'search_context',
          'candidate_destinations', 'candidate_tickets', 'candidate_mask', 'order_mask']


class PursuerExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history_tickets, reveal_positions, pursuer_positions, pursuer_tickets,
                evader_tickets, possible_mask, round_scalar, search_context,
                candidate_destinations, candidate_tickets, candidate_mask, order_mask):
        m = self.model
        latent, features, nodes = m.encode_policy_state(history_tickets, reveal_positions,
            pursuer_positions, pursuer_tickets, evader_tickets, possible_mask, round_scalar, search_context)
        order_ids = m.order_head(latent).masked_fill(~order_mask, -torch.inf).argmax(-1)
        orders = m.permutation_table[order_ids]
        hidden = m.recurrent_initial(torch.cat((latent, m.order_embedding(order_ids)), -1))
        previous = m.begin_action.unsqueeze(0)
        positions = pursuer_positions
        actions = torch.full_like(positions, -1)
        identities_grid = torch.arange(m.num_pursuers).unsqueeze(0)
        state_indices = torch.zeros(1, dtype=torch.long)
        for step in range(m.num_pursuers):
            identity = orders[:, step]
            origin = pursuer_positions[0, identity]
            destinations = candidate_destinations[0, identity]
            tickets = candidate_tickets[0, identity]
            occupied = (destinations.unsqueeze(-1) == positions.unsqueeze(1)) & (identities_grid != identity[:, None]).unsqueeze(1)
            valid = candidate_mask[0, identity] & ~occupied.any(-1)
            has_move = valid.any(-1)
            safe = valid | ((~has_move).unsqueeze(1) & (torch.arange(valid.shape[-1]) == 0))
            hidden, logits = m._step(hidden, previous, features[0, identity],
                m._candidate_features(destinations, tickets, state_indices, nodes), safe)
            selected = logits.argmax(-1)
            destination = destinations.gather(1, selected[:, None]).squeeze(1)
            ticket = tickets.gather(1, selected[:, None]).squeeze(1)
            actions = torch.where(identities_grid == identity[:, None],
                torch.where(has_move, selected, -torch.ones_like(selected))[:, None], actions)
            positions = torch.where(identities_grid == identity[:, None],
                torch.where(has_move, destination, origin)[:, None], positions)
            previous = m._next_previous_action(origin, destination, ticket, has_move, state_indices, nodes)
        return actions, orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, default=Path('az_selfplay/data/exp_all/evader_location_belief_backup'))
    parser.add_argument('--config', type=Path, default=Path('az_selfplay/configs/evader_location_belief_backup'))
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[2, 8, 16, 32, 118])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source))
    from shared_game_kernel import GameRules, permutation_id, permutations
    from shared_models.policy_value import build_network_pair
    from az_selfplay.alphazero.config import load_config
    from az_selfplay.alphazero.tensors import public_tensor_batch, pursuer_candidate_batch
    from az_selfplay.alphazero.pursuer_action_space import pursuer_order_mask
    from web.unified_web.map_data import map_payload
    torch.set_num_threads(1)
    config = load_config(args.source / args.config)
    supported = config.game.resolved_ruleset
    if (supported.ruleset_id.value != 'classic_no_double_unlimited_evader'
        or supported.evader_start_mode != 'official' or supported.pursuer_start_mode != 'official'
        or supported.start_mode_weights != (1.0, 0.0)
        or supported.transfer_pursuer_tickets_to_evader or supported.black_ticket_mode.value != 'hidden'
        or not config.search.canonicalize_pursuer_orders):
        raise ValueError('Browser export requires the official-start unlimited-Evader rules and canonical Pursuer orders')
    run_dir = args.source / args.run_dir
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    assert config.game.resolved_ruleset.payload() == manifest['execution_fingerprint']['resolved_ruleset']
    assert config.network.__dict__ == manifest['execution_fingerprint']['network']
    rules = GameRules(config.game.resolved_ruleset)
    board = rules.board
    out = args.output
    static = out / 'public/scotlandyard'
    (static / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'tests').mkdir(parents=True, exist_ok=True)
    catalog = []
    def inputs(state):
        public = public_tensor_batch([state.public], rules)
        candidates = pursuer_candidate_batch([state.public], rules)
        order_mask = pursuer_order_mask(state.public.pursuer_positions,
            rules.pursuer_candidate_rows(state.public), config.search.canonicalize_pursuer_orders)
        return (*public.network_args(), candidates.destinations, candidates.tickets,
                candidates.mask, torch.tensor([order_mask], dtype=torch.bool))
    for checkpoint in args.checkpoints:
        source = run_dir / f'checkpoints/checkpoint_{checkpoint:06d}.pth'
        saved = torch.load(source, map_location='cpu', weights_only=False)
        assert saved['run_id'] == manifest['run_id'] and saved['semantic_hash'] == manifest['semantic_hash']
        assert saved['generation'] == checkpoint
        model, _ = build_network_pair(config.network, rules)
        model.load_state_dict(saved['pursuer_model'], strict=True)
        model.eval()
        wrapper = PursuerExport(model).eval()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        key = f'v016-belief-{checkpoint:06d}'
        target = static / f'models/{key}-{source_hash[:12]}.onnx'
        rng = np.random.default_rng(7301)
        state = rules.initial_state(rng)
        state = rules.apply_evader(state, rules.legal_evader_moves(state.evader_private())[0])
        with torch.inference_mode():
            torch.onnx.export(wrapper, inputs(state), target, input_names=INPUTS,
                output_names=['actions', 'order'], opset_version=17, dynamo=False)
        onnx.checker.check_model(str(target))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
        names = {x.name for x in session.get_inputs()}
        fixtures = []
        count = 0
        for game_index in range(15):
            state = rules.initial_state(rng)
            if game_index >= 12:
                tickets = tuple((0, 0, 0) if i <= game_index - 12 else row
                                for i, row in enumerate(state.public.pursuer_tickets))
                state = replace(state, public=replace(state.public, pursuer_tickets=tickets))
            while rules.terminal_reason(state) is None:
                legal = rules.legal_evader_moves(state.evader_private())
                state = rules.apply_evader(state, legal[int(rng.integers(len(legal)))])
                tensors = inputs(state)
                with torch.inference_mode():
                    expected = [x.numpy() for x in wrapper(*tensors)]
                    native_ids, native_actions, _ = model.greedy_canonical_joint_actions(*tensors)
                    native_order = model.permutation_table[native_ids.long()]
                    np.testing.assert_array_equal(expected[0], native_actions.numpy())
                    np.testing.assert_array_equal(expected[1], native_order.numpy())
                feed = {k:v.numpy() for k,v in zip(INPUTS,tensors) if k in names}
                actual = session.run(None, feed)
                for a,b in zip(expected,actual):
                    np.testing.assert_array_equal(a,b)
                if len(fixtures) < 3 or (game_index >= 12 and state.public.round_number == 1):
                    fixtures.append({'inputs': {k: {'data':v.reshape(-1).tolist(),'dims':list(v.shape),'type':str(v.dtype)} for k,v in feed.items()},
                                     'actions': actual[0][0].tolist(), 'order': actual[1][0].tolist()})
                joint = rules.build_joint_action(state.public, permutation_id(tuple(actual[1][0])), actual[0][0])
                state = rules.apply_pursuers(state,joint)
                count += 1
        (out / f'tests/{key}-fixtures.json').write_text(json.dumps(fixtures))
        catalog.append({'id':key, 'label':f'Checkpoint {checkpoint}', 'checkpoint':checkpoint,
            'kind':'onnx', 'game':config.game.gameplay_payload(),
            'url':f'./models/{target.name}', 'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_version':'V0.16', 'source':str(source.relative_to(args.source)), 'source_sha256':source_hash,
            'run_id':manifest['run_id'], 'semantic_hash':manifest['semantic_hash'],
            'decision_protocol':{'mode':'deterministic_argmax','simulations':0,
                'canonicalize_pursuer_orders':config.search.canonicalize_pursuer_orders,'identity_relabeling':'fixed'},
            'inputs':[{'name':v.name,'type':v.type,'dims':v.shape} for v in session.get_inputs()],
            'parity_decisions':count})
        print(f'{key}: {target.stat().st_size} bytes; {count} native/wrapper/ONNX decisions matched', flush=True)
    payload = map_payload(board)
    payload.update(background='./assets/scotland_yard.jpg',
        actions={str(k):[{'destination':m.destination,'ticket':m.ticket.name.lower()} for m in v] for k,v in board.actions.items()},
        evader_actions={str(k):[{'destination':m.destination,'ticket':m.ticket.name.lower()} for m in v] for k,v in rules.evader_actions.items()},
        start_nodes=list(board.start_nodes), node_vocab_size=board.node_vocab_size,
        max_candidates=board.max_pursuer_actions, pursuer_orders=[list(order) for order in permutations(rules.num_pursuers)])
    (static/'board.json').write_text(json.dumps(payload))
    default = next((m['id'] for m in catalog if m['checkpoint']==8),catalog[0]['id'])
    (static/'models.json').write_text(json.dumps({'format_version':2,'protocol':'v016-belief-pilot-v1',
        'default_model':default,'pursuer':catalog,'evader':[], 'default_game':config.game.gameplay_payload(),
        'rulesets':{config.game.ruleset_id.value:config.game.gameplay_payload()}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
