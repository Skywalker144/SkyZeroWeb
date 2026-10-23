#include "etazero/rule_state.h"
#include <string>

namespace etazero {
std::unique_ptr<RuleState> make_xiangqi_state(int, int, const RuleOptions&);
}

using etazero::RuleState;
static std::string error;

extern "C" {
RuleState* xq_create(const char* fen) {
    try {
        etazero::RuleOptions options;
        options.fen = fen;
        return etazero::make_xiangqi_state(10, 10, options).release();
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}

RuleState* xq_clone(RuleState* state) { return state->clone().release(); }
void xq_delete(RuleState* state) { delete state; }
const char* xq_error() { return error.c_str(); }

int xq_play(RuleState* state, int action) {
    if (!state->legal(action)) { error = "Illegal Xiangqi move"; return 0; }
    state->play(action);
    return 1;
}

const int* xq_snapshot(RuleState* state) {
    static std::vector<int> out;
    const auto outcome = state->outcome();
    out = {state->player(), state->finished(), outcome.winner, int(outcome.reason)};
    const auto& board = state->board().cells;
    out.insert(out.end(), board.begin(), board.end());
    out.push_back(0);
    for (int a = 0; a < etazero::game_spec(etazero::GameKind::XIANGQI).fixed_actions; ++a)
        if (state->legal(a)) out.push_back(a);
    out[104] = int(out.size()) - 105;
    return out.data();
}

const float* xq_observation(RuleState* state) {
    static std::vector<float> out;
    out = state->observation(state->player(), 0, true);
    return out.data();
}

const int* xq_moves() { return &etazero::XIANGQI_MOVES[0][0]; }
}
