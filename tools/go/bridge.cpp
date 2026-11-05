#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include <string>

struct GoState {
    Board board;
    BoardHistory history;
    Player player;
    GoState(int size, float komi) : board(size, size), player(P_BLACK) {
        Rules rules = Rules::parseRulesWithoutKomi("chinese", komi);
        history = BoardHistory(board, player, rules, 0, BoardHistoryModes(false, false));
    }
};
static std::string error;
extern "C" {
GoState* go_create(int size, float komi) {
    try {
        if (size != 9 && size != 13 && size != 19) throw StringError("Unsupported board size");
        Board::initHash();
        return new GoState(size, komi);
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
GoState* go_clone(GoState* state) { return new GoState(*state); }
void go_delete(GoState* state) { delete state; }
const char* go_error() { return error.c_str(); }
int go_play(GoState* state, int action) {
    int size = state->board.x_size;
    if (state->history.isGameFinished || action < 0 || action > size*size) return 0;
    Loc loc = action == size*size ? Board::PASS_LOC : Location::getLoc(action % size, action / size, size);
    if (!state->history.isLegal(state->board, loc, state->player)) return 0;
    state->history.makeBoardMoveAssumeLegal(state->board, loc, state->player, nullptr);
    state->player = getOpp(state->player);
    return 1;
}
const int* go_snapshot(GoState* state) {
    static std::vector<int> out;
    auto sign = [](Player p) { return p == P_BLACK ? 1 : p == P_WHITE ? -1 : 0; };
    int size = state->board.x_size;
    out = {sign(state->player), state->history.isGameFinished, sign(state->history.winner), size};
    for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
        out.push_back(sign(state->board.colors[Location::getLoc(x,y,size)]));
    out.push_back(0);
    if (!state->history.isGameFinished) for (int a = 0; a <= size*size; a++) {
        Loc loc = a == size*size ? Board::PASS_LOC : Location::getLoc(a % size,a / size,size);
        if (state->history.isLegal(state->board,loc,state->player)) out.push_back(a);
    }
    out[4+size*size] = int(out.size()) - 5 - size*size;
    return out.data();
}
float go_score(GoState* state) { return state->history.finalWhiteMinusBlackScore; }
const float* go_observation(GoState* state) {
    static std::vector<float> out(22*19*19+19);
    MiscNNInputParams params;
    NNInputs::fillRowV7(state->board,state->history,state->player,params,19,19,false,out.data(),out.data()+22*19*19);
    return out.data();
}
}
