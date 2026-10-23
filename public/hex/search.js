(function (scope) {
    const puct = typeof module !== 'undefined' && module.exports
        ? require('../shared/puct-search.js') : scope.SkyZeroPUCT;

    function search({game, board, player, evaluate, ...options}) {
        return puct.search({...options, state: {board, player},
            legal: state => Array.from(state.board.keys()).filter(action => !state.board[action]),
            play: (state, action) => ({board: game.play(state.board, action, state.player), player: -state.player}),
            terminalValue: state => { const winner = game.winner(state.board); return winner ? winner * state.player : null; },
            evaluate: state => evaluate(state.board, state.player)});
    }

    scope.HexSearch = {search};
    if (typeof module !== 'undefined' && module.exports) module.exports = {search};
})(typeof self !== 'undefined' ? self : globalThis);
