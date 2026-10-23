(function (scope) {
    class XiangqiGame {
        constructor(native, fen = '', handle = null) {
            this.native = native;
            this.handle = handle ?? native.cwrap('xq_create', 'number', ['string'])(fen);
            if (!this.handle) throw new Error(native.UTF8ToString(native._xq_error()));
        }

        snapshot() {
            if (this.cached) return this.cached;
            const index = this.native._xq_snapshot(this.handle) >>> 2;
            const heap = this.native.HEAP32;
            return this.cached = {player: heap[index], finished: !!heap[index + 1], winner: heap[index + 2],
                reason: heap[index + 3], board: Array.from(heap.subarray(index + 4, index + 104)),
                legal: Array.from(heap.subarray(index + 105, index + 105 + heap[index + 104]))};
        }

        observation() {
            const index = this.native._xq_observation(this.handle) >>> 2;
            return {spatial: this.native.HEAPF32.slice(index, index + 11300),
                globals: this.native.HEAPF32.slice(index + 11300, index + 11308)};
        }

        play(action) {
            if (!Number.isInteger(action) || !this.native._xq_play(this.handle, action)) {
                throw new Error(this.native.UTF8ToString(this.native._xq_error()));
            }
            this.cached = null;
        }

        clone() { return new XiangqiGame(this.native, '', this.native._xq_clone(this.handle)); }
        dispose() { if (this.handle) this.native._xq_delete(this.handle); this.handle = 0; }
    }

    scope.XiangqiEngine = {XiangqiGame};
    if (typeof module !== 'undefined' && module.exports) module.exports = scope.XiangqiEngine;
})(typeof self !== 'undefined' ? self : globalThis);
