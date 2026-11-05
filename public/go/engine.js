(function (scope) {
    class GoGame {
        constructor(native, size = 19, komi = 7.5, handle = null) {
            this.native = native;
            this.size = size;
            this.handle = handle ?? native._go_create(size, komi);
            if (!this.handle) throw new Error(native.UTF8ToString(native._go_error()));
        }
        snapshot() {
            if (this.cached) return this.cached;
            const index = this.native._go_snapshot(this.handle) >>> 2;
            const heap = this.native.HEAP32, area = this.size ** 2;
            return this.cached = {player: heap[index], finished: !!heap[index+1], winner: heap[index+2], size: this.size,
                board: Array.from(heap.subarray(index+4,index+4+area)),
                legal: Array.from(heap.subarray(index+5+area,index+5+area+heap[index+4+area])),
                score: this.native._go_score(this.handle)};
        }
        observation() {
            const index = this.native._go_observation(this.handle) >>> 2;
            return {spatial: this.native.HEAPF32.slice(index,index+22*361),
                globals: this.native.HEAPF32.slice(index+22*361,index+22*361+19)};
        }
        play(action) {
            if (!Number.isInteger(action) || !this.native._go_play(this.handle,action)) throw new Error('Illegal Go move');
            this.cached = null;
        }
        clone() { return new GoGame(this.native,this.size,0,this.native._go_clone(this.handle)); }
        dispose() { if (this.handle) this.native._go_delete(this.handle); this.handle = 0; }
    }
    scope.GoEngine = {GoGame};
    if (typeof module !== 'undefined' && module.exports) module.exports = scope.GoEngine;
})(typeof self !== 'undefined' ? self : globalThis);
