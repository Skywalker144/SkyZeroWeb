const C_EMPTY = 0;
const C_BLACK = 1;
const C_WHITE = 2;
const C_WALL = 3;

class ForbiddenPointFinder {
    constructor(size = 15) {
        this.boardSize = size;
        this.cBoard = Array.from({ length: size + 2 }, () => new Int8Array(size + 2).fill(C_WALL));
        this.clear();
    }

    clear() {
        for (let i = 1; i <= this.boardSize; i++) {
            for (let j = 1; j <= this.boardSize; j++) {
                this.cBoard[i][j] = C_EMPTY;
            }
        }
    }

    setStone(x, y, cStone) {
        this.cBoard[x + 1][y + 1] = cStone;
    }

    isFive(x, y, nColor, nDir = null) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;

        if (nDir === null) {
            this.setStone(x, y, nColor);
            let found = false;
            for (let d = 1; d <= 4; d++) {
                let length = this._checkLineLength(x, y, nColor, d);
                if (nColor === C_BLACK) {
                    if (length === 5) { found = true; break; }
                } else {
                    if (length >= 5) { found = true; break; }
                }
            }
            this.setStone(x, y, C_EMPTY);
            return found;
        }

        this.setStone(x, y, nColor);
        let length = this._checkLineLength(x, y, nColor, nDir);
        this.setStone(x, y, C_EMPTY);

        return nColor === C_BLACK ? length === 5 : length >= 5;
    }

    _checkLineLength(x, y, nColor, nDir) {
        let [dx, dy] = this._getDir(nDir);
        let length = 1;

        let i = x + dx, j = y + dy;
        while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
            if (this.cBoard[i + 1][j + 1] === nColor) {
                length++; i += dx; j += dy;
            } else break;
        }

        i = x - dx; j = y - dy;
        while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
            if (this.cBoard[i + 1][j + 1] === nColor) {
                length++; i -= dx; j -= dy;
            } else break;
        }
        return length;
    }

    _getDir(nDir) {
        if (nDir === 1) return [1, 0];
        if (nDir === 2) return [0, 1];
        if (nDir === 3) return [1, 1];
        if (nDir === 4) return [1, -1];
        return [0, 0];
    }

    isOverline(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        this.setStone(x, y, C_BLACK);
        let bOverline = false;
        for (let d = 1; d <= 4; d++) {
            let length = this._checkLineLength(x, y, C_BLACK, d);
            if (length === 5) {
                this.setStone(x, y, C_EMPTY);
                return false;
            } else if (length >= 6) {
                bOverline = true;
            }
        }
        this.setStone(x, y, C_EMPTY);
        return bOverline;
    }

    isFour(x, y, nColor, nDir) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, nColor)) return false;
        if (nColor === C_BLACK && this.isOverline(x, y)) return false;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let found = false;
        for (let sign of [1, -1]) {
            let curDx = dx * sign, curDy = dy * sign;
            let i = x + curDx, j = y + curDy;
            while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
                let c = this.cBoard[i + 1][j + 1];
                if (c === nColor) { i += curDx; j += curDy; }
                else if (c === C_EMPTY) {
                    if (this.isFive(i, j, nColor, nDir)) found = true;
                    break;
                } else break;
            }
            if (found) break;
        }
        this.setStone(x, y, C_EMPTY);
        return found;
    }

    isOpenFour(x, y, nColor, nDir) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return 0;
        if (this.isFive(x, y, nColor)) return 0;
        if (nColor === C_BLACK && this.isOverline(x, y)) return 0;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let nLine = 1;

        let i = x - dx, j = y - dy;
        while (true) {
            if (!(i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize)) {
                this.setStone(x, y, C_EMPTY); return 0;
            }
            let c = this.cBoard[i + 1][j + 1];
            if (c === nColor) { nLine++; i -= dx; j -= dy; }
            else if (c === C_EMPTY) {
                if (!this.isFive(i, j, nColor, nDir)) {
                    this.setStone(x, y, C_EMPTY); return 0;
                }
                break;
            } else { this.setStone(x, y, C_EMPTY); return 0; }
        }

        i = x + dx; j = y + dy;
        while (true) {
            if (!(i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize)) break;
            let c = this.cBoard[i + 1][j + 1];
            if (c === nColor) { nLine++; i += dx; j += dy; }
            else if (c === C_EMPTY) {
                if (this.isFive(i, j, nColor, nDir)) {
                    this.setStone(x, y, C_EMPTY);
                    return nLine === 4 ? 1 : 2;
                }
                break;
            } else break;
        }
        this.setStone(x, y, C_EMPTY);
        return 0;
    }

    isDoubleFour(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, C_BLACK)) return false;
        let nFour = 0;
        for (let d = 1; d <= 4; d++) {
            let ret = this.isOpenFour(x, y, C_BLACK, d);
            if (ret === 2) nFour += 2;
            else if (this.isFour(x, y, C_BLACK, d)) nFour += 1;
        }
        return nFour >= 2;
    }

    isOpenThree(x, y, nColor, nDir) {
        if (this.isFive(x, y, nColor)) return false;
        if (nColor === C_BLACK && this.isOverline(x, y)) return false;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let found = false;
        for (let sign of [1, -1]) {
            let curDx = dx * sign, curDy = dy * sign;
            let i = x + curDx, j = y + curDy;
            while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
                let c = this.cBoard[i + 1][j + 1];
                if (c === nColor) { i += curDx; j += curDy; }
                else if (c === C_EMPTY) {
                    if (this.isOpenFour(i, j, nColor, nDir) === 1) {
                        if (nColor === C_BLACK) {
                            // In Renju, an open three must be able to become a legal open four.
                            if (!this.isDoubleFour(i, j) && !this.isDoubleThree(i, j) && !this.isOverline(i, j)) found = true;
                        } else found = true;
                    }
                    break;
                } else break;
            }
            if (found) break;
        }
        this.setStone(x, y, C_EMPTY);
        return found;
    }

    isDoubleThree(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, C_BLACK)) return false;
        let nThree = 0;
        for (let d = 1; d <= 4; d++) {
            if (this.isOpenThree(x, y, C_BLACK, d)) nThree++;
        }
        return nThree >= 2;
    }

    isForbidden(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        let nearbyBlack = 0;
        for (let i = Math.max(0, x - 2); i <= Math.min(this.boardSize - 1, x + 2); i++) {
            for (let j = Math.max(0, y - 2); j <= Math.min(this.boardSize - 1, y + 2); j++) {
                if (i === x && j === y) continue;
                if (this.cBoard[i + 1][j + 1] === C_BLACK) {
                    if (Math.abs(i - x) + Math.abs(j - y) !== 3) nearbyBlack++;
                }
            }
        }
        if (nearbyBlack < 2) return false;
        return this.isDoubleThree(x, y) || this.isDoubleFour(x, y) || this.isOverline(x, y);
    }
}

class Gomoku {
    constructor(boardSize = 15, useRenju = true) {
        this.boardSize = boardSize;
        this.useRenju = useRenju;
        this.numPlanes = 4;
        this.fpf = new ForbiddenPointFinder(boardSize);
    }

    /**
     * State is a single-element array containing one Int8Array board.
     * state[0] is the flat board of size boardSize*boardSize.
     * Values: 1 = black, -1 = white, 0 = empty.
     */
    getInitialState() {
        return [new Int8Array(this.boardSize * this.boardSize)];
    }

    isNearOccupied(board, r, c, dist) {
        for (let dr = -dist; dr <= dist; dr++) {
            for (let dc = -dist; dc <= dist; dc++) {
                const nr = r + dr;
                const nc = c + dc;
                if (nr >= 0 && nr < this.boardSize && nc >= 0 && nc < this.boardSize) {
                    if (board[nr * this.boardSize + nc] !== 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    getLegalActions(state, toPlay, restricted = true) {
        const currentBoard = state[0];
        const legalMask = new Uint8Array(this.boardSize * this.boardSize).fill(0);
        let hasStone = false;

        for (let i = 0; i < currentBoard.length; i++) {
            if (currentBoard[i] !== 0) {
                hasStone = true;
                break;
            }
        }

        if (restricted && !hasStone && this.useRenju) {
            const center = Math.floor(this.boardSize / 2) * this.boardSize + Math.floor(this.boardSize / 2);
            legalMask[center] = 1;
            return legalMask;
        }

        for (let i = 0; i < currentBoard.length; i++) {
            if (currentBoard[i] !== 0) {
                continue;
            }
            if (!restricted) {
                legalMask[i] = 1;
                continue;
            }

            const r = Math.floor(i / this.boardSize);
            const c = i % this.boardSize;
            if (this.isNearOccupied(currentBoard, r, c, 3)) {
                legalMask[i] = 1;
            }
        }
        return legalMask;
    }

    getNextState(state, action, toPlay) {
        const newBoard = new Int8Array(state[0]);
        newBoard[action] = toPlay;
        return [newBoard];
    }

    /**
     * Check for winner.
     * @param {Array} state - single-frame state [Int8Array]
     * @param {number|null} lastAction - last placed action index (for Renju forbidden check)
     * @param {number|null} lastPlayer - last player who moved (1 or -1)
     * @returns {number|null} 1=black wins, -1=white wins, 0=draw, null=ongoing
     */
    getWinner(state, lastAction = null, lastPlayer = null) {
        const board = state[0];
        const size = this.boardSize;

        // Renju forbidden move check: if Black just played on a forbidden point, White wins.
        if (this.useRenju && lastAction !== null && lastPlayer === 1) {
            const row = Math.floor(lastAction / size);
            const col = lastAction % size;
            this.fpf.clear();
            for (let i = 0; i < board.length; i++) {
                if (board[i] !== 0) {
                    const r = Math.floor(i / size);
                    const c = i % size;
                    if (r === row && c === col) continue;  // skip last placed stone
                    this.fpf.setStone(r, c, board[i] === 1 ? C_BLACK : C_WHITE);
                }
            }
            if (this.fpf.isForbidden(row, col)) {
                return -1;  // White wins
            }
        }

        const dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const stone = board[r * size + c];
                if (stone === 0) {
                    continue;
                }
                for (const [dr, dc] of dirs) {
                    const pr = r - dr;
                    const pc = c - dc;
                    if (pr >= 0 && pr < size && pc >= 0 && pc < size && board[pr * size + pc] === stone) {
                        continue;
                    }

                    let length = 1;
                    let nr = r + dr;
                    let nc = c + dc;
                    while (nr >= 0 && nr < size && nc >= 0 && nc < size && board[nr * size + nc] === stone) {
                        length++;
                        nr += dr;
                        nc += dc;
                    }

                    if (stone === 1 && length === 5) {
                        return 1;
                    }
                    if (stone === -1 && length >= 5) {
                        return -1;
                    }
                }
            }
        }

        if (board.every(x => x !== 0)) return 0; // Draw
        return null;
    }

    /**
     * Encode state for NN input: 4 planes [my_stones, opp_stones, black_forbidden, white_forbidden].
     * Only one forbidden plane is populated based on toPlay.
     */
    encodeState(state, toPlay) {
        const layerSize = this.boardSize * this.boardSize;
        const encoded = new Float32Array(this.numPlanes * layerSize);
        const board = state[0];

        for (let j = 0; j < layerSize; j++) {
            if (board[j] === toPlay) {
                encoded[j] = 1;
            } else if (board[j] === -toPlay) {
                encoded[layerSize + j] = 1;
            }
        }

        if (this.useRenju) {
            this.fpf.clear();
            for (let i = 0; i < board.length; i++) {
                if (board[i] !== 0) {
                    const r = Math.floor(i / this.boardSize);
                    const c = i % this.boardSize;
                    this.fpf.setStone(r, c, board[i] === 1 ? C_BLACK : C_WHITE);
                }
            }

            const forbiddenPlaneOffset = (toPlay === 1 ? 2 : 3) * layerSize;
            for (let i = 0; i < board.length; i++) {
                if (board[i] !== 0) {
                    continue;
                }
                const r = Math.floor(i / this.boardSize);
                const c = i % this.boardSize;
                if (this.fpf.isForbidden(r, c)) {
                    encoded[forbiddenPlaneOffset + i] = 1;
                }
            }
        }

        return encoded;
    }
}

if (typeof module !== "undefined") {
    module.exports = { Gomoku, ForbiddenPointFinder };
}
