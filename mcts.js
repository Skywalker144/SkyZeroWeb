class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];
        this.nnValue = new Float64Array([0, 1, 0]);
        this.nnPolicy = null;
        this.nnLogits = null;
        this.nnStError = -1;
        this.v = new Float64Array(3);
        this.vRaw = new Float64Array(3);
        this.weightSum = 0;
        this.weightSqSum = 0;
        this.utilitySqSum = 0;
        this.n = 0;
        this.materializationOrder = 0;
        this.nextMaterializationOrder = 1;
        this.biasKey = null;
        this.biasLastDelta = 0;
        this.biasLastWeight = 0;
        this.rootPolicyApplied = false;
    }

    isExpanded() {
        return this.children.length > 0;
    }

    updateTerminal(value, weight, drawUtility = 0) {
        for (let i = 0; i < 3; i++) {
            const contribution = weight * value[i];
            this.v[i] += contribution;
            this.vRaw[i] += contribution;
        }
        const utility = wdlUtility(value, drawUtilityInFrame(
            drawUtility, this.toPlay));
        this.weightSum += weight;
        this.weightSqSum += weight * weight;
        this.utilitySqSum += weight * utility * utility;
        this.n += 1;
    }
}

function softmax(logits) {
    let maximum = -Infinity;
    for (const value of logits) {
        if (Number.isFinite(value) && value > maximum) maximum = value;
    }
    const result = new Float64Array(logits.length);
    if (!Number.isFinite(maximum)) return result;
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        if (!Number.isFinite(logits[i])) continue;
        result[i] = Math.exp(logits[i] - maximum);
        sum += result[i];
    }
    if (sum > 0) {
        for (let i = 0; i < result.length; i++) result[i] /= sum;
    }
    return result;
}

function flipWdl(value) {
    return new Float64Array([value[2], value[1], value[0]]);
}

// KataGo PlayUtils::getSearchFactor, specialized to values already expressed
// from the fixed AI player's perspective. Only reduce search when the least
// winning of the last three completed AI move-searches is above the threshold.
function searchFactorWhenWinning(
    recentWinLossValues,
    threshold = 0.95,
    minimumFactor = 0.30
) {
    if (recentWinLossValues.length < 3 || threshold >= 1) return 1;
    const leastWinning = Math.min(...recentWinLossValues.slice(-3));
    if (leastWinning <= threshold) return 1;
    const lambda = Math.min(
        1, (leastWinning - threshold) / (1 - threshold));
    return 1 + lambda * (minimumFactor - 1);
}

function drawUtilityInFrame(drawUtility, toPlay) {
    return toPlay === 1 ? -drawUtility : drawUtility;
}

function wdlUtility(value, drawInFrame = 0) {
    return value[0] - value[2] + drawInFrame * value[1];
}

function uncertaintyWeight(stError, args) {
    if (!args.use_uncertainty || stError < 0) return 1;
    const uncertainty = Math.max(0, stError);
    const powered = args.uncertainty_exponent === 1
        ? uncertainty
        : Math.pow(uncertainty, args.uncertainty_exponent);
    const baseline = args.uncertainty_coeff
        / args.uncertainty_max_weight;
    return args.uncertainty_coeff / (powered + baseline);
}

function studentTCdfThree(z) {
    const root3 = Math.sqrt(3);
    return 0.5 + (
        Math.atan(z / root3)
        + root3 * z / (z * z + 3)
    ) / Math.PI;
}

// KataGo/SkyZero does not query the analytic CDF directly: it samples 2000
// points on [-50,50], pins the endpoints, then linearly interpolates.
let valueWeightTable = null;
function valueWeightCdf(z) {
    const tableSize = 2000;
    const minZ = -50;
    const maxZ = 50;
    if (valueWeightTable === null) {
        valueWeightTable = new Float64Array(tableSize);
        valueWeightTable[0] = 0;
        valueWeightTable[tableSize - 1] = 1;
        for (let i = 1; i < tableSize - 1; i++) {
            valueWeightTable[i] = studentTCdfThree(
                minZ + i * (maxZ - minZ) / (tableSize - 1));
        }
    }
    const position = (tableSize - 1) * (z - minZ) / (maxZ - minZ);
    if (position <= 0) return 0;
    const index = Math.floor(position);
    if (index >= tableSize - 1) return 1;
    const lambda = position - index;
    return valueWeightTable[index]
        + lambda * (valueWeightTable[index + 1] - valueWeightTable[index]);
}

function transformCoord(row, col, size, rotation, flip) {
    let r = row;
    let c = col;
    for (let k = 0; k < rotation; k++) {
        const nextR = c;
        const nextC = size - 1 - r;
        r = nextR;
        c = nextC;
    }
    if (flip) c = size - 1 - c;
    return [r, c];
}

function boardSymmetries(state, size) {
    const symmetries = [];
    for (let flip = 0; flip <= 1; flip++) {
        for (let rotation = 0; rotation < 4; rotation++) {
            let equal = true;
            for (let r = 0; r < size && equal; r++) {
                for (let c = 0; c < size; c++) {
                    const [tr, tc] = transformCoord(
                        r, c, size, rotation, Boolean(flip));
                    if (state[r * size + c] !== state[tr * size + tc]) {
                        equal = false;
                        break;
                    }
                }
            }
            if (equal) symmetries.push([rotation, Boolean(flip)]);
        }
    }
    return symmetries;
}

function rootSymmetryMask(state, size, toPlay) {
    const symmetries = boardSymmetries(state, size);
    if (symmetries.length <= 1) return null;
    const mask = new Uint8Array(size * size);
    const duplicate = new Uint8Array(size * size);
    const keep = (row, col) => {
        const action = row * size + col;
        if (duplicate[action]) return;
        mask[action] = 1;
        for (const [rotation, flip] of symmetries) {
            const [tr, tc] = transformCoord(
                row, col, size, rotation, flip);
            const transformed = tr * size + tc;
            if (transformed !== action) duplicate[transformed] = 1;
        }
    };
    if (toPlay === 1) {
        for (let col = size - 1; col >= 0; col--) {
            for (let row = 0; row < size; row++) keep(row, col);
        }
    } else {
        for (let col = 0; col < size; col++) {
            for (let row = size - 1; row >= 0; row--) keep(row, col);
        }
    }
    return mask;
}

function isEmptyBoard(state) {
    for (const value of state) if (value !== 0) return false;
    return true;
}

class MCTS {
    constructor(game, args) {
        this.game = game;
        this.args = Object.assign({
            c_puct: 1.1,
            c_puct_log: 0.45,
            c_puct_base: 500,
            fpu_reduction_max: 0.16,
            root_fpu_reduction_max: 0.1,
            fpu_pow: 2,
            fpu_loss_prop: 0,
            root_fpu_loss_prop: 0,
            cpuct_utility_stdev_prior: 0.40,
            cpuct_utility_stdev_prior_weight: 2,
            cpuct_utility_stdev_scale: 0.85,
            draw_utility: 0,
            use_uncertainty: true,
            uncertainty_coeff: 0.25,
            uncertainty_exponent: 1,
            uncertainty_max_weight: 8,
            value_weight_exponent: 0.5,
            subtree_value_bias_factor: 0.45,
            subtree_value_bias_weight_exponent: 0.8,
            subtree_value_bias_free_prop: 0.8,
            root_lcb_selection: true,
            lcb_stdevs: 5,
            min_visit_prop_for_lcb: 0.15,
            root_symmetry_pruning: true,
            chosen_move_subtract: 0,
            chosen_move_prune: 1,
        }, args);
        this.biasTable = new Map();
    }

    computeParentUtilityStdevFactor(node, parentUtility) {
        const prior = this.args.cpuct_utility_stdev_prior;
        const priorWeight = this.args.cpuct_utility_stdev_prior_weight;
        let stdev = prior;
        if (node.n > 0 && node.weightSum > 1) {
            const utilitySq = parentUtility * parentUtility;
            const observed = Math.max(
                node.utilitySqSum / node.weightSum, utilitySq);
            const variance = Math.max(0,
                ((utilitySq + prior * prior) * priorWeight
                    + observed * node.weightSum)
                / (priorWeight + node.weightSum - 1)
                - utilitySq);
            stdev = Math.sqrt(variance);
        }
        return 1 + this.args.cpuct_utility_stdev_scale
            * (stdev / prior - 1);
    }

    computeSelectParams(node, isRoot = false) {
        let totalChildWeight = 0;
        let visitedPolicyMass = 0;
        for (const child of node.children) {
            totalChildWeight += child.weightSum;
            if (child.n > 0) visitedPolicyMass += child.prior;
        }
        const cPuct = this.args.c_puct
            + this.args.c_puct_log * Math.log(
                (totalChildWeight + this.args.c_puct_base)
                / this.args.c_puct_base);
        const parentValue = node.weightSum > 0
            ? new Float64Array([
                node.v[0] / node.weightSum,
                node.v[1] / node.weightSum,
                node.v[2] / node.weightSum,
            ])
            : new Float64Array([0, 1, 0]);
        const draw = drawUtilityInFrame(
            this.args.draw_utility, node.toPlay);
        const parentUtility = wdlUtility(parentValue, draw);
        const stdevFactor = this.computeParentUtilityStdevFactor(
            node, parentUtility);
        const exploreScaling = cPuct
            * Math.sqrt(totalChildWeight + 0.01) * stdevFactor;
        const nnUtility = wdlUtility(node.nnValue, draw);
        const blend = Math.min(
            1, Math.pow(visitedPolicyMass, this.args.fpu_pow));
        let fpuValue = blend * parentUtility
            + (1 - blend) * nnUtility;
        const reductionMax = isRoot
            ? this.args.root_fpu_reduction_max
            : this.args.fpu_reduction_max;
        fpuValue -= reductionMax * Math.sqrt(visitedPolicyMass);
        const lossProp = isRoot
            ? this.args.root_fpu_loss_prop
            : this.args.fpu_loss_prop;
        fpuValue += (-1 - fpuValue) * lossProp;
        return {exploreScaling, fpuValue};
    }

    select(node) {
        const params = this.computeSelectParams(
            node, node.parent === null);
        let best = null;
        let bestScore = -Infinity;
        for (const child of node.children) {
            const q = child.weightSum > 0
                ? wdlUtility(new Float64Array([
                    child.v[2] / child.weightSum,
                    child.v[1] / child.weightSum,
                    child.v[0] / child.weightSum,
                ]), drawUtilityInFrame(
                    this.args.draw_utility, node.toPlay))
                : params.fpuValue;
            const explore = params.exploreScaling * child.prior
                / (1 + child.weightSum);
            const score = q + explore;
            if (score > bestScore) {
                bestScore = score;
                best = child;
            }
        }
        return best;
    }

    candidateMask(node, isRoot) {
        if (!isRoot) return null;
        const area = this.game.boardSize * this.game.boardSize;
        if (isEmptyBoard(node.state)) {
            const mask = new Uint8Array(area);
            const center = Math.floor(this.game.boardSize / 2);
            mask[center * this.game.boardSize + center] = 1;
            return mask;
        }
        if (this.args.root_symmetry_pruning) {
            return rootSymmetryMask(
                node.state, this.game.boardSize, node.toPlay);
        }
        return null;
    }

    expand(node, nnPolicy, nnValue, nnLogits, nnStError, isRoot = false) {
        node.nnValue = new Float64Array(nnValue);
        node.nnPolicy = new Float64Array(nnPolicy);
        node.nnLogits = new Float64Array(nnLogits);
        node.nnStError = nnStError;
        this.attachBias(node);
        const legal = this.game.getLegalActions(node.state, node.toPlay);
        const candidate = this.candidateMask(node, isRoot);
        const existing = new Map(
            node.children.map(child => [child.actionTaken, child]));
        const next = [];
        for (let action = 0; action < legal.length; action++) {
            if (!legal[action]
                || (candidate && !candidate[action])) continue;
            let child = existing.get(action);
            if (child) {
                child.prior = nnPolicy[action];
                existing.delete(action);
            } else {
                child = new Node(
                    this.game.getNextState(
                        node.state, action, node.toPlay),
                    -node.toPlay,
                    nnPolicy[action],
                    node,
                    action);
                child.materializationOrder =
                    node.nextMaterializationOrder++;
            }
            next.push(child);
        }
        for (const child of existing.values()) this.releaseTree(child);
        node.children = next;
        node.rootPolicyApplied = isRoot;
        this.recomputeNode(node);
    }

    refreshRoot(node, nnPolicy, nnValue, nnLogits, nnStError) {
        this.expand(
            node, nnPolicy, nnValue, nnLogits, nnStError, true);
    }

    attachBias(node) {
        if (!(this.args.subtree_value_bias_factor !== 0)
            || node.biasKey !== null
            || !node.parent
            || node.parent.actionTaken == null) return;
        const parent = node.parent;
        const size = this.game.boardSize;
        const action = node.actionTaken;
        const row = Math.floor(action / size);
        const col = action % size;
        const cells = [];
        for (let dr = -2; dr <= 2; dr++) {
            for (let dc = -2; dc <= 2; dc++) {
                const r = row + dr;
                const c = col + dc;
                if (r < 0 || r >= size || c < 0 || c >= size) continue;
                cells.push(parent.state[r * size + c]);
            }
        }
        node.biasKey = [
            -node.toPlay,
            action,
            parent.actionTaken,
            cells.join(","),
        ].join("|");
        let entry = this.biasTable.get(node.biasKey);
        if (!entry) {
            entry = {deltaSum: 0, weightSum: 0, refs: 0};
            this.biasTable.set(node.biasKey, entry);
        }
        entry.refs++;
    }

    releaseBias(node) {
        if (node.biasKey === null) return;
        const entry = this.biasTable.get(node.biasKey);
        if (entry) {
            const free = this.args.subtree_value_bias_free_prop;
            entry.deltaSum -= free * node.biasLastDelta;
            entry.weightSum -= free * node.biasLastWeight;
            entry.refs--;
            if (entry.refs <= 0) this.biasTable.delete(node.biasKey);
        }
        node.biasKey = null;
        node.biasLastDelta = 0;
        node.biasLastWeight = 0;
    }

    releaseTree(node, except = null) {
        if (!node || node === except) return;
        for (const child of node.children) {
            if (child !== except) this.releaseTree(child);
        }
        this.releaseBias(node);
    }

    clear(root = null) {
        if (root) this.releaseTree(root);
        this.biasTable.clear();
    }

    recomputeNode(node) {
        const ownWeight = uncertaintyWeight(
            node.nnStError, this.args);
        const draw = drawUtilityInFrame(
            this.args.draw_utility, node.toPlay);
        const ownRaw = new Float64Array(node.nnValue);
        let ownValue = new Float64Array(ownRaw);
        const ownRawUtility = wdlUtility(ownRaw, draw);
        let ownUtility = ownRawUtility;
        const children = [];
        let totalChildWeight = 0;
        let visits = 1;
        for (const child of node.children) {
            if (!(child.weightSum > 0)) continue;
            const value = flipWdl(new Float64Array([
                child.v[0] / child.weightSum,
                child.v[1] / child.weightSum,
                child.v[2] / child.weightSum,
            ]));
            const valueRaw = flipWdl(new Float64Array([
                child.vRaw[0] / child.weightSum,
                child.vRaw[1] / child.weightSum,
                child.vRaw[2] / child.weightSum,
            ]));
            children.push({
                value,
                valueRaw,
                utility: wdlUtility(value, draw),
                weight: child.weightSum,
                originalWeight: child.weightSum,
                weightSq: child.weightSqSum,
                utilitySqAverage:
                    child.utilitySqSum / child.weightSum,
            });
            totalChildWeight += child.weightSum;
            visits += child.n;
        }

        if (children.length > 0
            && totalChildWeight > 0
            && this.args.value_weight_exponent > 0) {
            let simpleUtility = 0;
            for (const child of children) {
                simpleUtility += child.utility * child.weight;
            }
            simpleUtility /= totalChildWeight;
            let adjustedTotal = 0;
            for (const child of children) {
                const precision = 1.5 * Math.sqrt(child.weight);
                const stdev = Math.sqrt(1e-8 + 1 / precision);
                const z = (child.utility - simpleUtility) / stdev;
                const probability = valueWeightCdf(z) + 1e-4;
                child.weight *= Math.pow(
                    probability, this.args.value_weight_exponent);
                adjustedTotal += child.weight;
            }
            const scale = totalChildWeight / adjustedTotal;
            for (const child of children) child.weight *= scale;
        }

        if (node.biasKey !== null) {
            const entry = this.biasTable.get(node.biasKey);
            if (entry) {
                if (children.length > 0) {
                    let childWeight = 0;
                    let childUtility = 0;
                    for (const child of children) {
                        childWeight += child.weight;
                        childUtility += child.weight * child.utility;
                    }
                    const contributionWeight = Math.pow(
                        totalChildWeight,
                        this.args.subtree_value_bias_weight_exponent);
                    const contributionDelta = (
                        childUtility / childWeight - ownRawUtility
                    ) * contributionWeight;
                    entry.deltaSum += contributionDelta
                        - node.biasLastDelta;
                    entry.weightSum += contributionWeight
                        - node.biasLastWeight;
                    node.biasLastDelta = contributionDelta;
                    node.biasLastWeight = contributionWeight;
                }
                if (entry.weightSum > 0.001) {
                    ownUtility +=
                        this.args.subtree_value_bias_factor
                        * entry.deltaSum / entry.weightSum;
                    const drawProbability = ownRaw[1];
                    const winLoss = ownUtility
                        - draw * drawProbability;
                    ownValue = new Float64Array([
                        (1 - drawProbability + winLoss) * 0.5,
                        drawProbability,
                        (1 - drawProbability - winLoss) * 0.5,
                    ]);
                }
            }
        }

        const valueSum = new Float64Array([
            ownWeight * ownValue[0],
            ownWeight * ownValue[1],
            ownWeight * ownValue[2],
        ]);
        const rawSum = new Float64Array([
            ownWeight * ownRaw[0],
            ownWeight * ownRaw[1],
            ownWeight * ownRaw[2],
        ]);
        let weightSum = ownWeight;
        let weightSqSum = ownWeight * ownWeight;
        let utilitySqSum = ownWeight * ownUtility * ownUtility;
        for (const child of children) {
            for (let i = 0; i < 3; i++) {
                valueSum[i] += child.weight * child.value[i];
                rawSum[i] += child.weight * child.valueRaw[i];
            }
            weightSum += child.weight;
            const scaling = child.weight / child.originalWeight;
            weightSqSum += scaling * scaling * child.weightSq;
            utilitySqSum += child.weight
                * child.utilitySqAverage;
        }
        node.v = valueSum;
        node.vRaw = rawSum;
        node.weightSum = weightSum;
        node.weightSqSum = weightSqSum;
        node.utilitySqSum = utilitySqSum;
        node.n = visits;
    }

    backpropagate(path, value, terminal = false) {
        if (path.length === 0) return;
        const leaf = path[path.length - 1];
        if (terminal) {
            leaf.updateTerminal(
                value,
                uncertaintyWeight(0, this.args),
                this.args.draw_utility);
        } else {
            this.recomputeNode(leaf);
        }
        for (let i = path.length - 2; i >= 0; i--) {
            this.recomputeNode(path[i]);
        }
    }

    childStats(root) {
        const draw = drawUtilityInFrame(
            this.args.draw_utility, root.toPlay);
        return root.children.map(child => {
            const value = child.weightSum > 0
                ? new Float64Array([
                    child.v[2] / child.weightSum,
                    child.v[1] / child.weightSum,
                    child.v[0] / child.weightSum,
                ])
                : new Float64Array([0, 1, 0]);
            return {
                child,
                action: child.actionTaken,
                n: child.n,
                weight: child.weightSum,
                weightSq: child.weightSqSum,
                prior: child.prior,
                utilitySqSum: child.utilitySqSum,
                q: wdlUtility(value, draw),
                value,
            };
        });
    }

    rootPlaySelection(root) {
        const stats = this.childStats(root);
        if (stats.length === 0) {
            return {
                action: -1,
                weights: new Float64Array(
                    this.game.boardSize ** 2),
                lcb: new Float64Array(this.game.boardSize ** 2),
                radius: new Float64Array(this.game.boardSize ** 2),
            };
        }
        const params = this.computeSelectParams(root, true);
        let bestIndex = 0;
        let bestGoodness = -Infinity;
        for (let i = 0; i < stats.length; i++) {
            const stat = stats[i];
            const discounted = stat.n > 0
                ? stat.weight * Math.max(0, stat.n - 1)
                    / Math.max(1, stat.n)
                : 0;
            const goodness = discounted + 2 * stat.prior;
            if (goodness > bestGoodness) {
                bestGoodness = goodness;
                bestIndex = i;
            }
        }
        const best = stats[bestIndex];
        const bestValue = best.q
            + params.exploreScaling * best.prior
                / (1 + best.weight);
        const weights = new Float64Array(stats.length);
        for (let i = 0; i < stats.length; i++) {
            const stat = stats[i];
            if (stat.n <= 0) continue;
            if (i === bestIndex) {
                weights[i] = stat.weight;
                continue;
            }
            const denominator = bestValue - stat.q;
            let wanted = stat.weight;
            if (denominator > 1e-12) {
                wanted = params.exploreScaling * stat.prior
                    / denominator - 1;
            }
            weights[i] = Math.ceil(Math.max(
                0, Math.min(wanted, stat.weight)));
        }
        let sum = weights.reduce((a, b) => a + b, 0);
        if (!(sum > 0)) {
            for (let i = 0; i < stats.length; i++) {
                weights[i] = Math.max(0, stats[i].prior);
            }
        }

        const lcb = new Float64Array(stats.length);
        const radius = new Float64Array(stats.length);
        for (let i = 0; i < stats.length; i++) {
            const stat = stats[i];
            radius[i] = 2 * this.args.lcb_stdevs;
            lcb[i] = -radius[i];
            if (!(stat.n > 0)
                || !(stat.weight > 0)
                || !(stat.weightSq > 0)) continue;
            let weightSum = stat.weight;
            let weightSq = stat.weightSq;
            let ess = weightSum * weightSum / weightSq;
            const utilityAverage = stat.q;
            let utilitySqAverage = Math.max(
                stat.utilitySqSum / weightSum,
                utilityAverage * utilityAverage + 1e-8);
            const priorWeight = weightSum / (ess * ess * ess);
            utilitySqAverage = (
                utilitySqAverage * weightSum
                + (utilitySqAverage + 1) * priorWeight
            ) / (weightSum + priorWeight);
            weightSum += priorWeight;
            weightSq += priorWeight * priorWeight;
            ess = weightSum * weightSum / weightSq;
            const variance = Math.max(
                0, utilitySqAverage
                    - utilityAverage * utilityAverage);
            radius[i] = Math.sqrt(variance / ess)
                * this.args.lcb_stdevs;
            lcb[i] = utilityAverage - radius[i];
        }

        const adjusted = new Float64Array(weights);
        if (this.args.root_lcb_selection) {
            let bestLcb = -Infinity;
            let bestLcbIndex = -1;
            for (let i = 0; i < stats.length; i++) {
                if (weights[i] > 0
                    && weights[i] >=
                        this.args.min_visit_prop_for_lcb
                        * best.weight
                    && lcb[i] > bestLcb) {
                    bestLcb = lcb[i];
                    bestLcbIndex = i;
                }
            }
            if (bestLcbIndex >= 0) {
                let value = weights[bestLcbIndex];
                for (let i = 0; i < stats.length; i++) {
                    if (i === bestLcbIndex) continue;
                    const excess = bestLcb - lcb[i];
                    if (excess < 0) continue;
                    const factor = (radius[i] + excess)
                        / (radius[i] + 0.2 * excess);
                    value = Math.max(
                        value, factor * factor * weights[i]);
                }
                adjusted[bestLcbIndex] = value;
            }
        }

        const maxAdjusted = adjusted.reduce(
            (maximum, value) => Math.max(maximum, value), 0);
        if (maxAdjusted > 0) {
            const subtract = Math.min(
                this.args.chosen_move_subtract,
                maxAdjusted / 64);
            const prune = Math.min(
                this.args.chosen_move_prune,
                maxAdjusted / 64);
            for (let i = 0; i < adjusted.length; i++) {
                adjusted[i] = adjusted[i] < prune
                    ? 0 : Math.max(0, adjusted[i] - subtract);
            }
        }

        let chosen = bestIndex;
        for (let i = 0; i < adjusted.length; i++) {
            if (adjusted[i] > adjusted[chosen]) chosen = i;
        }
        const area = this.game.boardSize ** 2;
        const actionWeights = new Float64Array(area);
        const actionLcb = new Float64Array(area).fill(NaN);
        const actionRadius = new Float64Array(area).fill(NaN);
        for (let i = 0; i < stats.length; i++) {
            actionWeights[stats[i].action] = adjusted[i];
            actionLcb[stats[i].action] = lcb[i];
            actionRadius[stats[i].action] = radius[i];
        }
        return {
            action: stats[chosen].action,
            weights: actionWeights,
            lcb: actionLcb,
            radius: actionRadius,
        };
    }

    getMCTSPolicy(root) {
        const policy = new Float32Array(
            this.game.boardSize * this.game.boardSize);
        let sum = 0;
        for (const child of root.children) {
            policy[child.actionTaken] = child.n;
            sum += child.n;
        }
        if (sum > 0) {
            for (let i = 0; i < policy.length; i++) {
                policy[i] /= sum;
            }
        }
        return this.duplicateRootValues(root, policy, 0);
    }

    getPlaySelectionPolicy(root) {
        const result = this.rootPlaySelection(root);
        let sum = result.weights.reduce((a, b) => a + b, 0);
        if (!(sum > 0)) sum = 1;
        const policy = new Float32Array(result.weights.length);
        for (let i = 0; i < policy.length; i++) {
            policy[i] = result.weights[i] / sum;
        }
        return this.duplicateRootValues(root, policy, 0);
    }

    getMCTSWinrate(root) {
        const values = new Float32Array(
            this.game.boardSize * this.game.boardSize).fill(NaN);
        for (const stat of this.childStats(root)) {
            if (stat.n > 0) {
                values[stat.action] = (stat.q + 1) * 0.5;
            }
        }
        return this.duplicateRootValues(root, values, NaN);
    }

    getLcb(root) {
        const result = this.rootPlaySelection(root);
        return {
            lcb: this.duplicateRootValues(root, result.lcb, NaN),
            radius: this.duplicateRootValues(
                root, result.radius, NaN),
        };
    }

    duplicateRootValues(root, values, emptyValue) {
        if (!this.args.root_symmetry_pruning) return values;
        const symmetries = boardSymmetries(
            root.state, this.game.boardSize);
        if (symmetries.length <= 1) return values;
        const output = new values.constructor(values);
        const mask = rootSymmetryMask(
            root.state, this.game.boardSize, root.toPlay);
        for (let action = 0; action < mask.length; action++) {
            if (!mask[action]) continue;
            const row = Math.floor(action / this.game.boardSize);
            const col = action % this.game.boardSize;
            for (const [rotation, flip] of symmetries) {
                const [tr, tc] = transformCoord(
                    row, col, this.game.boardSize, rotation, flip);
                output[tr * this.game.boardSize + tc] =
                    values[action] ?? emptyValue;
            }
        }
        return output;
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        Node,
        MCTS,
        softmax,
        flipWdl,
        uncertaintyWeight,
        valueWeightCdf,
        searchFactorWhenWinning,
        boardSymmetries,
        rootSymmetryMask,
    };
}
