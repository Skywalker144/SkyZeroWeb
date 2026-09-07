(function (scope) {
    const TRAINING_WIDTH = 450;
    const TRAINING_HEIGHT = 600;
    const POSITION_SCALE = 250;
    const VELOCITY_SCALE = 300;
    const DAMAGE_SCALE = 30;
    const RADIUS_SCALE = 20;
    const BLAST_SCALE = 150;
    const LASER_CHARGE = 1;
    const LASER_FIRE = 0.55;
    const BOMB_FUSE = 1.8;
    const THREAT_COUNT = 32;
    const LASER_COUNT = 3;
    const BOMB_COUNT = 3;
    const PICKUP_COUNT = 2;
    const THREAT_FEATURES = 7;
    const LASER_FEATURES = 6;
    const BOMB_FEATURES = 5;
    const PICKUP_FEATURES = 3;
    const PLAYER_FEATURES = 8;
    const OBSERVATION_SIZE = THREAT_COUNT * THREAT_FEATURES
        + LASER_COUNT * LASER_FEATURES
        + BOMB_COUNT * BOMB_FEATURES
        + PICKUP_COUNT * PICKUP_FEATURES
        + PLAYER_FEATURES;
    const DECISION_STEP = 1 / 30;

    function clamp(value, lower, upper) {
        return value < lower ? lower : value > upper ? upper : value;
    }

    function buildObservation(state, types, {
        maxHp = 100,
        hitInvuln = 0.6,
    } = {}) {
        const player = state.player;
        const scaleX = TRAINING_WIDTH / state.W;
        const scaleY = TRAINING_HEIGHT / state.H;
        const playerX = player.x * scaleX;
        const playerY = player.y * scaleY;
        const output = new Float32Array(OBSERVATION_SIZE);
        const threats = state.projectiles.map(projectile => {
            const type = types[projectile.kind];
            return [
                (projectile.x - player.x) * scaleX,
                (projectile.y - player.y) * scaleY,
                projectile.vx * scaleX,
                projectile.vy * scaleY,
                type.dmg / DAMAGE_SCALE,
                type.r / RADIUS_SCALE,
            ];
        }).sort((left, right) =>
            left[0] ** 2 + left[1] ** 2 - right[0] ** 2 - right[1] ** 2);
        let offset = 0;
        for (let index = 0; index < Math.min(THREAT_COUNT, threats.length); index++) {
            const values = threats[index];
            output[offset] = clamp(values[0] / POSITION_SCALE, -1, 1);
            output[offset + 1] = clamp(values[1] / POSITION_SCALE, -1, 1);
            output[offset + 2] = clamp(values[2] / VELOCITY_SCALE, -1, 1);
            output[offset + 3] = clamp(values[3] / VELOCITY_SCALE, -1, 1);
            output[offset + 4] = values[4];
            output[offset + 5] = values[5];
            output[offset + 6] = 1;
            offset += THREAT_FEATURES;
        }
        offset = THREAT_COUNT * THREAT_FEATURES;
        const lasers = state.lasers.map(laser => {
            const cosine = Math.cos(laser.ang);
            const sine = Math.sin(laser.ang);
            const perpendicular = (player.x - laser.ox) * sine
                - (player.y - laser.oy) * cosine;
            return [
                Math.abs(perpendicular),
                perpendicular,
                cosine,
                sine,
                laser.phase === "charge" ? laser.charge : 0,
                laser.phase === "fire" ? laser.fire : 0,
            ];
        }).sort((left, right) => left[0] - right[0]);
        for (let index = 0; index < Math.min(LASER_COUNT, lasers.length); index++) {
            const values = lasers[index];
            output[offset] = clamp(values[1] * scaleX / POSITION_SCALE, -1, 1);
            output[offset + 1] = values[2];
            output[offset + 2] = values[3];
            output[offset + 3] = clamp(values[4] / LASER_CHARGE, 0, 1);
            output[offset + 4] = clamp(values[5] / LASER_FIRE, 0, 1);
            output[offset + 5] = 1;
            offset += LASER_FEATURES;
        }
        offset = THREAT_COUNT * THREAT_FEATURES + LASER_COUNT * LASER_FEATURES;
        const bombs = state.bombs.map(bomb => {
            const dx = bomb.x - player.x;
            const dy = bomb.y - player.y;
            return [
                dx ** 2 + dy ** 2,
                dx,
                dy,
                bomb.blastR,
                bomb.phase === "fuse" ? Math.max(0, bomb.fuse) : 0,
            ];
        }).sort((left, right) => left[0] - right[0]);
        for (let index = 0; index < Math.min(BOMB_COUNT, bombs.length); index++) {
            const values = bombs[index];
            output[offset] = clamp(values[1] * scaleX / POSITION_SCALE, -1, 1);
            output[offset + 1] = clamp(values[2] * scaleY / POSITION_SCALE, -1, 1);
            output[offset + 2] = clamp(values[3] * scaleX / BLAST_SCALE, 0, 1);
            output[offset + 3] = clamp(values[4] / BOMB_FUSE, 0, 1);
            output[offset + 4] = 1;
            offset += BOMB_FEATURES;
        }
        offset = THREAT_COUNT * THREAT_FEATURES
            + LASER_COUNT * LASER_FEATURES
            + BOMB_COUNT * BOMB_FEATURES;
        const pickups = state.pickups.map(pickup => [
            (pickup.x - player.x) * scaleX,
            (pickup.y - player.y) * scaleY,
        ]).sort((left, right) =>
            left[0] ** 2 + left[1] ** 2 - right[0] ** 2 - right[1] ** 2);
        for (let index = 0; index < Math.min(PICKUP_COUNT, pickups.length); index++) {
            output[offset] = clamp(pickups[index][0] / POSITION_SCALE, -1, 1);
            output[offset + 1] = clamp(pickups[index][1] / POSITION_SCALE, -1, 1);
            output[offset + 2] = 1;
            offset += PICKUP_FEATURES;
        }
        offset = THREAT_COUNT * THREAT_FEATURES
            + LASER_COUNT * LASER_FEATURES
            + BOMB_COUNT * BOMB_FEATURES
            + PICKUP_COUNT * PICKUP_FEATURES;
        output[offset] = player.hp / maxHp;
        output[offset + 1] = Math.min(1, player.invuln / hitInvuln);
        output[offset + 2] = playerX / TRAINING_WIDTH;
        output[offset + 3] = playerY / TRAINING_HEIGHT;
        output[offset + 4] = clamp(playerX / POSITION_SCALE, 0, 1);
        output[offset + 5] = clamp((TRAINING_WIDTH - playerX) / POSITION_SCALE, 0, 1);
        output[offset + 6] = clamp(playerY / POSITION_SCALE, 0, 1);
        output[offset + 7] = clamp((TRAINING_HEIGHT - playerY) / POSITION_SCALE, 0, 1);
        return output;
    }

    function decodeWeights(encoded) {
        const binary = scope.atob(encoded);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index++) {
            bytes[index] = binary.charCodeAt(index);
        }
        return new Float32Array(bytes.buffer);
    }

    function expectedWeightCount(architecture) {
        return architecture.reduce(
            (total, [inputs, outputs]) => total + inputs * outputs + outputs,
            0);
    }

    function forward(weights, architecture, input) {
        let layerInput = input;
        let offset = 0;
        for (const [inputs, outputs, activation] of architecture) {
            const layerOutput = new Float32Array(outputs);
            for (let output = 0; output < outputs; output++) {
                let value = 0;
                const row = offset + output * inputs;
                for (let index = 0; index < inputs; index++) {
                    value += weights[row + index] * layerInput[index];
                }
                layerOutput[output] = value;
            }
            offset += outputs * inputs;
            for (let output = 0; output < outputs; output++) {
                layerOutput[output] += weights[offset + output];
            }
            offset += outputs;
            if (activation === "tanh") {
                for (let output = 0; output < outputs; output++) {
                    layerOutput[output] = Math.tanh(layerOutput[output]);
                }
            } else if (activation === "relu") {
                for (let output = 0; output < outputs; output++) {
                    if (layerOutput[output] < 0) layerOutput[output] = 0;
                }
            } else if (activation !== "linear") {
                throw new Error(`unsupported activation: ${activation}`);
            }
            layerInput = layerOutput;
        }
        return layerInput;
    }

    function createPolicyController({
        policy,
        getState,
        types,
        maxHp = 100,
        hitInvuln = 0.6,
    }) {
        let weights = null;
        let elapsed = 0;
        const move = { x: 0, y: 0 };
        function load() {
            if (weights) return true;
            try {
                if (!policy || !policy.continuous || policy.actDim !== 2
                    || policy.obsDim !== OBSERVATION_SIZE
                    || !Array.isArray(policy.arch) || !policy.weightsB64) return false;
                const decoded = decodeWeights(policy.weightsB64);
                if (decoded.length !== expectedWeightCount(policy.arch)) return false;
                const outputSize = policy.arch[policy.arch.length - 1][1];
                if (outputSize < 2) return false;
                weights = decoded;
                return true;
            } catch (_) {
                weights = null;
                return false;
            }
        }
        function tick(delta) {
            const state = getState();
            if (!weights || !state.W) {
                move.x = 0;
                move.y = 0;
                return;
            }
            elapsed += delta;
            if (elapsed < DECISION_STEP) return;
            elapsed %= DECISION_STEP;
            const output = forward(
                weights,
                policy.arch,
                buildObservation(state, types, { maxHp, hitInvuln }),
            );
            let x = Math.tanh(output[0]);
            let y = Math.tanh(output[1]);
            const magnitude = Math.hypot(x, y);
            if (magnitude > 1) {
                x /= magnitude;
                y /= magnitude;
            }
            move.x = x;
            move.y = y;
        }
        function reset() {
            move.x = 0;
            move.y = 0;
            elapsed = 0;
        }
        return {
            load,
            tick,
            reset,
            move,
            get available() { return !!policy?.weightsB64; },
            get version() { return policy?.version || ""; },
        };
    }

    const api = {
        OBSERVATION_SIZE,
        buildObservation,
        createPolicyController,
        forward,
    };
    scope.SkyZeroDodgeAI = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
