# SkyZeroWeb

聚合游戏平台，最初是 AI 五子棋页面，随后加入了多个自训练 AI。游戏逻辑和 AI 推理全部在浏览器本地运行；仅排行榜使用 Cloudflare Pages Functions 和 D1，不存在后端推理服务器。

## 当前游戏

| 游戏 | 页面 | AI | 主要入口 |
|---|---|---|---|
| 五子棋 | `/gomoku` | AlphaZero 风格神经网络与 MCTS | `gomoku.html`、`gomoku-game.js`、`gomoku-view.js`、`worker.js` |
| 2048 | `/2048` | Stochastic Gumbel AlphaZero value net 与搜索 | `2048.html`、`worker2048.js`、`mcts2048.js`、`ai2048.js` |
| 躲避 | `/channel-dodge` | SAC 连续策略 | `channel-dodge.html`、`channel-dodge.js`、`dodge-ai.js` |
| ScotlandYard | `/scotlandyard/` | AZ Selfplay / PPO GNN 纯网络策略 | `scotlandyard/index.html`、`scotlandyard/` |

躲避游戏的「AI 托管」使用连续 SAC actor，网络为 `271→256→256→4`，前两维高斯均值经 `tanh` 后得到移动向量。权重以 little-endian float32 base64 打包在 `dodge-policy.js` 中，由 `dodge-ai.js` 同步前向运行，不依赖 ONNX、WASM 或 Worker。271 维观测包含最近 32 个威胁、3 条激光、3 颗炸弹、2 个血包、玩家状态与四向墙距，并与 `../SkyZero/DodgeSAC/env_dodge.py` 的 `_obs_vector` 对齐；决策频率为 30 Hz。游戏物理固定在 `450×600` 逻辑坐标系中，页面缩放只影响渲染，敌人弹幕仍由手写行为脚本控制。

ScotlandYard 的模型、规则、导出和验证约定见 [`scotlandyard/README.md`](scotlandyard/README.md)。

## 本地开发

```bash
cd /home/sky/RL/SkyZeroWeb
python3 serve.py
# Open http://localhost:8000
```

页面必须通过 HTTP 访问；Worker 的 `importScripts` 和模型 `fetch` 无法在 `file://` 下工作。`serve.py` 在 `:8000` 提供静态服务，并模拟 Cloudflare Pages clean URL。

需要排行榜 API 和本地 D1 时运行完整 Cloudflare 栈：

```bash
npx wrangler pages dev .
```

## 测试

```bash
export PATH=/home/sky/.nvm/versions/node/v24.16.0/bin:$PATH
npm test

# 单个测试文件
node --test tests/test_gomoku.mjs
node --test tests/test_mcts2048.mjs
```

测试使用 Node 18+ 内置 test runner，覆盖 `gomoku.js`、`mcts.js`、`ai2048.js` 和 `mcts2048.js` 等纯逻辑模块。UI 需要在真实浏览器中验证。项目无构建步骤，也没有 ESLint 或 Prettier 配置。

`mcts2048.js` 测试使用由训练侧 `python/mcts.py` 生成的 Gumbel 根与 PUCT 根 fixture，双方共享写死的 synthetic evaluator。训练侧搜索逻辑变化后，在包含 PyTorch 和 NumPy 的环境中重新生成：

```bash
python tests/gen_mcts2048_fixture.py
python tests/gen_mcts2048_puct_fixture.py
```

## 架构入口

- `index.html`：游戏入口页。
- `gomoku.html`、`style.css`：五子棋界面。
- `2048.html`：2048 界面和 AI 控制。
- `channel-dodge.html`、`channel-dodge.css`、`channel-dodge.js`：躲避页面与游戏。
- `dodge-ai.js`、`dodge-policy.js`：SAC actor 运行时与导出权重。
- `scotlandyard/index.html`、`scotlandyard/`：ScotlandYard 入口与独立模块。
- `main.js`：五子棋启动入口。
- `gomoku-context.js`：五子棋视图与控制器共享的实时对局状态。
- `gomoku-game.js`：五子棋对局状态、交互与 Worker 调度。
- `gomoku-view.js`：五子棋 Canvas、分析图表和候选视图。
- `gomoku-controls.js`：五子棋页面级弹层和选择控件。
- `worker.js`、`worker-tree.js`：五子棋搜索 Worker。
- `gomoku-inference.js`：两个五子棋 Worker 共用的 ONNX 推理适配。
- `worker2048.js`：2048 value net 与 expectimax Worker。
- `mcts.js`：V7.19 单线程 PUCT、加权回传、SVB 和 LCB 选择。
- `ai2048.js`：2048 afterstate 与一层 value-net expectimax。
- `gomoku.js`：V7.19 Renju、Standard、Freestyle 规则及输入协议。
- `models/manifest.json`：五子棋模型目录。
- `tools/`：模型导出与验证脚本。
- `functions/api/`：排行榜 API。

其他任务文档由 [`docs/README.md`](docs/README.md) 路由。

## 五子棋模型

五子棋模型来自 `../SkyZero/SkyZero_V7.19/data/web/models/`。当前六档是 V7.19 `b9c96tflrs` 的 iter 206、300、419、654、1000、2448，按训练迭代排序，不复用不可比较的旧 V7.1 ELO。

```bash
conda run -n pytorch python tools/export_v719_onnx.py \
    --ckpt ../SkyZero/SkyZero_V7.19/data/web/models/iter2448.pt \
    --out models/level6.onnx
```

导出器会验证重建 PyTorch、TorchScript 与 ONNX Runtime 数值一致。导出后更新 `models/manifest.json` 的顺序、标签、文件和参数。`/gomoku/lv1` 到 `/gomoku/lv6` 可直接选择对应档位；`/gomoku` 保留普通模型选择器。

导出环境除 PyTorch 外需要 `onnx`、`onnxscript` 和 `onnxruntime`。

## 2048 模型

2048 模型来自 `../SkyZero/SkyZero_2048_V2`。发布最新网络时，以 `data2048/nets/<net>/latest.pt` 和相邻的 `latest.meta.json` 为事实源；`server_models/model_iter<N>.pt` 是可能滞后的手动发布快照。

当前网络为 `b5c96`，模型是 traced TorchScript。其 value 位于 scale 30 的 h-space，因此导出时同时指定缩放和逆变换：

```bash
conda run -n pytorch python tools/export_onnx_2048.py \
    --ckpt ../SkyZero/SkyZero_2048_V2/data2048/nets/b5c96/latest.pt \
    --out models/2048.onnx \
    --value-scale 30 \
    --value-transform
```

导出后将 `2048.html` 中的 `AI_MODEL_VERSION` 更新为 `v2-<net>-vt-iter<N>`，其中迭代数来自 meta 文件。可用随机 one-hot 局面对比 TorchScript 经 value scale 和逆变换后的输出与 ONNX Runtime 输出。

浏览器 AI 使用 value head 做一层 expectimax：`Q(a) = reward(a) + γ·E_spawn[V(next)]`，比训练自弈使用的完整 Gumbel MCTS 更轻但更弱。`tests/test_ai2048.mjs` 使用训练侧 `python/game.py` 生成的 fixture 对拍滑动、生成和编码逻辑。

## 躲避模型

模型来自 `../SkyZero/DodgeSAC`。导出命令如下：

```bash
conda run -n pytorch python tools/export_dodge_sac.py \
    --ckpt ../SkyZero/DodgeSAC/runs/<name>/best.pt \
    --version <tag>
```

导出器包含 PyTorch 与打包权重的自检。导出后同步更新 `channel-dodge.html` 中 `dodge-policy.js?v=...` 的版本参数。

## 五子棋搜索循环

五子棋在 ponder 与 move-search 之间交替，并通过 `worker.js` 的 `applyMove` 将落子对应的子节点提升为新根，持续复用搜索树。`gomoku-game.js` 的 `triggerAISearch()` 通过 `isPonderTurn()` 选择路径。

- Ponder：对弈模式下用于玩家回合，分析模式下用于任意一方。每块固定 `ANALYSIS_CHUNK = 128` 次 simulation，复用树继续搜索，直到累计根访问达到 `ANALYSIS_CAP_MIN = 2000`。对弈时状态栏保持“轮到你”，候选、胜率和热力图继续刷新；玩家落子时以 `searchId++` 中止在途块。
- Move-search：AI 回合使用 anytime PUCT，运行至 `thinkMs` 或累计根访问达到 `SEARCH_VISIT_CAP = 2000`，以先到者为准，再使用 V7.19 retrospective weight 与 LCB 选着。连续三次 AI 落子搜索处于极优局面时，`searchFactorWhenWinning = 0.30` 会将时间和访问上限平滑降至 30%；ponder 不参与统计。`thinkMs` 只控制 AI 回合，不增加 ponder 深度。

完整循环为 ready → `newGame` → 玩家回合 ponder → 玩家落子 → `move` 与 AI move-search → AI 落子 → `move` → 回到 ponder。两个上限都计算跨树复用后的累计根访问，并刻意保持相等；修改其中一个时必须同步核对另一个。

## V7.19 浏览器搜索约定

五子棋使用 V7.19 play package：固定 15×15 网络画布上的 5 spatial 与 7 global 特征，运行棋盘为 11×11 至 15×15；包含 main、opponent、optimistic policy heads、uncertainty-weighted recompute backup、value-weighted child aggregation、subtree value bias、variance-scaled PUCT、root/non-root FPU、随机 D4 推理、空盘中心限制、根对称裁剪、树复用、retrospective play-selection weights 和 LCB 选着。

PDA 可在 `-1` 到 `1` 之间以 `0.1` 调节，默认 `0.5`，接近中点时吸附到 `0`。它改变神经网络条件但不改变搜索预算；对弈模式参考色是 AI，分析模式固定参考黑方。PDA 数值或参考方改变时必须清空搜索树和神经网络缓存。

Renju 禁手仍允许黑方落下并作为白胜终局；Standard 双方恰好五连获胜；Freestyle 五连及以上获胜。浏览器适配为单线程、单 leaf，每次 cache miss 随机选择一个 D4 变换，不做八变换集成。RNNM 明确不实现；训练专用 `value_td` 不进入浏览器 ONNX 图。

分析抽屉默认显示 current、optimistic、opponent 和经 LCB 调整的 play-selection 热力图；原始 visits 仅作为可展开诊断项，两张 future-position 热力图不展示。

## 其他页面与共享工具

- `mcts-tree.html`、`mcts-tree.js`、`worker-tree.js`：逐 simulation 展示五子棋搜索树。`worker-tree.js` 有一份从 `worker.js` 推理路径裁剪而来的实现，修改推理或全局特征时必须同时检查两边。
- `i18n.js`：仅由 `gomoku.html` 使用。`index.html`、`2048.html` 和 `channel-dodge.html` 仍各自内联翻译字典。
- `seg-slide.js`：通过 `MutationObserver` 监听 `.seg-btn` 的 `aria-pressed`，为分段控件提供滑块动画。

## 缓存与运行约束

- `_headers` 为 JavaScript 设置缓存。修改由 `importScripts` 加载的 `gomoku.js`、`mcts.js`、`ai2048.js` 等文件时，必须同步更新引用 URL 的 `?v=<version>`。替换模型时也必须更新模型 URL 或对应版本标识。
- ONNX Runtime Web 从 CDN 加载，固定使用单线程 WASM，避免依赖 SharedArrayBuffer 跨域隔离。
- 浏览器游戏、规则、特征或搜索逻辑变化时，必须与对应 Python 训练实现对拍。fixture 只代表生成它时的训练侧行为，训练实现变化后需要重新生成。

## 排行榜

排行榜是仅昵称、无登录的荣誉系统，覆盖 2048 与躲避游戏。后端位于 `functions/api/`，数据库结构在 `schema.sql`，前端入口为 `leaderboard.js` 的 `window.SkzLB.submit(game, score)`。

完整搭建和重置流程见 [`LEADERBOARD_SETUP.md`](LEADERBOARD_SETUP.md)。

## 部署

- 通过 Cloudflare Pages 部署，无构建步骤，源码即产物。
- 本仓库是独立 Git 根，Pages 根目录为 `/`，`wrangler.toml` 的 `pages_build_output_dir = "."`。
- `functions/api/*` 由 Pages 自动编译为 Functions。
- 缓存策略定义在 `_headers`：ONNX 模型长期 immutable，manifest 缓存 300 秒，其余静态资源缓存 1 小时。
- 推送到远端会触发自动部署；执行部署相关 Git 操作前应明确确认发布意图。
