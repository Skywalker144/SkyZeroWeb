# SkyZeroWeb

聚合游戏平台，最初是 AI 五子棋页面，随后加入了多个自训练 AI。五子棋、2048、躲避和 ScotlandYard 的游戏逻辑与 AI 推理在浏览器本地运行；排行榜使用 Cloudflare Pages Functions 和 D1。德州扑克通过 Pages API 接入 Linux 上的作者原版 PokerSkill 策略服务，并转发玩家自己的 DeepSeek 请求。

## 当前游戏

| 游戏 | 页面 | AI | 主要入口 |
|---|---|---|---|
| 德州扑克 | `/poker/` | DeepSeek 与作者原版 PokerSkill | `public/poker/` |
| 五子棋 | `/gomoku` | AlphaZero 风格神经网络与 MCTS | `public/gomoku/` |
| 2048 | `/2048` | Stochastic Gumbel AlphaZero value net 与搜索 | `public/2048/` |
| 躲避 | `/channel-dodge` | SAC 连续策略 | `public/channel-dodge/` |
| ScotlandYard | `/scotlandyard/` | V0.16 AZ 纯网络 checkpoint 盲测 | `public/scotlandyard/` |

躲避游戏的「AI 托管」使用连续 SAC actor，网络为 `271→256→256→4`，前两维高斯均值经 `tanh` 后得到移动向量。权重以 little-endian float32 base64 打包在 `public/channel-dodge/ai/policy.js` 中，由同目录的 `runtime.js` 同步前向运行，不依赖 ONNX、WASM 或 Worker。271 维观测包含最近 32 个威胁、3 条激光、3 颗炸弹、2 个血包、玩家状态与四向墙距，并与 `../SkyZero/DodgeSAC/env_dodge.py` 的 `_obs_vector` 对齐；决策频率为 30 Hz。游戏物理固定在 `450×600` 逻辑坐标系中，页面缩放只影响渲染，敌人弹幕仍由手写行为脚本控制。

德州扑克的运行、配置与验证见 [`docs/poker.md`](docs/poker.md)。

ScotlandYard 的模型、规则、导出和验证约定见 [`docs/scotlandyard.md`](docs/scotlandyard.md)。

## 本地开发

```bash
cd /home/sky/RL/SkyZeroWeb
npm run dev
# Open http://localhost:8000
```

页面必须通过 HTTP 访问；Worker 的 `importScripts` 和模型 `fetch` 无法在 `file://` 下工作。`scripts/serve.py` 在 `:8000` 提供 `public/`，并模拟 Cloudflare Pages 路由。

需要排行榜 API 和本地 D1 时运行完整 Cloudflare 栈：

```bash
npm run dev:cloudflare
```

## 测试

```bash
export PATH=/home/sky/.nvm/versions/node/v24.16.0/bin:$PATH
npm test

# 单个测试文件
node --test tests/test_gomoku.mjs
node --test tests/test_mcts2048.mjs
```

测试使用 Node 18+ 内置 test runner，覆盖各游戏 `engine/`、公共 Worker 运行时和发布结构。UI 需要在真实浏览器中验证。项目无构建步骤，也没有 ESLint 或 Prettier 配置。

2048 MCTS 测试使用由训练侧 `python/mcts.py` 生成的 Gumbel 根与 PUCT 根 fixture，双方共享写死的 synthetic evaluator。训练侧搜索逻辑变化后，在包含 PyTorch 和 NumPy 的环境中重新生成：

```bash
python tests/gen_mcts2048_fixture.py
python tests/gen_mcts2048_puct_fixture.py
```

## 架构入口

- `public/`：Cloudflare Pages 唯一静态发布目录。
- `public/shared/`：字体、主题基础样式、排行榜客户端、分段控件和 ONNX Worker 运行时。
- `public/gomoku/app/`：五子棋页面状态、视图、控制器和启动入口。
- `public/gomoku/engine/`：五子棋规则、MCTS 和推理协议。
- `public/gomoku/workers/`：对弈与搜索树 Worker；`search-core.js` 是共同的单步搜索事实源。
- `public/gomoku/models/`：五子棋模型与 manifest。
- `public/2048/`、`public/channel-dodge/`、`public/scotlandyard/`：其他游戏的独立垂直模块。
- `functions/api/`：排行榜与德州扑克 Pages Functions。
- `services/poker/`：Linux 原版策略服务。
- `database/`：D1 数据库结构。
- `tools/`：模型导出与验证脚本；默认写入对应的 `public/<game>/` 目录。
- `tests/`：逻辑对拍、结构契约和浏览器 smoke test。

其他任务文档由 [`docs/README.md`](docs/README.md) 路由。

## 五子棋模型

五子棋模型来自 `../SkyZero/SkyZero_V7.19/data/web/models/`。当前六档是 V7.19 `b9c96tflrs` 的 iter 206、300、419、654、1000、2448，按训练迭代排序，不复用不可比较的旧 V7.1 ELO。

```bash
conda run -n pytorch python tools/export_v719_onnx.py \
    --ckpt ../SkyZero/SkyZero_V7.19/data/web/models/iter2448.pt \
    --out public/gomoku/models/level6.onnx
```

导出器会验证重建 PyTorch、TorchScript 与 ONNX Runtime 数值一致。导出后更新 `public/gomoku/models/manifest.json` 的顺序、标签、文件和参数。`/gomoku/lv1` 到 `/gomoku/lv6` 可直接选择对应档位；`/gomoku` 保留普通模型选择器。

导出环境除 PyTorch 外需要 `onnx`、`onnxscript` 和 `onnxruntime`。

## 2048 模型

2048 模型来自 `../SkyZero/SkyZero_2048_V2`。发布最新网络时，以 `data2048/nets/<net>/latest.pt` 和相邻的 `latest.meta.json` 为事实源；`server_models/model_iter<N>.pt` 是可能滞后的手动发布快照。

当前网络为 `b5c96`，模型是 traced TorchScript。其 value 位于 scale 30 的 h-space，因此导出时同时指定缩放和逆变换：

```bash
conda run -n pytorch python tools/export_onnx_2048.py \
    --ckpt ../SkyZero/SkyZero_2048_V2/data2048/nets/b5c96/latest.pt \
    --out public/2048/models/value.onnx \
    --value-scale 30 \
    --value-transform
```

导出后将 `public/2048/app.js` 中的 `AI_MODEL_VERSION` 更新为 `v2-<net>-vt-iter<N>`，其中迭代数来自 meta 文件。可用随机 one-hot 局面对比 TorchScript 经 value scale 和逆变换后的输出与 ONNX Runtime 输出。

浏览器 AI 使用 value head 做一层 expectimax：`Q(a) = reward(a) + γ·E_spawn[V(next)]`，比训练自弈使用的完整 Gumbel MCTS 更轻但更弱。`tests/test_ai2048.mjs` 使用训练侧 `python/game.py` 生成的 fixture 对拍滑动、生成和编码逻辑。

## 躲避模型

模型来自 `../SkyZero/DodgeSAC`。导出命令如下：

```bash
conda run -n pytorch python tools/export_dodge_sac.py \
    --ckpt ../SkyZero/DodgeSAC/runs/<name>/best.pt \
    --version <tag>
```

导出器包含 PyTorch 与打包权重的自检。导出后同步更新 `public/channel-dodge/index.html` 中 `ai/policy.js?v=...` 的版本参数。

## 五子棋搜索循环

五子棋在 ponder 与 move-search 之间交替，并通过 `public/gomoku/workers/search.js` 的 `applyMove` 将落子对应的子节点提升为新根，持续复用搜索树。`public/gomoku/app/game.js` 的 `triggerAISearch()` 通过 `isPonderTurn()` 选择路径。

- Ponder：对弈模式下用于玩家回合，分析模式下用于任意一方。每块固定 `ANALYSIS_CHUNK = 128` 次 simulation，复用树继续搜索，直到累计根访问达到 `ANALYSIS_CAP_MIN = 2000`。对弈时状态栏保持“轮到你”，候选、胜率和热力图继续刷新；玩家落子时以 `searchId++` 中止在途块。
- Move-search：AI 回合使用 anytime PUCT，运行至 `thinkMs` 或累计根访问达到 `SEARCH_VISIT_CAP = 2000`，以先到者为准，再使用 V7.19 retrospective weight 与 LCB 选着。连续三次 AI 落子搜索处于极优局面时，`searchFactorWhenWinning = 0.30` 会将时间和访问上限平滑降至 30%；ponder 不参与统计。`thinkMs` 只控制 AI 回合，不增加 ponder 深度。

完整循环为 ready → `newGame` → 玩家回合 ponder → 玩家落子 → `move` 与 AI move-search → AI 落子 → `move` → 回到 ponder。两个上限都计算跨树复用后的累计根访问，并刻意保持相等；修改其中一个时必须同步核对另一个。

## V7.19 浏览器搜索约定

五子棋使用 V7.19 play package：固定 15×15 网络画布上的 5 spatial 与 7 global 特征，运行棋盘为 11×11 至 15×15；包含 main、opponent、optimistic policy heads、uncertainty-weighted recompute backup、value-weighted child aggregation、subtree value bias、variance-scaled PUCT、root/non-root FPU、随机 D4 推理、空盘中心限制、根对称裁剪、树复用、retrospective play-selection weights 和 LCB 选着。

PDA 可在 `-1` 到 `1` 之间以 `0.1` 调节，默认 `0.5`，接近中点时吸附到 `0`。它改变神经网络条件但不改变搜索预算；对弈模式参考色是 AI，分析模式固定参考黑方。PDA 数值或参考方改变时必须清空搜索树和神经网络缓存。

Renju 禁手仍允许黑方落下并作为白胜终局；Standard 双方恰好五连获胜；Freestyle 五连及以上获胜。浏览器适配为单线程、单 leaf，每次 cache miss 随机选择一个 D4 变换，不做八变换集成。RNNM 明确不实现；训练专用 `value_td` 不进入浏览器 ONNX 图。

分析抽屉默认显示 current、optimistic、opponent 和经 LCB 调整的 play-selection 热力图；原始 visits 仅作为可展开诊断项，两张 future-position 热力图不展示。

## 其他页面与共享工具

- `public/gomoku/tree/` 与 `public/gomoku/workers/tree.js`：逐 simulation 展示五子棋搜索树，与对弈 Worker 共用 `search-core.js`、规则、MCTS 和推理实现。
- `public/gomoku/app/i18n.js`：仅由五子棋使用；其他页面保留自己的小型翻译字典。
- `public/shared/ui/seg-slide.js`：为分段控件提供滑块动画。

## 缓存与运行约束

- `public/_headers` 定义静态缓存。修改 Worker 通过 `importScripts` 加载的引擎文件时，必须同步更新引用 URL 的 `?v=<version>`。替换模型时也必须更新模型 URL 或对应版本标识。
- ONNX Runtime Web 从 CDN 加载，固定使用单线程 WASM，避免依赖 SharedArrayBuffer 跨域隔离。
- 浏览器游戏、规则、特征或搜索逻辑变化时，必须与对应 Python 训练实现对拍。fixture 只代表生成它时的训练侧行为，训练实现变化后需要重新生成。

## 排行榜

排行榜是仅昵称、无登录的荣誉系统，覆盖 2048 与躲避游戏。后端位于 `functions/api/`，数据库结构在 `database/schema.sql`，前端入口为 `public/shared/leaderboard.js` 的 `window.SkzLB.submit(game, score)`。

完整搭建和重置流程见 [`docs/leaderboard.md`](docs/leaderboard.md)。

## 部署

- 通过 Cloudflare Pages 部署，无构建步骤，`public/` 中的源码即静态产物。
- 本仓库是独立 Git 根，`wrangler.toml` 的 `pages_build_output_dir = "./public"`。
- `functions/api/*` 由 Pages 自动编译为 Functions。
- 缓存策略定义在 `public/_headers`：ONNX 模型长期 immutable，五子棋 manifest 缓存 300 秒，JavaScript 与 CSS 缓存 1 小时。
- 推送到远端会触发自动部署；执行部署相关 Git 操作前应明确确认发布意图。
