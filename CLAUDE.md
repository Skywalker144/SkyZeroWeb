# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# SkyZeroWeb

聚合游戏平台（最初是一个 AI 五子棋页面，后来把多个自训练 AI 都塞了进来）。
所有 AI **全部在浏览器本地运行**，没有后端推理服务器。

## 当前包含的游戏

| 游戏 | 页面 | AI | 相关文件 |
|------|------|----|---------|
| 五子棋 | `/gomoku` | AlphaZero 风格神经网络 + MCTS | `gomoku.html` `main.js` `worker.js` `mcts.js` `gomoku.js` |
| 2048 | `/2048` | Stochastic Gumbel AlphaZero value net + 搜索 | `2048.html` `worker2048.js` `mcts2048.js` `ai2048.js` |
| 躲避 (channel-dodge) | `/channel-dodge` | PPO 策略网络（纯 JS 内联权重，无 ONNX/Worker） | `channel-dodge.html`（单文件自包含）`dodge-policy.js` |

躲避游戏现在有一个 **SAC 训练的自动驾驶 AI**（页面上「AI 托管」按钮，玩家可随时开关）。和五子棋/2048 不同，它的网络极小（连续 SAC actor，MLP `271→256→256→4`，relu；输出前 2 维=移动向量的高斯均值，贪心动作 = `tanh(mean)` 裁剪到单位圆），所以**没有用 onnxruntime-web**：actor 权重打包成 `dodge-policy.js`（base64 little-endian float32，`continuous:true`），在 `channel-dodge.html` 内用 ~30 行**纯 JS 同步前向**跑（无 CDN、无 WASM、无 Worker、无 async），保持页面单文件自包含。271 维观测（最近 32 个威胁的相对位置/速度/伤害/半径 + 3 条激光 + 3 颗炸弹 + 2 个血包 + 玩家血量/无敌/位置/四向墙距）在 JS 里逐字复刻训练环境 `../SkyZero/DodgeSAC/env_dodge.py` 的 `_obs_vector`；决策固定 **30 Hz**。**关键：游戏物理必须跑在固定 450×600 逻辑坐标系**（`SIM_W/SIM_H`，`resize()` 里只缩放渲染，不缩放物理）——否则手机等小屏上 obs 会被 `POS_NORM/VEL_NORM` 裁剪/失真，精确的连续 SAC 直接秒死（旧离散 argmax 策略粗糙反而扛得住，所以这坑曾被掩盖）。敌人弹幕仍是手写行为脚本（AI 只接管玩家移动）。

## 常用命令

```bash
# 纯静态预览（不含排行榜 API；file:// 不行，Worker 的 importScripts/fetch 需要 HTTP）。
# serve.py 在 :8000 起静态服务，并模拟 Cloudflare Pages 的 clean URL（/gomoku → gomoku.html）；等价于 npm run dev
python serve.py

# 含 Functions + 本地 D1 的完整栈（排行榜 API 可用）
npx wrangler pages dev .

# 全部单元测试（纯逻辑：gomoku / mcts / ai2048 / mcts2048；UI 靠人工在浏览器里测）
export PATH=/home/sky/.nvm/versions/node/v24.16.0/bin:$PATH
npm test

# 只跑单个测试文件
node --test tests/test_gomoku.mjs
node --test tests/test_mcts2048.mjs

# mcts2048.js 的测试是对拍 SkyZero_2048 的 python/mcts.py 生成的 fixture；
# 改了训练侧的搜索逻辑后，需要用装了 torch/numpy 的 python 重新生成
python tests/gen_mcts2048_fixture.py        # → tests/mcts2048_fixture.json（Gumbel 根）
python tests/gen_mcts2048_puct_fixture.py   # → tests/mcts2048_puct_fixture.json（PUCT 根）
```

无构建步骤，也未配置 lint（仓库里没有 eslint/prettier 配置）。更详细的导出模型 / 排行榜运维命令，分别见 `README.md` 与 `LEADERBOARD_SETUP.md`。

## 五子棋搜索逻辑（对弈循环）

五子棋是「ponder（分析，不落子）↔ move-search（搜索并落子）」交替循环，**全程复用搜索树**（`worker.js` 的 `applyMove` 把刚走的那一手对应子节点直接当新根）。两条路径在 `main.js` `triggerAISearch()` 里分流（`isPonderTurn()` 判定），用**不同的搜索方式和不同的访问上限**：

- **ponder（对弈模式你的回合 / 分析模式任意手）**：固定 `ANALYSIS_CHUNK = 96` sims/块的 PUCT，结果回来后只要**累计根访问 < `ANALYSIS_CAP_MIN = 2000`** 就复用树再下一块，直到 ≥2000 才停（见结果处理 `main.js:1708`）。对弈模式下安静跑（状态栏只显示「轮到你」），但候选/胜率/热力图实时刷新；你一落子就 `searchId++` 中止当前块。
- **move-search（对弈模式 AI 自己的回合）**：anytime PUCT，跑满 `thinkMs`（工具栏「思考时间」下拉，默认 3000ms）**或**累计根访问到 `SEARCH_VISIT_CAP`（`worker.js`，现 = 2000，与 ponder 上限对齐）——**先到者停**，然后落最高访问手。`thinkMs` 只影响 AI 自己这一手的搜索时长，不会加深你的回合 ponder（后者永远是 96-块到 2000 封顶，与 `thinkMs` 无关）。

完整一轮：开页面 ready→`newGame`→ponder 你的回合；你落子→`move`（树复用）→move-search AI 的手→AI 落子→`move`（树复用）→回到 ponder。**两个上限都是「跨树复用的累计根访问」，不是单次搜索量**；因为复用，中/残局往往没跑满时间或没下满块就已到顶。改其中一个上限记得另一个同步（两者刻意保持相等）。

## 其他页面 / 共享工具

- `mcts-tree.html`（首页「搜索树可视化」入口）+ `mcts-tree.js` + `worker-tree.js`：五子棋 AlphaZero 搜索树的调试可视化工具，逐 simulation 画树。`worker-tree.js` 复用 `gomoku.js`/`mcts.js`，但**是独立于 `worker.js` 的另一份 worker**（源码注释自称是 `worker.js::inference` 的裁剪版）——改五子棋的推理/全局特征逻辑时两边都要看，两份实现容易只改一处而分叉。
- `i18n.js`：只被 `gomoku.html` 引入的共享中英切换模块（`getLang`/`setLang`/`t`/`applyI18n`，`localStorage` key `skz_lang`）。`index.html`/`2048.html`/`channel-dodge.html` 各自内联自己的翻译字典，并未复用它——处理多语言文案时注意这一不一致，别默认改 `i18n.js` 就能覆盖全站。
- `seg-slide.js`：分段控件（segmented control）选中态的滑块动画，靠 `MutationObserver` 监听 `.seg-btn` 的 `aria-pressed` 变化驱动，加个 `seg-slide` class 即可接入、无需按控件单独接线；`gomoku.html` 的模式/执子切换和 `index.html` 的语言/主题切换在用。

## 部署（重要）

- 通过 **Cloudflare Pages** 部署，**无构建步骤**：源码即产物，`git push` 自动上线。
- 本仓库**自身就是 git 根**（已从 SkyZero monorepo 拆出为独立 repo `Skywalker144/SkyZeroWeb`）；Cloudflare Pages 的「根目录」设为 `/`，`wrangler.toml` 里 `pages_build_output_dir = "."` 与之对应。
- `functions/api/*` 会被 Pages 自动编译成 Functions（排行榜后端）。
- 缓存策略在 `_headers`：`*.onnx` 永久 immutable，`manifest.json` 300s，其余 1h。

## 关键约定 / 易踩的坑

- **缓存击穿**：因为 `_headers` 给 JS 设了 `max-age=3600`，改了 `gomoku.js`/`mcts.js`/`ai2048.js` 等被 `importScripts` 加载的文件后，必须靠 URL 上的 `?v=<ts>` query 才能让浏览器和 Worker 拿到新版本（见 `worker.js` / `worker2048.js` 顶部、`main.js` 里的 cache-bust 逻辑）。换模型同理要给 `models/2048.onnx` 之类的 URL 加 cache-bust。
- onnxruntime-web 从 **CDN** 加载，强制 **单线程 WASM**（`ort.env.wasm.numThreads = 1`，规避 SharedArrayBuffer 跨域问题）。
- 模型来自 **SkyZero monorepo**：五子棋来自 `../SkyZero/SkyZero_V7.1/`，用 `tools/export_onnx.py` 导出到 `models/` 再改 `models/manifest.json`（5 档 ELO 目录）；**2048 来自 `../SkyZero/SkyZero_2048_V2`**（迁移前是 `../SkyZero/SkyZero_2048/`）。**真正最新的网络在 `data2048/nets/<net>/latest.pt`**（已 traced 的 TorchScript，旁边 `latest.meta.json` 给出 `iter`/`network`/`value_scale`）；`server_models/model_iter<N>.pt` 只是**手动晋升的发布快照、常滞后于训练**，所以要上最新就直接拿 `nets/<net>/latest.pt`（当前网络是 `b5c96`）。部署流程（`pytorch` conda 环境）：① `python tools/export_onnx_2048.py --ckpt <…>/data2048/nets/<net>/latest.pt --out models/2048.onnx --value-scale 30 --value-transform`（脚本直接吃 TorchScript、无需 `--net`；`--value-scale` 取自 meta = V2 `run.cfg` 的 `VALUE_SCALE=30`，V2 value 恒 h 空间故必带 `--value-transform`）；② 把 `2048.html` 的 `AI_MODEL_VERSION` 改成 `v2-<net>-vt-iter<N>`（N 取自 meta）做 cache-bust。可选数值复核：torch（TorchScript×value_scale 再 h⁻¹）vs onnxruntime 在随机 one-hot 板面上对差，应 ≪1 raw point。导出步骤与所需 conda 环境见 `README.md`。
- **躲避走另一条路（不是 ONNX）**：模型来自 `../SkyZero/DodgeSAC`（同在 SkyZero 仓库内的 SAC 训练项目），用 `tools/export_dodge_sac.py --ckpt ../SkyZero/DodgeSAC/runs/<name>/best.pt --version <tag>` 把 SAC actor 权重导出成根目录的 `dodge-policy.js`（导出时带 torch-vs-打包自检，布局错了会直接报错）。重训后**重新导出**，并把 `channel-dodge.html` 里 `<script src="dodge-policy.js?v=…">` 的 `?v=` 改成新 `<tag>` 做 cache-bust（脚本会打印要粘贴的那行）。Python↔JS 一致性除导出自检外，还可用 `node` 无头跑整页游戏逻辑复核（曾用此定位手机坐标系 bug）。
- **JS 端游戏/搜索逻辑改动要跟 Python 训练侧对拍**：`tests/test_ai2048.mjs` 靠 `tests/ai2048_fixture.json`（由 `SkyZero_2048/python/game.py` 生成）校验滑动/生成/编码逻辑；`tests/test_mcts2048.mjs` 靠 `tests/mcts2048_fixture.json` / `mcts2048_puct_fixture.json`（由本仓库 `tests/gen_mcts2048_*_fixture.py` 跑参考 `python/mcts.py` 生成，双方用同一个写死的 synthetic evaluator）校验 Gumbel 与 PUCT 两条根搜索路径。改了对应 Python 侧逻辑后要重新生成 fixture，否则测试只是在验证一份过时快照。
- 浏览器版相对训练版做了简化（无 8 重对称集成、无并行 MCTS 等），细节见 `README.md` 末尾「Differences from V5」。

## 排行榜（Cloudflare D1）

仅昵称、无登录的「荣誉系统」榜单，覆盖 2048 与躲避（五子棋无连续分数，故排除）。
后端在 `functions/api/`，库表见 `schema.sql`，前端是 `leaderboard.js`（`window.SkzLB.submit(game, score)`）。
完整搭建/重置说明见 `LEADERBOARD_SETUP.md`。
