# ScotlandYard 人类测评

入口为 `/scotlandyard/`。玩家扮演 Evader，Pursuer 在浏览器 Worker 中以 ONNX Runtime Web 进行纯网络推理。

模型来源、规则身份、输入协议与校验值以 [`models.json`](../public/scotlandyard/models.json) 为准。权重来自相邻 ScotlandYardRL 仓库的 `ScotlandYardRL_V0.16/az_selfplay/data/exp_all/evader_location_belief_backup/`。决策使用 canonical order mask、确定性 argmax 和固定 Pursuer 身份，不执行搜索或 belief backup。原生批量评测另有随机身份重标记及半精度设置，因此不声称浏览器与批量评测逐局结果相同。

## 参与流程

点击“开始测评”建立一组固定配额、随机顺序的对局；配额与协议的唯一配置位于 [`evaluation-store.js`](../public/scotlandyard/evaluation-store.js)。顺序和开局在创建测评组时保存，刷新、下载失败与中途导出不重排。关闭页面后可继续；当前局结算后点击“下一局”，整组完成后可以“再测一组”。每组有独立编号，同一浏览器保留同一匿名参与者编号和所有测评组。

允许无限悔棋；被抓或无路可走仍可撤回，逃脱成功或主动认输才结算。退回开局按实际退回的轮数计入累计悔棋数，已结算结果不能更改。挑战规则由 [`pilot-challenge.js`](../public/scotlandyard/pilot-challenge.js) 管理。当前没有每局时间限制。

页面、对局标题、加载状态和结果提示隐藏 checkpoint。JSON 仍包含真实模型身份，供研究者汇总；这是参与界面的盲测，不防止参与者检查导出文件、浏览器存储或静态模型资源来识别对手。

点击开始或恢复未完成对局时，先下载并校验整个测评目录中的全部权重，显示合计下载进度，再开放对局。加载入口见 [`local-api.js`](../public/scotlandyard/local-api.js)，下载校验和内存缓存由 [`inference-worker.js`](../public/scotlandyard/inference-worker.js) 管理。下载失败后再次点击开始或继续即可重试，已校验成功的文件在当前页面内复用。

同一页面内切换对手、开始下一组无需再次下载权重；推理时只保留一个模型会话。刷新或关闭页面后，内存缓存释放，重新准备时由浏览器的 HTTP 缓存决定是否需要网络传输；不保证离线重新打开页面。

## 保存与导出

进度、结果和随机安排保存到同一浏览器、同一站点的 localStorage；协议和存储键见 `evaluation-store.js`。此前的自由预试数据保留在原存储键中，不纳入当前测评。另一个标签页更新进度后，旧页面须刷新才能继续写入，避免覆盖结果。

可随时导出累计 JSON，包括参与者编号、协议、模型身份、总体汇总、每组汇总和已开始的逐局简表。进行中的对局标记为 `in_progress`，不参与胜率和均值；未开始的安排、完整棋局和撤回栈不导出。

汇总以成功或认输的对局为分母。回合数是最终保留路径上的 Evader 行动次数，悔棋数包含所有成功回退；胜局平均悔棋数仅统计成功局。合并多次导出时，以参与者编号、测评组编号和对局编号去重，并用更新的记录替换早先的进行中记录。重复测评属于同一参与者，不能视为新增参与者。

## 揭示显示

揭示发生在配置规定的回合中、Evader 移动之后，公开的是移动落点，黑票不会取消该次揭示。回合条区分“待揭示”和“已揭示”；地图仅在当次揭示后显示红圈，进入下一回合时隐藏，不显示回合或节点文字。动画结束前不提前显示新揭示，悔棋后按恢复的回合显示。显示由 [`timeline_view.js`](../public/scotlandyard/timeline_view.js)、[`map_controller.js`](../public/scotlandyard/map_controller.js) 驱动，裁判规则仍由 [`engine.js`](../public/scotlandyard/engine.js) 实现。

## 本地运行与验证

在 SkyZeroWeb 根目录执行 `npm run dev`，打开 `/scotlandyard/`。需要 HTTP localhost 或 HTTPS；首次加载需要联网获取本站权重及固定版本的 ONNX Runtime CDN 资源。推理无需 GPU 或后端服务器。

```bash
node --test tests/test_scotlandyard_v016.mjs tests/test_scotlandyard_pilot.mjs tests/test_scotlandyard_evaluation.mjs tests/test_scotlandyard_reveals.mjs tests/test_scotlandyard_preload.mjs
conda run -n pytorch python tests/browser_scotlandyard_smoke.py --url http://127.0.0.1:8000
```

浏览器测试覆盖完整一组、重复测评、身份隐藏、全部揭示回合、撤回、断点恢复、全量预下载、下载失败恢复、下载后断网切换全部对手与重复测评、导出汇总及移动布局，并验证全部模型的 WASM 动作 fixture。

修改裁判、编码或推理协议时，从 V0.16 原生实现重新核验：

```bash
conda run -n pytorch python tools/export_scotlandyard.py --source ../ScotlandYardRL/ScotlandYardRL_V0.16
conda run -n pytorch python tools/validate_scotlandyard.py --source ../ScotlandYardRL/ScotlandYardRL_V0.16
```

`local-api.js` 的 `/api/` 是浏览器内方法分发，不发送游戏状态到服务器。模型文件名带来源 hash；更新页面模块时同时更新缓存版本。
