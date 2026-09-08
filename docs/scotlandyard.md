# ScotlandYard

SkyZeroWeb 的纯静态 ScotlandYard 页面，入口 `public/scotlandyard/index.html`，地图、布局、主题、移动动画和操作模块来自 ScotlandYardRL V0.11 的统一 Web。

玩家固定为 Evader，3 名 Pursuer 使用浏览器 Worker 内的 ONNX Runtime Web WASM 纯网络确定性推理。按网络选出的顺序逐个解码动作，动态避让其他 Pursuer；无搜索。模型只接收公开信息，不包含 Evader 私有位置，也不导出训练用 critic。

默认 AZ Selfplay `checkpoint_000470.pth` 的 Pursuer；可切换 PPO GNN `update_000500.pt`。首次打开自动下载默认 AZ 权重，按实际接收字节显示进度，校验和初始化完成后直接开始对局。已有会话自动恢复。右上角仅提供 AZ / PPO GNN 选择，切换后立即加载模型并使用当前 seed 重置对局；规则固定、Evader 始终由玩家控制。加载时检查文件长度和 SHA-256。完整来源、模型输入、环境指纹和校验值见 `models.json`。

页面主题沿用站点首页设置，游戏页不提供单独的主题入口。

规则：finite_tickets_v0，23 回合，暴露于 3/8/13/18 回合；Pursuer 初始 Taxi/Bus/Underground 为 11/8/4，Evader 普通票按原规则实际存储为 1,000,000 并显示 ∞，黑票 5；无双动。按整队行动后判断捕获，捕获优先于回合上限。

页面支持撤回、同 seed 重开、新 seed 开局、刷新恢复。浏览器 seed 使用 `mulberry32-fold53-v1`，同 seed 可在本页复现；不与原 Python 的 NumPy seed 序列等同。会话 metadata 记录实际起点。会话存于 sessionStorage；禁用存储时仍能玩，但不能刷新恢复。

## 本地运行

在 SkyZeroWeb 根目录执行 `npm run dev`，打开 `/scotlandyard/`。需要 HTTP localhost 或 HTTPS；首次使用需联网获取固定版本 `onnxruntime-web@1.17.0` 的 CDN JS/WASM 和本站 ONNX 权重。无需 Python 游戏服务器或 GPU。

## 导出与验证

```bash
conda run -n pytorch python tools/export_scotlandyard.py --source ../ScotlandYardRL/ScotlandYardRL_V0.11
conda run -n pytorch python tools/validate_scotlandyard.py --source ../ScotlandYardRL/ScotlandYardRL_V0.11
node --test tests/test_scotlandyard.mjs
conda run -n pytorch python tests/browser_scotlandyard_smoke.py --url http://127.0.0.1:8000
```

导出脚本从 AZ run manifest 和 PPO checkpoint 自带配置重建网络，严格加载原始 state_dict，检查两者规则一致，使用 opset 17、固定 batch=1 导出。验证原生模型、导出包装器和 ONNX 的动作与顺序一致，包括耗尽车票的强制停留场景。规则验证在 40 局中逐步对照 Python 原裁判和模型公开输入（1,654 次状态变化）。已通过 AZ 95 次、PPO 93 次原生/ONNX 决策一致性验证。浏览器测试还执行 ONNX 固定输入回归、实际对局、撤回、恢复、模型切换、自动开局、真实下载进度、失败重试和移动布局检查。

前端事实源位于 `public/scotlandyard/`：`engine.js` 是规则移植；`local-api.js` 在浏览器内实现原 UI 所需的状态接口，`/api/games` 字符串仅用作本地方法分发，不向服务器请求。`inference-worker.js` 负责校验并运行模型。修改 V0.11 规则或输入协议后须重新导出并运行对照测试。
