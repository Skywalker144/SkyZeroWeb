# 中国象棋

页面入口为 [`public/xiangqi/`](../public/xiangqi/)，模型来自 `../EtaZero-lab/EtaZero_V3/data/az_go_xiangqi_5h/az_xiangqi`。发布模型、SWA 来源、源文件哈希和数值验证结果以 [`manifest`](../public/xiangqi/models/manifest.json) 为准；目录中的 `5h` 是运行名，导出器选择已发布的最新模型。

规则和历史观测直接编译训练侧的 XiangqiState 与 Pikafish 规则代码。源码快照及逐文件来源哈希位于 [`tools/xiangqi/`](../tools/xiangqi/)，页面提供 GPLv3 许可证和对应构建源码包。浏览器不使用 Pikafish 的搜索、NNUE 或权重。棋盘、合法着法、历史编码和终局裁定只来自该 WASM；主线程负责交互，Worker 负责规则和 ONNX 推理。

浏览器使用与 Hex 共用的 [`轻量单叶 PUCT`](../public/shared/puct-search.js)，复用训练侧的红黑坐标变换和随机水平镜像。搜索未复现训练侧的图搜索、FPU、子树价值偏差、LCB 等修正，因此不代表原生引擎同等棋力。思考时间与访问上限见 [`Worker`](../public/xiangqi/worker.js)。

## 更新与验证

需要 Conda `pytorch` 环境中的 PyTorch、ONNX、ONNX Runtime、Playwright，以及 PATH 中的 Emscripten `em++`。规则编译与模型导出是离线准备步骤；Cloudflare Pages 仍直接发布 `public/`。

```bash
conda run -n pytorch python tools/gen_xiangqi_fixture.py
conda run -n pytorch python tools/export_xiangqi_onnx.py
conda run -n pytorch python tools/build_xiangqi_rules.py
node --test tests/test_xiangqi.mjs tests/test_hex.mjs tests/test_project_structure.mjs
npm run dev
conda run -n pytorch python tests/browser_xiangqi_smoke.py
```

更新训练侧规则快照时，使用 `tools/build_xiangqi_rules.py --sync-source ../EtaZero-lab/EtaZero_V3`，重新生成原生 fixture 并运行对拍。普通构建使用仓库内快照，不要求相邻训练仓库。模型导出器验证原 TorchScript、重建网络和 ONNX；浏览器测试覆盖双侧视角与镜像、合法着法屏蔽、对弈、取消搜索、悔棋、首页入口及手机布局。

规则、模型文件和 manifest 使用内容哈希关联；更新 JavaScript 或 CSS 后，同步更新 HTML、Worker 和主线程加载 URL 的版本参数。修改公共 PUCT 时同时运行 Hex 逻辑测试与 [`浏览器测试`](../tests/browser_hex_smoke.py)。
