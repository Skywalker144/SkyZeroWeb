# 围棋

页面入口为 [`public/go/`](../public/go/)，模型来源、发布轮次、SWA checkpoint、文件哈希与导出数值误差以 [`manifest`](../public/go/models/manifest.json) 为准。模型来自 EtaZero V4 的 `az_pcr_go/runs/no_pcr_225`，导出器读取运行的已提交发布状态。

规则与 V7 历史特征直接使用训练侧原版 C++，源码快照与来源哈希在 [`tools/go/`](../tools/go/)。页面使用 manifest 中的棋规与贴目，双方连续停一手后由原版引擎计分；结束前应完成死子提取与边界。浏览器使用公共单叶 PUCT，搜索强度不等同于训练侧完整搜索。

```bash
conda run -n pytorch python tools/build_go_rules.py --sync-source ../EtaZero-lab/EtaZero_V4 --native
conda run -n pytorch python tools/export_go_onnx.py
conda run -n pytorch python tools/build_go_rules.py
node --test tests/test_go.mjs
conda run -n pytorch python tests/browser_go_smoke.py
```

普通构建只使用仓库内的源码快照，需要 Emscripten `em++`；原生对拍数据生成需要 `g++`。浏览器测试通过 HTTP 访问页面。离线验证时，可将官方 ONNX Runtime Web 1.17.0 分发文件放在本地目录，设置 `SKYZERO_ORT_DIR`，浏览器测试会直接加载这些真实运行时文件。

四个棋类页面共用 [`board-shell.css`](../public/shared/board-shell.css) 与 [`board-controls.js`](../public/shared/ui/board-controls.js)；Go 和象棋共用 [`board-session.js`](../public/shared/board-session.js)。新增通用交互应在公共模块维护。

规则源码包含 KataGo 的 MIT 代码与训练侧象棋依赖 Pikafish 的 GPLv3 代码，构建器同步生成对应源码包和许可证链接。
