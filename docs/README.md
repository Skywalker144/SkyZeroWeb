# SkyZeroWeb 文档索引

执行仓库任务时，先阅读根目录的 [`RULES.md`](../RULES.md) 和 [`README.md`](../README.md)，再按任务进入下列文档。

| 任务 | 文档或事实源 |
|---|---|
| 本地开发、架构、模型更新、部署约定 | [`README.md`](../README.md) |
| ScotlandYard 浏览器实现、模型与验证 | [`scotlandyard.md`](scotlandyard.md) |
| Hex 模型导出、规则与浏览器验证 | [`README.md`](../README.md#hex-模型)、[`test_hex.mjs`](../tests/test_hex.mjs)、[`browser_hex_smoke.py`](../tests/browser_hex_smoke.py) |
| 围棋模型、原版规则 WASM 与浏览器验证 | [`go.md`](go.md) |
| 中国象棋模型、原版规则 WASM 与浏览器验证 | [`xiangqi.md`](xiangqi.md) |
| 德州扑克、原版策略服务与 API 配置 | [`poker.md`](poker.md) |
| 排行榜与 Cloudflare D1 运维 | [`leaderboard.md`](leaderboard.md) |
| Cloudflare Pages 行为 | [`wrangler.toml`](../wrangler.toml)、[`public/_headers`](../public/_headers)、[`public/_redirects`](../public/_redirects)、[`public/_routes.json`](../public/_routes.json) |
| 游戏和搜索逻辑验证 | [`tests/`](../tests/) |
| 模型目录与版本信息 | [`public/gomoku/models/manifest.json`](../public/gomoku/models/manifest.json)、[`public/scotlandyard/models.json`](../public/scotlandyard/models.json) |

新增文档时，应先判断信息能否由代码、配置或测试直接表达。只有稳定约定、操作入口或代码中不存在的唯一信息才写入文档，并在本索引中增加任务路由。
