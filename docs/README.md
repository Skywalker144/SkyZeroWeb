# SkyZeroWeb 文档索引

执行仓库任务时，先阅读根目录的 [`RULES.md`](../RULES.md) 和 [`README.md`](../README.md)，再按任务进入下列文档。

| 任务 | 文档或事实源 |
|---|---|
| 本地开发、架构、模型更新、部署约定 | [`README.md`](../README.md) |
| ScotlandYard 浏览器实现、模型与验证 | [`scotlandyard/README.md`](../scotlandyard/README.md) |
| 排行榜与 Cloudflare D1 运维 | [`LEADERBOARD_SETUP.md`](../LEADERBOARD_SETUP.md) |
| 历史审计记录 | [`AUDIT_2026-07.md`](../AUDIT_2026-07.md) |
| Cloudflare Pages 行为 | [`wrangler.toml`](../wrangler.toml)、[`_headers`](../_headers)、[`_redirects`](../_redirects)、[`_routes.json`](../_routes.json) |
| 游戏和搜索逻辑验证 | [`tests/`](../tests/) |
| 模型目录与版本信息 | [`models/manifest.json`](../models/manifest.json)、[`scotlandyard/models.json`](../scotlandyard/models.json) |

新增文档时，应先判断信息能否由代码、配置或测试直接表达。只有稳定约定、操作入口或代码中不存在的唯一信息才写入文档，并在本索引中增加任务路由。
