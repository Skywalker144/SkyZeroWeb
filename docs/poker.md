# 德州扑克

页面入口为 [`public/poker/`](../public/poker/)，Cloudflare API 位于 [`functions/api/poker/`](../functions/api/poker/)。裁判、筹码和手内记录保存在每个浏览器的页面内存；刷新会结束练习。

作者原版模块位于 [`services/poker/vendor/`](../services/poker/vendor/)，版本固定在 [`REVISION`](../services/poker/REVISION)，许可见 [`THIRD_PARTY_NOTICES.md`](../services/poker/THIRD_PARTY_NOTICES.md)。引擎必须使用 Linux x86_64 CPython 3.9。独立服务由 [`engine.py`](../services/poker/engine.py) 和 [`skyzeroweb-poker.service`](../services/poker/skyzeroweb-poker.service) 定义。

Cloudflare Pages 的生产与预览环境都需要 `POKER_ENGINE_URL` 和 `POKER_ENGINE_TOKEN`。服务端令牌保存于被 Git 忽略的 `.runtime/poker-engine.env`；令牌不进入静态页面。用户的 DeepSeek API key 仅经 Pages 转发给 DeepSeek，不发送给策略服务，不落盘保存。

## 运行与验证

```sh
systemctl --user status skyzeroweb-poker
journalctl --user -u skyzeroweb-poker -n 30
POKER_PYTHON="$HOME/.local/bin/python3.9" node --test tests/test_poker_*.mjs
```

页面可通过 `npm run dev` 检查；完整 API 使用 `npm run dev:cloudflare`，本地配置放在被忽略的 `.dev.vars`。线上检查 `/api/poker/health`；它应返回作者引擎、Python 版本和固定仓库版本。

Linux 使用 Tailscale Funnel 提供策略服务的 HTTPS 入口 `https://sky-5090.taila816d7.ts.net`，检查命令为 `tailscale funnel status`。启动入口需要在 Linux 执行 `sudo /usr/bin/tailscale funnel --bg --yes --https=443 http://127.0.0.1:8173`；后台配置持续有效。用户服务已启用开机运行与 linger。Linux 服务与公网 HTTPS 入口必须持续在线。默认不使用网站所有者的 DeepSeek key，玩家自行填写自己的 key。
