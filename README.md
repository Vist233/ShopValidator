# 店判 ShopValidator

店判把餐饮经营决策拆成可核对的事实、待验证的假设和有停止线的小实验。现有网站保留项目介绍和一个虚构的餐饮案例。

- [项目展示](https://shopvalidator.zhangyvjing.com/)
- [案例演示](https://demo.shopvalidator.zhangyvjing.com/)

## 静态网站

`public/` 是网站的完整发布目录。用 Wrangler 运行 `wrangler dev` 可在本地预览页面。确认 `wrangler deploy --dry-run` 的发布内容后，用 `wrangler deploy` 更新 Cloudflare 上的静态文件。

## 历史实现

仓库中的 `src/`、`scripts/`、`tests/` 和研究资料保留供阅读。原系统的问诊、分析和排行榜已停止提供线上服务。静态发布只读取 `public/`。

[查看静态化前的 README](https://github.com/Vist233/ShopValidator/blob/40adf0588efd3d2f5f38f4dcef1018c489e6a66b/README.md)
