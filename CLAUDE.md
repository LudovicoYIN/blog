# Ludovico's Blog (AstroPaper)

基于 [Astro](https://astro.build) + [AstroPaper](https://astro-paper.pages.dev/) — 部署在 GitHub Pages。

## 发布新文章

文章存放在 `src/content/posts/`。创建 `.md` 文件，推送到 `main` 分支即可。

### 1. 写文章

在 `src/content/posts/<slug>.md` 创建文件，使用以下 frontmatter：

```markdown
---
author: Ludovico
pubDatetime: 2026-05-28T12:00:00Z
title: 文章标题
featured: false
draft: false
tags:
  - 标签1
  - 标签2
description: 简短描述，用于预览和 SEO。
---

Markdown 正文内容。
```

必填：`title`、`pubDatetime`、`description`。
可选：`featured`（是否在首页精选区展示）、`draft`（草稿不发布）、`tags`（默认 `["others"]`）、`ogImage`。

### 2. 图片

图片放在 `src/assets/images/` 下，文章中引用：

```markdown
![描述](@/assets/images/my-image.png)
```

子目录以下划线 `_` 开头则不参与 URL 路由，适合放图片等资源。例如 `src/content/posts/_assets/photo.png` 不会被发布为文章。

### 3. 发布

```bash
git add src/content/posts/<slug>.md
git commit -m "post: <标题>"
git push origin main
```

GitHub Actions 自动构建部署，约 1 分钟上线。

## 本地开发

```bash
npm run dev      # http://localhost:4321
npm run build    # 构建到 dist/
```

## 首次推送前需要修改的配置

- `astro-paper.config.ts`：`site.url` 改为实际的 GitHub Pages URL（`https://ludovicoyin.github.io`）
- `astro-paper.config.ts`：`socials` 中的 GitHub 链接
- `astro.config.ts`：`base` 改为仓库名（如 `/blog`，若仓库名是 `<user>.github.io` 则改为 `/`）
