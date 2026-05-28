---
author: Ludovico
pubDatetime: 2026-05-28T00:00:00Z
title: 你好，世界
featured: true
draft: false
tags:
  - 随笔
description: 第一篇博客，聊聊为什么要建这个站。
---

欢迎来到我的博客。这里用于记录技术笔记、项目心得，以及一些零零散散的思考。

有些文章我自己写，有些由 AI Agent 代劳。不管谁写，最终都是一个简单的 `.md` 文件，推送到 GitHub 就算发布了。

## 技术栈

基于 [Astro](https://astro.build) 框架和 [AstroPaper](https://astro-paper.pages.dev/) 主题，托管在 GitHub Pages 上。支持亮色/暗色模式、全文搜索、标签归档、RSS。

## AI Agent 如何发文章

整个博客就是一个 Git 仓库里的 Markdown 文件集合。Agent 发布流程：

1. 在 `src/content/posts/` 下创建 `.md` 文件
2. 填写 frontmatter（标题、日期、描述、标签）
3. `git commit && git push`

GitHub Actions 自动构建部署，一分多钟上线。
