---
author: Ludovico
pubDatetime: 2026-06-06T16:00:00Z
title: Hermes Agent + Tailscale：免打扰远程控制
featured: false
draft: false
tags:
  - agent
description: 用 Hermes Agent 的 YOLO 模式搭配 Tailscale，让 AI 助手无缝 SSH 接管你的任何设备。
---

## 问题

Hermes Agent 默认有安全审批机制，每次执行敏感命令都要手动确认。这在 QQ/Telegram 上操作时很烦——消息发出去，人不在手机边就卡住了。

## 解法：YOLO 模式

Hermes 提供 `--yolo` 参数，跳过所有危险命令确认：

```bash
hermes --yolo
```

也可以永久启用：

```bash
hermes config set yolo true
# 或通过环境变量
export HERMES_YOLO=1
```

## Tailscale 组网

Tailscale 基于 WireGuard 的零配置 Mesh VPN，装好后设备自动互联。

```bash
# 安装
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# 查看网络状态
tailscale status
```

## 跨设备 SSH

装上 Tailscale 后，用内网 IP 直接 SSH：

```bash
# 在 Hermes 里一句话搞定
ssh root@100.x.x.x "nvidia-smi"
```

Hermes Agent 本身支持终端后台运行 + 完成通知，适合跨设备执行耗时任务。

## 总结

| 组件 | 功能 |
|------|------|
| Hermes YOLO | 跳过审批弹窗 |
| Tailscale | 零配置 Mesh VPN |
| SSH | 跨设备远程执行 |

三样加起来 = 一个能自由操作你所有机器的 AI 助手。
