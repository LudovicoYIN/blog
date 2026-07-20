---
author: Ludovico
pubDatetime: 2026-07-20T10:30:00+08:00
title: project 的 GitLab CI/CD 与 Python 包发版流程
featured: false
draft: false
tags:
  - CI/CD
description: 记录 project 从专用 GitLab Runner、部署回归，到 wheel 构建、隔离验证、包发布和 Release 创建的完整流程。
---

project 的 CI/CD 不是把本地打出来的 wheel 上传一下。部署后端依赖 GPU、交叉编译工具链和厂商 SDK，发布产物还包含编译后的 Python 扩展。流水线需要同时解决两件事：测试环境可复现，发布包确实来自通过验证的构建产物。

最终流程如下：

```text
Merge Request / 默认分支
        |
        v
部署回归测试

vX.Y.Z tag
        |
        v
构建 wheel -> 校验 wheel -> 从隔离目录运行测试
        |
        v
上传内部 PyPI Registry -> 创建 GitLab Release
```

## 为什么需要专用 Runner

普通共享 Runner 适合单元测试，不适合 Project 的部署回归。部署测试会调用 GPU 环境、Android NDK 和模型转换 SDK；这些依赖体积大、安装受许可限制，也不适合跟着每次 CI 临时下载。

因此把运行环境拆成两部分：

- Runner 主机保存受控的 SDK 与 NDK，只给需要它们的 CI 容器只读挂载。
- CI 镜像固定 Python、CUDA、编译器和通用 Python 依赖。

项目任务统一使用一个专用 tag，例如 `project-docker`。GitLab 只会把带相同 tag 的任务调度到这个 Runner。这样能避免无关项目占用环境，也不会让缺 SDK 的共享 Runner 接到任务。

Runner 使用 Docker executor。注册时需要打开 Docker 访问和 GPU 支持，SDK、NDK 的挂载在 Runner 配置中完成，不写进仓库。一个脱敏后的配置结构如下：

```toml
[[runners]]
  name = "model-toolchain-runner"
  url = "https://gitlab.example.com/"
  token = "<runner-registration-token>"
  executor = "docker"
  tags = ["project-docker"]

  [runners.docker]
    image = "registry.example.com/project-ci:py310-cuda"
    privileged = true
    gpus = "all"
    volumes = [
      "/cache",
      "/srv/sdk/qnn:/opt/sdk/qnn:ro",
      "/srv/sdk/android-ndk:/opt/android-ndk:ro"
    ]
```

这里的关键不是具体路径，而是边界：SDK 在主机上维护；容器内路径固定；挂载只读；CI 通过环境变量引用容器内路径。任务开始时先检查 NDK 可执行文件和 SDK 目录是否存在，配置错了直接失败，不让测试跑到一半才报工具找不到。

基础镜像使用固定版本，包含 Python 3.10、CUDA 开发环境、C/C++ 编译工具、LLVM 运行库和测试依赖。Python 版本被锁定是因为 wheel 构建会生成 CPython 扩展，解释器 ABI 不能漂移。

## 触发规则

流水线只接受三类触发：

- Merge Request：拦截代码合并前的部署回归。
- 默认分支提交：验证合并后的主线状态。
- `vX.Y.Z` 格式的 tag：执行正式发版。

前两类只运行测试；只有 tag 触发构建、发布和 Release。这样普通提交不会污染包仓库，也不会意外生成版本号。

```yaml
workflow:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
    - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'
    - if: '$CI_COMMIT_TAG =~ /^v[0-9]+\.[0-9]+\.[0-9]+$/'
```

版本号的唯一来源仍是 `pyproject.toml`。tag 触发后，流水线会取出 `v` 后的版本号，与包元数据逐字比较；不一致就停止。这样可以防止 `v0.2.12` 实际上传了 `0.2.11` 的 wheel。

## 测试阶段

测试阶段运行部署模块的回归集。它覆盖模型转换、量化路径、产物校验和精度检查；依赖 SDK 的用例在专用 Runner 上执行，不把它们降级为本地人工检查。

测试结果以 JUnit XML 作为 artifact 上传，失败时同样保留一段时间。GitLab 可以直接显示失败用例，后续排查不需要重新跑一遍才能拿到报告。

```yaml
run-deploy-tests:
  stage: test
  image: "registry.example.com/project-ci:py310-cuda"
  script:
    - test -x "$ANDROID_NDK_ROOT/ndk-build"
    - test -d "$QNN_SDK_ROOT"
    - python -m pytest tests/tdeploy -v --junitxml=reports/tdeploy.xml
  artifacts:
    when: always
    reports:
      junit: reports/tdeploy.xml
```

## 打包与隔离验证

tag pipeline 的第一个发布阶段只做一件事：构建并验证 wheel。

构建后有三层检查：

1. 用 Python 的 zip 校验检查 wheel 是否损坏。
2. 校验 tag 版本和项目版本一致。
3. 将 wheel 安装到临时目录，从另一个临时工作目录执行部署测试。

第三步最重要。测试进程不能从源码目录导入 `project`，否则即使 wheel 漏打文件、动态库缺失或打进了旧代码，测试也可能误用当前 checkout 而通过。隔离目录配合 `--import-mode=importlib`，能确认测试实际加载的是刚刚构建的 wheel。

```bash
python -m build --wheel --outdir dist
python -m zipfile -t dist/*.whl

WHEEL_SITE="$(mktemp -d)"
python -m pip install --no-deps --target "$WHEEL_SITE" dist/*.whl
cd "$(mktemp -d)"
PYTHONPATH="$WHEEL_SITE:$CI_PROJECT_DIR" \
  python -m pytest --import-mode=importlib \
  "$CI_PROJECT_DIR/tests/tdeploy" -v
```

构建产物和 wheel 测试报告作为同一 job 的 artifact 保存。后面的发布 job 只消费这份 artifact，不重新构建。这保证被上传的文件就是通过验证的文件。

## 发布与 Release

发布 job 使用 GitLab job token 上传到项目的 PyPI Package Registry。认证信息来自 CI 环境变量，不提交到仓库，也不出现在安装文档的命令里。

上传成功后才创建 GitLab Release。Release 附加 wheel 的下载链接，链接按构件校验值定位具体文件，而不是只放包索引页。使用方可以从 Release 找到准确版本，也可以通过内部 Registry 安装固定版本。

Release 创建遵循不可变原则：首次创建成功即结束；如果该 tag 已经有 Release，流水线不覆盖原说明或替换已有资产。版本需要修改时，应当递增版本号并重新打 tag，而不是重发同一个版本。

```text
vX.Y.Z
  -> 构建并验证 project-X.Y.Z-<abi>-<platform>.whl
  -> 上传内部 PyPI Registry
  -> 创建同名 GitLab Release
  -> Release 链接指向该 wheel
```

## 发版操作

发版前先完成代码合并，并确认默认分支的部署回归已通过。随后按下面顺序操作：

1. 修改 `pyproject.toml` 中的版本号。
2. 提交并合并版本提交。
3. 在该提交上创建 `vX.Y.Z` tag，版本号必须与 `pyproject.toml` 相同。
4. 推送 tag，等待 tag pipeline 完成。
5. 在 GitLab Package Registry 确认包已出现，在 Release 页面确认 wheel 链接可下载。

本机构建只用于开发和排查，正式发布只认 tag pipeline。这样发布环境、测试环境和最终 wheel 都有可追溯记录，也避免开发机上的缓存、SDK 版本或未提交文件混进发布包。
