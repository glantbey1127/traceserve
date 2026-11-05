# GitHub 发布指南（Windows PowerShell）

- 目标仓库：https://github.com/glantbey1127/traceserve
- 本地项目：E:\codex\traceserve
- 可见性：Public
- 默认分支：main

本文给出首次上传和以后更新时的完整流程。命令应在 PowerShell 中逐段执行，上一段成功后再继续；文档中的步骤本身不代表已经完成上传。

## 1. GitHub 上创建空仓库

登录 glantbey1127，打开 [创建仓库页面](https://github.com/new)：

| 项目 | 填写内容 |
| --- | --- |
| Owner | glantbey1127 |
| Repository name | traceserve |
| Description | KV-aware LLM inference with CUDA Graph policies, streaming APIs and request-level observability, based on nano-vLLM. |
| Visibility | Public |
| Add a README file | 不勾选，本地已有 README.md |
| Add .gitignore | 不添加，本地已有 |
| Choose a license | 不添加，本地已有原 MIT 许可证 |

点击 Create repository。需要创建的是空仓库，以便直接推送本地提交；这一点与 [GitHub 官方本地项目导入流程](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) 一致。

README 文件的规范名称是 README.md，把它放在项目根目录即可随代码一起上传，GitHub 会自动在仓库首页展示。无需额外创建 read.me 文件。[README 官方说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)

## 2. 进入正确的本地目录

~~~powershell
Set-Location -LiteralPath 'E:\codex\traceserve'
git --version
git rev-parse --show-toplevel
git status --short
~~~

show-toplevel 应指向 E:/codex/traceserve，避免把 E:\codex 下的其他项目一起提交。

如果 Git 提示 detected dubious ownership，通常是项目由其他 Windows 账户（例如 Codex 沙箱账户）创建。确认这是本项目后，在你自己的 PowerShell 中将这个准确目录加入当前账户的 Git 信任列表：

~~~powershell
git config --global --add safe.directory "E:/codex/traceserve"
~~~

然后重试原命令。此配置需要由运行 git push 的 Windows 账户执行。

此项目已初始化 Git。只有复制到另一个完全没有 .git 的目录时，才需要运行：

~~~powershell
git init -b main
~~~

## 3. 配置本仓库的提交身份

下面使用该 GitHub 账号的 noreply 邮箱，避免将私人邮箱写进公开提交；配置只作用于当前仓库：

~~~powershell
git config --local user.name "glantbey1127"
git config --local user.email "336104078+glantbey1127@users.noreply.github.com"
git config --get user.name
git config --get user.email
~~~

这组身份属于 glantbey1127。如果以后更换 GitHub 账号，应改用对应账号 Settings → Emails 页面给出的 noreply 邮箱。

提交身份和登录身份是两件事：user.name / user.email 只标注作者，不会让 Git 自动取得推送权限。

## 4. 本地验证与提交内容

如果还没建立开发环境，先在当前目录运行：

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q nanovllm traceserve tests examples
~~~

没有设置 TRACESERVE_TEST_MODEL 时，2 项 GPU 用例跳过属于预期行为，不代表真实 GPU 执行已经通过。已完成的验证和环境见 [PROJECT_STATUS.md](../PROJECT_STATUS.md)。

查看准备加入提交的文件：

~~~powershell
git add --dry-run .
~~~

正常应包括源码、测试、示例、docs、README.md、UPSTREAM.md、LICENSE 和 .github/workflows/tests.yml。.gitignore 会排除虚拟环境、构建目录、运行 trace、模型权重和本地 .env 文件。

本项目的来源说明和原作者 MIT 许可证需要保留。README 已注明上游代码与本项目工程改造的边界。

## 5. 生成首次提交

如果运行 git log -1 --oneline 已能看到首次提交，跳过本步骤，继续连接远程仓库。

~~~powershell
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Initial TraceServe implementation based on nano-vLLM"
git branch -M main
git log -1 --oneline
~~~

commit 保存的是本地版本；执行到这里，代码仍未上传 GitHub。

## 6. 连接远程仓库

先查看是否已经有 origin：

~~~powershell
git remote -v
~~~

没有 origin 时添加：

~~~powershell
git remote add origin https://github.com/glantbey1127/traceserve.git
~~~

已有 origin、但地址不是目标仓库时才执行：

~~~powershell
git remote set-url origin https://github.com/glantbey1127/traceserve.git
~~~

再次确认目标：

~~~powershell
git remote -v
~~~

fetch 和 push 都应指向 glantbey1127/traceserve.git。

## 7. 登录并推送

Windows 上安装 Git for Windows 后，通常可以使用 Git Credential Manager 通过浏览器登录。

~~~powershell
git push -u origin main
~~~

如果出现登录页面，确认登录的是 glantbey1127，并在 GitHub 页面完成授权。不要把 GitHub 账户密码填入旧式 Git 密码提示；HTTPS Git 使用凭据管理器、Token 等方式认证。[GitHub 认证说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)

需要单独发起登录时，在你自己的普通 PowerShell 窗口运行：

~~~powershell
git credential-manager github login --username glantbey1127 --browser
git push -u origin main
~~~

如果运行环境无法打开浏览器，可使用设备登录，按终端显示的地址和一次性代码在浏览器完成授权：

~~~powershell
git credential-manager github login --username glantbey1127 --device
git push -u origin main
~~~

如果 Codex、远程或受限会话提示无法访问 wincredman，优先到本机桌面的普通 PowerShell 窗口完成登录与推送。不要因此把 Token 写入 README、远程 URL 或源码，也不要将凭据改为明文存储。

## 8. 核对上传结果

~~~powershell
git status
git log -1 --oneline
git ls-remote origin refs/heads/main
~~~

本地 main 的 commit SHA 应与远程 main 一致。随后打开：

- [仓库首页](https://github.com/glantbey1127/traceserve)
- [README](https://github.com/glantbey1127/traceserve/blob/main/README.md)
- [GitHub Actions](https://github.com/glantbey1127/traceserve/actions)

检查首页展示 README，源码、测试和 LICENSE 都在，并查看 CPU CI 的实际结果。CI 只验证 CPU / API / 构建，不验证真实 CUDA 推理。

可在 About 区域填写上面的 Description，Topics 可使用 llm-inference、kv-cache、cuda-graphs、continuous-batching、pytorch、triton、qwen3、ai-infra。

## 9. 以后修改代码时的更新流程

工作区干净时先同步远程：

~~~powershell
Set-Location -LiteralPath 'E:\codex\traceserve'
git status
git pull --ff-only
~~~

修改完成并运行相应检查后，查看、提交和推送：

~~~powershell
git diff
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Describe the actual change"
git push
~~~

如果只修改 README，可将 git add . 换成 git add README.md。GitHub 首页会随 main 分支上的 README 更新，不需要额外发布操作。

## 10. 常见问题

| 提示 | 处理方法 |
| --- | --- |
| detected dubious ownership | 在自己的 PowerShell 中执行 git config --global --add safe.directory "E:/codex/traceserve"，然后重试推送 |
| src refspec main does not match any | 先检查是否有 commit，再检查分支名；完成首次提交后使用 git branch -M main |
| remote origin already exists | 查看 git remote -v，地址错误时使用 git remote set-url origin |
| Repository not found | 确认 traceserve 已在正确账号下创建、URL 拼写正确，登录账号有权限 |
| Authentication failed / 403 | 在本机 PowerShell 重新登录正确账号，确认凭据拥有目标仓库写权限 |
| non-fast-forward / fetch first | 目标已有提交；先查看远程历史，不要直接强推。首次上传建议使用空仓库；已有项目可 clone 后复制源码并正常提交 |
| GitHub Actions 失败 | 打开失败 job 的日志，修复对应依赖、检查或测试；本地通过不等于云端已通过 |
| 大文件被拒绝 | 检查是否误加入模型权重或环境目录。首次、尚未推送的提交可取消这些文件的跟踪后修正提交；本项目不需要上传模型权重 |
| README 没有正常展示 | 确认文件名是 README.md，并位于 main 分支仓库根目录 |

远程 README 被改过、本地又有提交时，应正常获取和整合历史，不使用强推覆盖现有内容。
