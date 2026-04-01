# Ubuntu 云端服务器环境搭建工作文档

本文件用于发给云端服务器自带的 AI。请该 AI 阅读后，直接在 Ubuntu 服务器上完成量化项目的环境搭建、代码拉取、训练工作流准备与验证。

## 1. 你的角色

你现在是这台 Ubuntu 云端服务器上的运维与开发助手。你的任务是为一个本地开发、云端训练的量化交易项目搭建完整运行环境，使其尽量与本地项目保持一致，并能够承担后续 GPU 深度学习训练任务。

请严格按文档执行，并在完成后输出：

1. 已执行步骤
2. 关键软件版本
3. GPU / CUDA / PyTorch 可用性
4. git 同步状态
5. 训练脚本是否可运行
6. 遇到的问题和未完成项

## 2. 项目背景

这是一个量化交易项目，当前本地项目已经具备：

- A 股日频与分钟级数据模块
- `LightGBM / XGBoost` 多因子模型
- `M3-Net Stage 1` 轻量实现
- 本地开发 + 云端训练工作流

当前项目中已经存在以下关键文件，请优先使用：

- `requirements.txt`
- `requirements-cloud.txt`
- `scripts/setup_cloud_env.sh`
- `scripts/pull_and_train.sh`
- `scripts/train_m3net_stage1.py`
- `DEPLOY_TO_CLOUD.md`

## 3. 总目标

请在这台 Ubuntu 服务器上完成以下目标：

1. 安装基础开发环境
2. 安装 git / Python / venv / tmux
3. 安装项目依赖
4. 安装 GPU 训练依赖
5. 从远端仓库拉取项目代码
6. 建立适合训练的目录结构
7. 验证 `python`、`git`、`torch`、GPU 是否正常
8. 验证项目训练脚本是否具备运行条件
9. 给出最终环境总结

## 4. 工作原则

请遵守：

- 优先使用项目中已有脚本，不要重复造轮子
- 不要随意改动项目业务代码，除非是为了解决环境兼容问题
- 若缺少仓库地址、分支名、数据路径，请明确标记为“需要用户补充”
- 训练产物不要放入 git 跟踪
- 优先采用 `Ubuntu 22.04` 通用做法
- 所有安装步骤尽量可复现

## 5. 需要你获取或确认的信息

如果下面信息尚未提供，请列为待补充项：

- 远端 git 仓库地址
- 默认分支名，默认使用 `main`
- 服务器 GPU 型号
- CUDA 版本
- 是否已有 NVIDIA 驱动
- 数据存放目录
- 是否需要把分钟级数据挂载到独立磁盘

如果这些信息无法自动获取，请先完成其余步骤，再在总结中列出来。

## 6. 目标目录约定

请按以下方式组织：

```bash
~/quant_trading_project
~/quant_trading_project/.venv
~/quant_trading_project/artifacts
~/quant_trading_project/logs
~/quant_trading_project/data
~/quant_trading_project/data/raw
~/quant_trading_project/data/processed
~/quant_trading_project/data/cache
```

## 7. 基础环境安装任务

请检查并安装：

- `git`
- `python3`
- `python3-venv`
- `python3-pip`
- `build-essential`
- `tmux`
- `wget`
- `curl`
- `htop`

如未安装，请使用类似命令：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential tmux wget curl htop
```

## 8. GPU 环境检查任务

请执行并汇报结果：

```bash
nvidia-smi
python3 --version
which python3
```

如果 `nvidia-smi` 不可用，请说明：

- 是否未安装 NVIDIA 驱动
- 是否实例本身没有 GPU
- 是否仅需 CPU 环境

## 9. 克隆或同步项目代码

如果用户已经提供仓库地址，请执行：

```bash
git clone <REPO_URL> ~/quant_trading_project
cd ~/quant_trading_project
```

如果项目目录已存在，则：

```bash
cd ~/quant_trading_project
git pull origin main
```

如果用户还没有提供仓库地址，请不要伪造地址。请在总结里明确写出需要用户补充仓库地址。

## 10. 项目环境搭建任务

进入项目目录后，优先执行项目已有脚本：

```bash
cd ~/quant_trading_project
bash scripts/setup_cloud_env.sh ~/quant_trading_project
```

如果脚本失败，请按脚本意图手动完成：

1. 创建虚拟环境 `.venv`
2. 激活环境
3. 升级 `pip setuptools wheel`
4. 安装 `requirements.txt`

对应命令参考：

```bash
cd ~/quant_trading_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 11. GPU 深度学习依赖安装任务

项目后续要做 GPU 深度学习训练，因此请额外安装：

- `torch`
- `torchvision`
- `torchaudio`
- `tensorboard`
- `accelerate`

优先根据服务器 CUDA 版本选择正确的 PyTorch 安装方式。

如果服务器使用 CUDA 12.1，可参考：

```bash
source ~/quant_trading_project/.venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install tensorboard accelerate tqdm
```

如果 CUDA 版本不同，请自动选择对应官方 wheel 源，并在总结中说明你选择的版本依据。

## 12. 目录与权限整理

请确保以下目录存在：

```bash
mkdir -p ~/quant_trading_project/artifacts
mkdir -p ~/quant_trading_project/logs
mkdir -p ~/quant_trading_project/data/raw
mkdir -p ~/quant_trading_project/data/processed
mkdir -p ~/quant_trading_project/data/cache
```

## 13. 验证任务

请在虚拟环境激活后执行以下验证：

```bash
cd ~/quant_trading_project
source .venv/bin/activate
python --version
pip --version
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('cuda_available', torch.cuda.is_available())"
python -c "import lightgbm, xgboost, sklearn; print('lightgbm ok, xgboost ok, sklearn ok')"
python -c "import pandas, numpy; print('pandas ok, numpy ok')"
git status
```

如果 `torch.cuda.is_available()` 返回 `False`，请进一步检查：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

并说明原因。

## 14. 训练脚本可用性检查

请检查下列脚本是否存在：

- `scripts/train_m3net_stage1.py`
- `scripts/pull_and_train.sh`

然后执行一次“只检查、不真实训练”的准备动作：

1. 确认 Python 可导入训练模块
2. 不要求真实数据齐全
3. 如果缺少数据目录，只需报告“环境已就绪，等待数据”

可参考执行：

```bash
cd ~/quant_trading_project
source .venv/bin/activate
python -c "from models.m3net import M3NetStage1Model, M3NetStage1Config; print('m3net import ok')"
python scripts/train_m3net_stage1.py --help
```

如果存在实际数据目录：

```bash
bash scripts/pull_and_train.sh ~/quant_trading_project
```

若不存在实际数据，请不要伪造训练数据。

## 15. tmux 长任务支持

请确保服务器支持 `tmux`，并说明用户后续可以用以下方式跑训练：

```bash
cd ~/quant_trading_project
tmux new -s quant_train
source .venv/bin/activate
bash scripts/pull_and_train.sh ~/quant_trading_project
```

## 16. 输出格式要求

完成后请按以下结构输出：

### A. 已完成内容

逐条列出：

- 已安装的软件
- 已创建的目录
- 已配置的虚拟环境
- 已安装的 Python 包
- git 状态

### B. 关键版本信息

至少给出：

- Ubuntu 版本
- Python 版本
- pip 版本
- git 版本
- torch 版本
- CUDA 可用性
- GPU 型号

### C. 当前是否可训练

明确给出结论：

- 是否可运行 `LightGBM / XGBoost`
- 是否可运行 `PyTorch`
- 是否可用 GPU 训练
- 是否仍缺少数据或仓库地址

### D. 未完成项

列出所有缺失信息或阻塞项。

## 17. 不要做的事情

请不要：

- 不经确认修改量化策略逻辑
- 不经确认删除任何文件
- 把大模型权重、分钟级大数据提交进 git
- 编造仓库地址、数据路径或训练结果

## 18. 项目兼容目标

请以“和本地项目工作流匹配”为核心目标，最终希望达到：

- 本地开发
- `git push`
- 云端 `git pull`
- 云端 GPU 训练
- 产出保存在 `artifacts/`

如果可以，请尽量让最终环境直接兼容后续：

- `PyTorch`
- 时序模型
- 深度学习实验
- 更高阶段的 `M3-Net`

