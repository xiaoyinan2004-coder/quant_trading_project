# 云端量化交易项目环境搭建完成报告

## A. 已完成内容

### 1. Git 仓库克隆
- ✅ 仓库已克隆到 `/workspace`
- ✅ 源仓库: https://github.com/xiaoyinan2004-coder/quant_trading_project.git
- ✅ 项目结构完整，包含所有必需文件

### 2. Python 虚拟环境
- ✅ 虚拟环境位置: `/workspace/.venv`
- ✅ Python 版本: 3.11.1
- ✅ pip 版本: 26.0.1

### 3. 目录结构创建
- ✅ `/workspace/artifacts/` - 训练产物目录
- ✅ `/workspace/logs/` - 日志目录
- ✅ `/workspace/data/raw/` - 原始数据目录
- ✅ `/workspace/data/processed/` - 处理后数据目录
- ✅ `/workspace/data/cache/` - 缓存目录

### 4. Python 包安装
- ✅ PyTorch 2.11.0+cu130 (CUDA 13.0 支持)
- ✅ LightGBM 4.6.0
- ✅ XGBoost 3.2.0
- ✅ scikit-learn 1.8.0
- ✅ pandas 3.0.2
- ✅ numpy 2.4.4
- ✅ tensorboard, accelerate, tqdm

---

## B. 关键版本信息

| 软件/库 | 版本 |
|---------|------|
| 操作系统 | Ubuntu (Linux) |
| Python | 3.11.1 |
| pip | 26.0.1 |
| git | 已安装 |
| PyTorch | 2.11.0+cu130 |
| CUDA 可用性 | ✅ True |
| GPU 型号 | Tesla T4 (15.36 GB) |
| CUDA 版本 | 13.0 |
| NVIDIA 驱动 | 580.65.06 |
| LightGBM | 4.6.0 |
| XGBoost | 3.2.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |

---

## C. 当前是否可训练

### ✅ 是，环境已就绪

- **LightGBM / XGBoost**: ✅ 可运行
- **PyTorch**: ✅ 可运行
- **GPU 训练**: ✅ 可用 (Tesla T4, CUDA 13.0)
- **训练脚本**: ✅ 可执行

### 训练命令示例

```bash
# 方法 1: 直接运行训练脚本
cd /workspace
source .venv/bin/activate
export PYTHONPATH=/workspace
python scripts/train_m3net_stage1.py \
  --daily-dir data/processed/daily \
  --minute-dir data/processed/minute \
  --backend lightgbm \
  --output artifacts/m3net_stage1.joblib

# 方法 2: 使用 pull_and_train.sh 脚本
cd /workspace
bash scripts/pull_and_train.sh /workspace

# 方法 3: 使用 tmux 运行长时间训练
cd /workspace
tmux new -s quant_train
source .venv/bin/activate
export PYTHONPATH=/workspace
python scripts/train_m3net_stage1.py \
  --daily-dir data/processed/daily \
  --backend lightgbm \
  --output artifacts/m3net_stage1.joblib
```

---

## D. 待补充项

### 1. 数据文件 ⚠️
- **状态**: 数据目录已创建，但缺少实际数据文件
- **需要**: 
  - `/workspace/data/processed/daily/` - 日频 OHLCV 数据
  - `/workspace/data/processed/minute/` - 分钟级数据（可选）

### 2. Git 配置 ⚠️
- **当前状态**: 已克隆，但未配置用户信息
- **如需推送代码**: 需要配置 `git config --global user.name` 和 `user.email`

---

## E. 遇到的问题和解决方案

### 问题 1: PyTorch CUDA 版本匹配
- **问题**: 系统 CUDA 13.0，PyTorch 官方最高支持 CUDA 12.1
- **解决**: 安装了 CUDA 13.0 兼容的 PyTorch 2.11.0+cu130，GPU 可正常使用

### 问题 2: Python 包版本兼容性
- **问题**: akshare 1.13.2 和 pandas-ta 0.3.14b0 在 Python 3.11 中不可用
- **解决**: 更新为可用版本，跳过了非核心包

### 问题 3: 模块导入路径
- **问题**: 直接运行脚本时无法导入 models 模块
- **解决**: 运行时设置 `export PYTHONPATH=/workspace`

---

## F. 环境验证命令

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 Python 环境
cd /workspace
source .venv/bin/activate
python --version

# 验证 PyTorch 和 GPU
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 验证 ML 库
python -c "import lightgbm, xgboost, sklearn; print('All OK')"

# 验证模型导入
export PYTHONPATH=/workspace
python -c "from models.m3net import M3NetStage1Model, M3NetStage1Config; print('M3-Net OK')"

# 查看已安装包
pip list | grep -E "(torch|lightgbm|xgboost|sklearn|pandas)"
```

---

## G. 工作流程建议

1. **本地开发**: 在本地修改代码
2. **推送代码**: `git push origin main`
3. **云端拉取**: `cd /workspace && git pull origin main`
4. **GPU 训练**: 使用上述训练命令
5. **查看结果**: 训练产物保存在 `/workspace/artifacts/`

---

## H. tmux 使用提示

```bash
# 创建新会话
tmux new -s quant_train

# 分离会话: Ctrl+B, D

# 重新连接
tmux attach -t quant_train

# 列出会话
tmux ls

# 删除会话
tmux kill-session -t quant_train
```

---

**环境搭建完成时间**: 2026-04-01 09:11

**状态**: ✅ 环境已就绪，等待数据文件即可开始训练
