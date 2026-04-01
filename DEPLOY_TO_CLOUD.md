# Local Development + Cloud Training Workflow

This project is now structured for a split workflow:

- local machine: coding, debugging, light experiments
- cloud server: long training jobs, GPU experiments, large backtests

## 1. What git does here

`git` is used to keep the codebase synchronized between your local machine and the cloud server.

It does not move GPU compute by itself. The actual pattern is:

1. write code locally
2. commit and push to remote repository
3. cloud server pulls latest code
4. cloud server runs training scripts
5. trained artifacts are stored outside git

## 2. Files added for this workflow

- `.gitignore`
- `requirements-cloud.txt`
- `scripts/setup_cloud_env.sh`
- `scripts/connect_git_remote.ps1`
- `scripts/pull_and_train.sh`

## 3. Recommended repository policy

Commit into git:

- source code
- configs
- scripts
- documentation
- tests

Do not commit into git:

- raw market data
- processed minute bars
- model checkpoints
- large `joblib`, `pt`, `pth`, `ckpt` files
- logs and tensorboard outputs

## 4. Local setup

From the project root on your Windows machine:

```powershell
git init -b main
git add .
git commit -m "Initial quant trading project"
```

If you already created a remote repository, you can configure it with:

```powershell
.\scripts\connect_git_remote.ps1 -RemoteUrl "git@github.com:YOUR_NAME/YOUR_REPO.git"
git push -u origin main
```

## 5. Cloud server setup

Clone the repository on the Linux server:

```bash
git clone git@github.com:YOUR_NAME/YOUR_REPO.git ~/quant_trading_project
cd ~/quant_trading_project
bash scripts/setup_cloud_env.sh
```

If you need GPU PyTorch, install the correct wheel for the server CUDA version. Example for CUDA 12.1:

```bash
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 6. Training workflow on cloud

Recommended pattern with `tmux`:

```bash
cd ~/quant_trading_project
tmux new -s quant_train
source .venv/bin/activate
bash scripts/pull_and_train.sh ~/quant_trading_project
```

Default training script:

```bash
python scripts/train_m3net_stage1.py \
  --daily-dir data/processed/daily \
  --minute-dir data/processed/minute \
  --backend lightgbm \
  --output artifacts/m3net_stage1.joblib
```

## 7. Data strategy

Best practice is:

- keep code in git
- keep data in `data/` on the server
- keep artifacts in `artifacts/` or object storage

If your cloud data path is large, mount it separately and pass the path into scripts.

## 8. Pulling trained results back

Do not push large artifacts into git.

Instead use one of:

- `scp`
- `rsync`
- object storage
- a mounted data disk

Example:

```bash
scp user@server:~/quant_trading_project/artifacts/m3net_stage1.joblib ./artifacts/
```

## 9. Suggested next upgrade

After this base workflow is stable, the next clean step is:

1. add `torch`-based sequence models on cloud
2. add remote experiment configs
3. add `tmux` or `systemd` job wrappers
4. add model registry and rolling retraining
