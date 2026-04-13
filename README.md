# ToolSMDP

Segment-level RL for LLM tool use via the Semi-Markov Decision Process framework. Decomposes LLM trajectories at tool-call boundaries for per-segment credit assignment using PPO.

## Project Structure

```
core/
  code_block_detector.py   # Detect ```python blocks: post-hoc (detect_code_block) + real-time (CodeBlockWatcher)
  context_block_detector.py # Detect <context>...</context> blocks for assimilation
  replacement.py           # Replace code blocks with comments + stdout (code vanishes, output remains)
  reward.py                # Answer extraction (GSM8K ####, QA patterns, FinQA numeric) + exact match
  segment_rollout.py       # Segment and Trajectory dataclasses (training-loop bookkeeping)

sandbox/
  executor.py              # Execute Python in sandboxed subprocess (5s timeout, Jupyter-style auto-display)

data/
  download_and_format.py   # Download datasets from HuggingFace -> unified JSONL

retrieval/
  search.py                # Pyserini BM25 over Wikipedia (21M passages)

analysis/
  pre_training_characterization.py  # Base model eval with tool rollouts
  tier_classification.py            # Tier 1/2 splits + rollout generation for critic warmup

scripts/
  test_base_model.py       # Inference script: code_gen | pipeline | eval modes (requires GPU)
  interactive.py            # Interactive REPL for testing

tests/                     # 158 tests covering all core components
```

## Quick Start

### Run Tests (bare metal)
```bash
conda activate toolsmdp
pytest tests/ -v
pytest tests/test_executor.py::TestAutoDisplay -v  # specific test class
```

### Run Tests (Docker)
```bash
docker compose run test              # all tests, 30s timeout
docker compose run dev               # interactive shell inside container
```

### Download Data
```bash
python -m data.download_and_format --datasets gsm8k hotpotqa finqa --max-samples 50
```

### Run Inference (GPU required)
```bash
python -m scripts.test_base_model --mode code_gen       # does model generate code blocks?
python -m scripts.test_base_model --mode pipeline        # full SMDP segment loop
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl
```

### Debug in VS Code
- **Dev Container**: Ctrl+Shift+P -> "Reopen in Container" (requires Docker Desktop)
- **Debug Tests**: F5 -> select "Debug Tests" or "Debug Current Test File" from launch configs

## Key Config

| Env var | Default | Purpose |
|---------|---------|---------|
| `DATA_ROOT` | `./data_local` | Dataset storage |
| `CHECKPOINT_ROOT` | `./checkpoints_local` | Model checkpoints |
| `SANDBOX_USER` | `sandbox` | Unprivileged user for code execution |

---

## Local Development Setup

### One-time setup

```bash
# Create conda environment
conda create -n toolsmdp python=3.11 -y
conda activate toolsmdp

# Install everything (torch ~2.5 GB)
pip install -e ".[train,dev]"

# Verify
pytest tests/ -v
python scripts/try_search.py
```

Java is auto-detected from PATH (installed via winget). If not found:
```bash
# Windows CMD
set JAVA_HOME=C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot

# Git Bash / WSL
export JAVA_HOME="/c/Program Files/Microsoft/jdk-21.0.10.7-hotspot"
```

### VS Code
`Ctrl+Shift+P` -> "Python: Select Interpreter" -> choose `toolsmdp`

### Key dependencies

| Package | Why |
|---|---|
| `torch` | Model inference/training |
| `transformers` | Qwen2.5-3B-Instruct loading |
| `vllm` | Fast inference with batching |
| `pyserini`, `faiss-cpu` | Wikipedia search (21M passages) |
| `datasets` | HuggingFace dataset download |
| `pytest` | Tests |

First search run downloads the 9.2 GB Wikipedia index (cached at `~/.cache/pyserini/`).

---

## Lightning.ai Setup (GPU inference)

Lightning Studios are isolated Linux VMs with dedicated GPUs. No Docker needed.

```bash
# Connect
ssh s_01kme7kcjmrxjkpchmqf8kd09f@ssh.lightning.ai

# Setup
git clone https://github.com/research-alpha-wt/toolsmdp.git && cd toolsmdp
pip install -e ".[train,dev]"
sudo apt update && sudo apt install -y openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
pytest tests/ -v

# Run inference
python -m scripts.test_base_model --mode eval --data data_local/eval_splits/gsm8k_test_50.jsonl
```

Disk: Model ~8 GB + Pyserini index ~9.2 GB (cached after first run).

---

## Azure ML Setup

**Workspace:** `bingdmml` (westus2). Singularity virtual cluster with MI300X AMD GPUs.

### aml-helper CLI

All job management uses the `aml-helper` package (standalone, at `../aml-helper/`):

```bash
pip install -e ../aml-helper
# Config: copy ../aml-helper/aml.example.yaml to project root as aml.yaml (gitignored)
```

Commands: `aml submit`, `aml status`, `aml logs`, `aml download`, `aml cancel`, `aml list`.
See `/aml` skill or `../aml-helper/README.md` for full reference.

### Infrastructure

| Component | Value |
|-----------|-------|
| Compute | `Singularity.ND24is_MI300X_v5` (AMD MI300X) |
| Environment | `amd-inference:8` (custom Docker, Wikipedia BM25 index baked in) |
| Identity | `UAI_BingDMML` (managed identity for blob storage access) |
| Storage | `bingdmml6443962715` blob storage |

### Singularity Gotchas

1. **Premium SLA tier** required (otherwise "zero total quota")
2. **UAI managed identity** (cannot use account key auth)
3. **Read-only venv** — use `PYTHONPATH=$PWD` instead of `pip install`
4. **No git metadata** — `aml submit` strips `.git/` automatically
5. **Generic experiment names** (e.g., `infer-milestone-2`)

All handled automatically by `aml-helper` config.
