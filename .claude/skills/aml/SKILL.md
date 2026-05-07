---
name: aml
description: Manage Azure ML Singularity jobs. Use when submitting GPU jobs, checking status, downloading results, or cancelling runs.
---

# AML Job Manager

Manage Azure ML Singularity jobs for the ToolSMDP project using the `aml-helper` CLI.

## Commands

### Submit a job
```bash
aml submit --name <descriptive-name> --command "export PYTHONPATH=\$PWD:\$PYTHONPATH && <your command>"
```

Example — tier classification run:
```bash
aml submit --name tier-v4-all-5000 --command "export PYTHONPATH=\$PWD:\$PYTHONPATH && python -m analysis.tier_classification --input-dir data_local/processed --output-dir ./outputs --max-samples 5000"
```

Example — eval run:
```bash
aml submit --name eval-v4-500 --command "export PYTHONPATH=\$PWD:\$PYTHONPATH && python -m analysis.pre_training_characterization --input-dir data_local/eval_splits --output-dir ./outputs --num-rollouts 4 --datasets gsm8k hotpotqa"
```

### Monitor
```bash
aml status <job-name>     # Check status (Queued/Running/Completed/Failed)
aml logs <job-name>       # Stream logs
aml list                  # List recent jobs
```

### Download results
```bash
aml download <job-name> --output downloads/<descriptive-name> --path ./outputs
```

### Cancel / archive
```bash
aml cancel <job-name>
aml archive <job-name>
```

### Upload data
```bash
aml upload-data --name <dataset-name> --path <local-path>
```

### Create/update environment
```bash
aml env-create --name <env-name> --dockerfile ./Dockerfile
```

## Naming Convention

- Job names: `<purpose>-<version>-<scope>` (e.g., `tier-v4-all-5000`, `eval-v4-500`)
- Experiment names: generic (e.g., `infer-milestone-2`), NOT project-specific
- Results stored in: `downloads/<descriptive-name>/`

## Post-Job Workflow

After a job completes:
1. `aml download <job> --output downloads/<name> --path ./outputs`
2. Update `results.md` with new numbers and job name
3. Update `plan.md` step status (DONE/BLOCKED/etc.)

## Singularity Requirements (hard-won lessons)

These are **mandatory**. Missing any one causes cryptic failures:

1. **Premium SLA tier** — without it: "zero total quota" error
2. **UAI managed identity** — Singularity cannot use account key auth
3. **Common runtime** — `AZUREML_COMPUTE_USE_COMMON_RUNTIME=true`
4. **No pip install** — container venv is read-only. Use `PYTHONPATH=$PWD`
5. **No git metadata** — `aml submit` strips `.git/` automatically
6. **Generic experiment names** — not project-specific

All handled automatically by `aml-helper` via `aml.yaml` config.

## Setup

```bash
pip install -e ../aml-helper
# Config: aml.yaml in project root (gitignored)
# Template: ../aml-helper/aml.example.yaml
```

## Infrastructure

- **Workspace:** `bingdmml` (westus2)
- **Compute:** Singularity MI300X AMD GPUs (`Singularity.ND24is_MI300X_v5`)
- **Environment:** `amd-inference:8` (custom Docker, Wikipedia BM25 index baked in)
- **Identity:** `UAI_BingDMML` (client ID `37ec208a-d44d-4a10-bb4b-f567373cda57`)
- **Milestones 2-3**: works on AMD (inference only)
- **Milestone 5 (PPO)**: ROCm + OpenRLHF risky — may need NVIDIA GPUs
