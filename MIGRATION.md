# ASIM Migration Guide — Qwen3.5 vLLM GRPO Training

How to set up this Qwen3.5-9B GRPO HVAC training pipeline on a fresh machine.

## TL;DR

```bash
# 1. Clone repo
git clone https://github.com/ggsongze/asim.git
cd asim

# 2. Install both venvs (see Section 2)
python3.11 -m venv .venv          # legacy: EnergyPlus + controllables + gymnasium
python3.11 -m venv .venv_qwen35   # modern: vllm 0.19.1 + torch 2.10 + transformers 5.6
.venv/bin/pip install -r requirements_venv_legacy.txt
.venv_qwen35/bin/pip install -r requirements_qwen35.txt

# 3. Get the Stage1 LoRA checkpoint (~191 MB)
# Either copy from source machine or re-train Stage 1; we use:
#   result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414/

# 4. Launch
bash launch_v15_qwen35_9b_g4_vllm.sh
```

---

## 1. Hardware

| Component | Spec used | Notes |
|---|---|---|
| GPU | 2× NVIDIA RTX 6000 Ada (46 GB each) | TP=2, ~25 GB/GPU at peak |
| GPU mem total | 92 GB across 2 GPUs | minimum ~50 GB total for TP=2 |
| CUDA | 12.8 | torch 2.10 was built for cu128 |
| RAM | 64 GB+ | EnergyPlus + 4 rollouts in-flight |
| Disk | ~150 GB | model_cache (69 GB), result/ outputs grow |

**Single-GPU is NOT supported** — empirically OOMs (vLLM 9 GB model + KV cache + HF 18 GB model + ~10 GB activations all on one 46 GB card).

## 2. Two-venv hybrid setup

This project intentionally uses **two Python venvs** with different package sets:

| venv | Purpose | Key packages |
|---|---|---|
| `.venv` | Environment simulator (legacy) | `energyplus-core==0.1.0a0`, `controllables-core` (git pin), `gymnasium==0.28.1`, `pythermalcomfort`, `torch==2.7` |
| `.venv_qwen35` | LLM training (modern) | `vllm==0.19.1`, `torch==2.10.0+cu128`, `transformers==5.6.2`, `peft==0.18.1` |

The trainer launches with `.venv_qwen35/bin/python` and **appends `.venv/lib/.../site-packages` to `sys.path`** at import time (see [train_qwen3_houston_gspo_stage2_steplevel_vllm.py:31-34](train_qwen3_houston_gspo_stage2_steplevel_vllm.py#L31-L34)). So at runtime, modules from BOTH venvs are visible — modern stuff overrides legacy on import order, legacy provides `controllables` / `energyplus` / `gymnasium` which never made it into the modern venv.

**Don't try to merge into one venv** — `vllm 0.19.1` requires `torch 2.10`, but `controllables` / `energyplus-core` were built against `torch 2.7` and won't import cleanly with newer torch.

### Install legacy venv

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements_venv_legacy.txt
```

`controllables-core` is a private GitHub package pinned to a specific commit:
```
controllables-core @ git+https://github.com/NTU-CCA-HVAC-OPTIM-a842a748/EnergyPlus-OOEP@0978ccfc00d258d6564e703665e6f07601df4e32
```
Make sure you have GitHub access to that repo (private). If not, contact the author.

### Install modern venv

```bash
python3.11 -m venv .venv_qwen35
.venv_qwen35/bin/pip install --upgrade pip wheel
# Install torch first with explicit CUDA 12.8 wheel
.venv_qwen35/bin/pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# Then the rest
.venv_qwen35/bin/pip install -r requirements_qwen35.txt
```

Critical version pins (don't deviate):
- `vllm==0.19.1` — sleep_mode + LoRA hot-swap + Qwen3.5 ConditionalGeneration architecture
- `transformers==5.6.2` — Qwen3.5 chat template + XML tool format
- `peft==0.18.1` — LoRARequest int_id format

## 3. Files to transfer

| Path | Size | Required? |
|---|---|---|
| Code (`*.py`, `*.sh`) | ~10 MB | **YES** (clone from GitHub) |
| `result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414/` | 191 MB | **YES** (LoRA to resume) |
| `miami_stage2_10min.idf` | < 1 MB | **YES** (building model) |
| `result/gspo/miami_gspo_dataset_stage2_10min.jsonl` | ~1 MB | **YES** (training prompts) |
| `miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv` | ~1 MB | **YES** (forecast data) |
| `.tmp_todo_random_start_cell0.py` | 26 KB | **YES** (env factory module) |
| Weather `.epw` (in IDF dir) | ~1 MB | **YES** |
| `.model_cache/` (HF cache) | 69 GB | optional — saves Qwen3.5-9B re-download time (~30 min) |
| Stage1 base model `Qwen/Qwen3-8B` | 18 GB | optional — HF will auto-download if missing |
| Qwen3.5-9B base model | 18 GB | optional — HF will auto-download if missing |

### Minimal tarball

```bash
tar czf asim_minimal.tar.gz \
    --exclude='.venv*' \
    --exclude='.model_cache' \
    --exclude='result/gspo/qwen35_*' \
    --exclude='result/gspo/v15_*' \
    --exclude='result/comparisons' \
    --exclude='*.log' \
    --exclude='.git' \
    /home/AD/user/lab/asim/
# ~210 MB (code + Stage1 ckpt + IDF + dataset + forecast)
```

## 4. EnergyPlus binary

The `energyplus-core==0.1.0a0` Python package ships with bundled EnergyPlus 23.2.0-7636e6b3e9 binary at:
```
.venv/lib/python3.11/site-packages/energyplus/core/lib/libenergyplusapi.so
```

No separate EnergyPlus install needed.

## 5. HuggingFace setup

Set `HF_HOME` to control where models are cached:
```bash
export HF_HOME=/path/to/.model_cache
```

The launch script uses `/home/AD/user/lab/asim/.model_cache` — adjust to your filesystem.

The base model `Qwen/Qwen3.5-9B` will auto-download (~18 GB) on first launch if not in cache.

## 6. Launch and verify

```bash
# Sanity check: imports + env module load
.venv_qwen35/bin/python -c "
import sys
sys.path.insert(0, '/path/to/asim/.venv/lib/python3.11/site-packages')
import vllm; import torch; import transformers; import peft
import controllables; import energyplus; import gymnasium
print('venv OK')
"

# vLLM TP=2 smoke test (kill after [VLLM] loaded confirmed)
bash launch_v15_qwen35_9b_g4_vllm.sh
tail -F result/gspo/qwen35_9b_v15_vllm_*/vllm.log  # watch
# After "[VLLM]   slept; HF retains GPU." appears, training is healthy
```

## 7. Architecture quick reference

| Aspect | Setting | Why |
|---|---|---|
| Single rank (no DDP) | TP=2 already uses both GPUs | DDP would conflict |
| HF model | `--device cuda:0` | rank-0 placement, conventional |
| vLLM TP | `--vllm-tp 2` | spreads 9B model across both GPUs |
| vLLM mem util | `--vllm-gpu-mem-util 0.45` | leaves ~25 GB on GPU0 for HF |
| Sleep mode | enabled | swaps vLLM out during HF backward |
| LoRA hot-swap | `save_pretrained` → `LoRARequest` | vLLM caches by adapter id |
| `enforce_eager` | `False` (CUDA graphs ON) | False is 11.6× faster than True |
| Process manager | `nohup` (not tmux) | tmux server died once unexplained |

## 8. Common pitfalls

- **"No module named 'controllables'"** — `.venv_qwen35/bin/python` ran without sys.path append. Make sure you're invoking through the trainer (`train_qwen3_houston_gspo_stage2_steplevel_vllm.py`) which auto-appends `.venv` to sys.path; or manually add the path before importing.
- **vLLM `RuntimeError("cancelled")` after ~1h50m** — vLLM 0.19.1 + TP=2 + sleep/wake cycles can crash on `shm_broadcast.acquire_read`. If it happens, kill + restart from latest checkpoint. Putting HF on cuda:0 instead of cuda:1 helped in our testing.
- **GPU memory not released after kill** — `nvidia-smi` shows lingering vLLM workers. Run `pkill -KILL -f VLLM` and `pkill -KILL -f EngineCore`.
- **`enforce_eager=True` is 11.6× slower** — keep it `False` (the default in this trainer).
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is INCOMPATIBLE with vLLM `CuMemAllocator` (sleep mode). Don't set it.
- **CVD=0 alone (single GPU TP=1)** — empirically OOMs at vLLM init when HF model is also on the same GPU.

## 9. File layout summary

```
asim/
├── train_qwen3_houston_gspo_stage2_steplevel_vllm.py   # main trainer
├── llm_setpoint_planner_vllm.py                         # vLLM backend with tool-call loop
├── llm_setpoint_planner_qwen35.py                       # Qwen3.5 XML tool parser (parent class)
├── llm_setpoint_planner_unified.py                      # planner core (shared)
├── llm_setpoint_planner.py                              # base abstractions
├── grpo_miami_bandit.py                                 # RL bandit env
├── launch_v15_qwen35_9b_g4_vllm.sh                      # launch script (TP=2 + nohup)
├── miami_stage2_10min.idf                               # building model
├── .tmp_todo_random_start_cell0.py                      # env module (loaded dynamically)
├── result/gspo/
│   ├── stage1_checkpoints/.../checkpoint-16/            # Stage1 LoRA (resume target)
│   └── miami_gspo_dataset_stage2_10min.jsonl            # training prompts
├── miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv  # forecast
├── requirements_qwen35.txt                              # modern venv freeze
├── requirements_venv_legacy.txt                         # legacy venv freeze
└── README.md                                            # main project doc
```
