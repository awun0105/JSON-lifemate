# Running on a RunPod pod

Profile `configs/env/runpod.toml`: data at `/workspace/zac2026/train+public_test`, runs at
`/workspace/zac2026/runs`, HF cache at `/workspace/hf_cache`, token from `HF_TOKEN`. `/workspace` is the
persistent volume; the rest of the pod disk is lost on stop.

## Pod

- GPU: sm80+ to match the organisers' A30 (A30, A100, A6000, L40S). 24 GB is enough for the 4B LLM in bf16
  plus whisper-large-v3 (CTranslate2, `Systran/faster-whisper-large-v3`); training wants 24 GB+.
- Image: any CUDA 12.x image with Python, e.g. `runpod/pytorch`. The script installs `uv` and the locked deps.
- Set `HF_TOKEN` in the pod's environment (RunPod secrets) only when a model is gated. Never write it to files.
- Docker build/smoke (`docs/docker.md`) needs a pod that can run Docker (most RunPod templates cannot). Build
  the image elsewhere if so; everything else runs directly on the pod.

## One command

```bash
curl -fsSL https://raw.githubusercontent.com/<user>/<repo>/main/ops/scripts/runpod_setup.sh \
  | bash -s -- --repo https://github.com/<user>/<repo>.git --ref main
```

It installs tools, clones or updates the code into `/workspace/lifemate`, syncs the locked dependencies
including vLLM, downloads the two model snapshots pinned in `configs/v0.toml` into `artifacts/models/llm` and
`artifacts/models/asr`, checks the data folder, runs the offline tests, then the 30-sample smoke and prints the runs
table. Options: `--no-clone` (code already there), `--no-smoke`.

## Upload the data

The competition data must not be public. From your machine:

```bash
runpodctl send train+public_test.zip      # prints a code; on the pod: runpodctl receive <code>
# or: rclone copy gdrive:zac2026/train+public_test /workspace/zac2026/train+public_test
```

Then `unzip` so that `/workspace/zac2026/train+public_test/train/...` exists.

## After setup

```bash
cd /workspace/lifemate
make splits                                            # split id lists are git-ignored; setup already ran this
make run ENV=runpod IDS=configs/splits/val_seen.txt            # agent, gold transcripts
make eval ENV=runpod TAG=base                                     # baseline: real ASR on val_seen + val_unseen
make train-deps                                        # ms-swift 4.6.0, trl, peft for fine-tuning
uv run python -m lifemate.train.llm_pipeline --config configs/train/llm_v1.toml --env runpod
uv run python -m lifemate.run.report train /workspace/zac2026/runs/train/<name> --png loss.png
```

Runs and checkpoints stay on `/workspace`; copy merged models you want to keep to HF private or Kaggle
Models (`artifacts/models/registry.csv` records where).
