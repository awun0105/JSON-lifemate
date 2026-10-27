# Docker image (SPEC 6.3, D1/D2)

The organisers mount the test split at `/data` (same layout as `public_test`), run the image with
no network, and collect `result/submission.jsonl`. Files: `Dockerfile`, `ops/docker/Dockerfile.dockerignore`,
`ops/docker/entrypoint.sh`, `ops/docker/smoke.sh`, Makefile targets `docker-build` / `docker-run`.

## Base image

`nvidia/cuda:12.4.1-runtime-ubuntu22.04` for both stages, with CPython 3.11 installed by uv
(`uv python install 3.11` into `/opt/python`, copied to the runtime stage).

Why not `vllm/vllm-openai`: it ships its own vLLM/torch in a system Python, so the versions would
not come from `uv.lock` and `uv sync --frozen` would install a second copy next to it. With the
plain CUDA runtime base, every Python package (vLLM, torch and its bundled CUDA libraries) comes from
`uv.lock`, the same as `make install-gpu`. The image only needs the NVIDIA driver from the host
(`--gpus all`, nvidia-container-toolkit). The base tag can be changed with `--build-arg CUDA_IMAGE=...`.

## Build

Weights are baked in: put them in a folder of the build context (default `artifacts/models/`) and build:

```bash
# artifacts/models/llm  <- the LLM snapshot (config.json, tokenizer, safetensors)
# artifacts/models/asr  <- the ASR snapshot, only when [asr].backend = "asr"
docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR=artifacts/models -t lifemate:dev .     # or: make docker-build
```

`MODEL_DIR` is copied to `/app/models`. `configs/docker.toml` (the image default) points at those paths; keep its other keys in step with `configs/v0.toml`:

```toml
[llm]
model = "/app/models/llm"
[asr]
model = "/app/models/asr"   # only when backend = "asr"
```

The build fails if `[llm].model` is empty, or if it is an absolute path that is not a folder in the
image. Use another config with `--build-arg LIFEMATE_CONFIG=configs/<x>.toml` (also the run-time default).
The image sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`,
`PYTHONUNBUFFERED=1`, runs as uid 10001, `WORKDIR /app`. `.git` is not copied, so `config.json`
records the code version from a `VERSION` file if present, else `unknown`.

## Run (organisers' flow)

```bash
docker run --rm --gpus all --network none \
  -v <test split dir>:/data:ro \
  -v $PWD/result:/app/result \
  lifemate:dev
# or: make docker-run DATA_SPLIT_DIR=<test split dir>
```

The entrypoint finds the single `/data/*_tools_catalog.json` (fails with exit 2 on 0 or more than 1),
links `/tmp/zaic_root/<split> -> /data` (ToolEnv wants `<root>/<split>/...`; `zaic.py` compares resolved
paths so the link is fine), then execs

```
python -m lifemate.run.cli --config ${LIFEMATE_CONFIG:-configs/docker.toml} --env docker --split <split> --out result --resume [extra args]
```

`--resume` is always passed: `RunStore` creates an empty `result/` and finds no done ids, and after an
interruption a rerun skips samples already in `trace.jsonl`. Extra `docker run` arguments go to the CLI
(e.g. `--limit 5`).

## Collecting results

`result/` (= `/app/result`) holds `submission.jsonl` (what the organisers take), plus `trace.jsonl`,
`summary.json`, `config.json`, `check.txt`. The host folder must be writable by uid 10001
(`chmod a+rwx result`, done by `make docker-run` and the smoke script). `/data` is mounted read-only;
nothing is written there.

## Checks on a rented GPU box (not runnable on the dev machine)

```bash
DATA_SPLIT_DIR=data/train+public_test/public_test ops/docker/smoke.sh
```

builds `lifemate:dev`, runs it with `--gpus all --network none`, validates with
`python -m zaic_tools --split <split> --root <parent> --validate result/submission.jsonl` and prints the line
count (public_test: 634 expected). `--network none` is the offline test: any download attempt fails the run.

Determinism (with `batch_invariant = true` on sm80+):

```bash
OUT_DIR=$PWD/result_a ops/docker/smoke.sh
OUT_DIR=$PWD/result_b ops/docker/smoke.sh
cmp result_a/submission.jsonl result_b/submission.jsonl && echo identical
```

Also check `summary.json` for no sample over 30 s.

## Not verified yet

Nothing here has been built or run: the dev machine has no Docker and no GPU. Open points:

- The image build itself (uv-managed Python on the CUDA base, vLLM wheel vs CUDA 12.4 runtime vs host driver).
- vLLM loading from `/app/models/llm` fully offline, and whether it tries to write caches outside `$HOME`
  (`/home/appuser` exists; the root filesystem is otherwise writable only where the image allows).
- 634 lines, `--validate` clean, two identical runs, all samples under 30 s.
- `/app/models/llm` and `/app/models/asr` must be real snapshots (`huggingface-cli download <id> --revision <sha> --local-dir artifacts/models/llm`); the build only checks that the folders exist.
- `result/` and `artifacts/models/` are not in `.gitignore` yet.
