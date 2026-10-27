"""Kaggle GPU smoke run for LifeMate (private kernel quanlam/lifemate-smoke).

Steps: find the private data dataset, clone the code from GitHub, install the locked deps plus vLLM,
run the offline tests, measure ASR quality at float32 vs the auto dtype on val_seen, then run the
30 smoke samples with real vLLM (gold transcripts and/or PhoWhisper). Outputs land in
/kaggle/working/runs and /kaggle/working/asr_eval for `kaggle kernels output`.

Push:  kaggle kernels push -p ops/kaggle/smoke --accelerator NvidiaTeslaT4
Data:  private dataset quanlam/zac2026-lifemate (must never be public).
Knobs (env vars, read at run time): LIFEMATE_REF (git ref, default main),
RUN_GOLD / RUN_ASR / RUN_ASR_EVAL ("1" to run, default all "1").
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/awun0105/JSON-lifemate.git"
REF = os.environ.get("LIFEMATE_REF", "main")
WORK = Path("/kaggle/working")
CODE = WORK / "lifemate"
INPUT = Path("/kaggle/input")


def flag(name: str) -> bool:
    return os.environ.get(name, "1") == "1"


def sh(cmd: str, check: bool = True) -> int:
    """Run a shell command in the code folder (or /kaggle/working before the clone), echo time and exit code."""
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    code = subprocess.call(cmd, shell=True, cwd=CODE if CODE.is_dir() else WORK)
    print(f"[exit {code}, {time.time() - t0:.0f}s]", flush=True)
    if check and code != 0:
        sys.exit(code)
    return code


def edit(path: Path, pattern: str, repl: str) -> None:
    """Replace one `key = value` line in a TOML file; fail loudly if the key is missing."""
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{path}: pattern {pattern!r} not found")
    path.write_text(new, encoding="utf-8")


def find_data_root() -> Path:
    """Folder holding train/ and public_test/. Mount layout differs between Kaggle versions and entries
    can be symlinks, so walk with followlinks and print the top of the tree for the log."""
    for root, dirs, _files in os.walk(INPUT, followlinks=True):
        depth = len(Path(root).relative_to(INPUT).parts)
        if depth <= 3:
            print("  " * depth + root, flush=True)
        if "train" in dirs and "public_test" in dirs:
            return Path(root)
        if depth >= 5:
            dirs[:] = []
    sys.exit("no folder with train/ and public_test/ under /kaggle/input")


# 1. Data and code.
data_root = find_data_root()
print("data_root:", data_root, flush=True)
sh(f"git clone --depth 1 --branch {REF} {REPO} {CODE}")
sh("git log --oneline -1")

# 2. Kaggle profile: real mount path; both models are public, so no HF secret is needed.
profile = CODE / "configs/env/kaggle.toml"
edit(profile, r"^data_root = .*$", f'data_root = "{data_root}"')
edit(profile, r"^hf_token_source = .*$", 'hf_token_source = "env:HF_TOKEN"')

# 3. Run configs derived from v0.
v0 = CODE / "configs/v0.toml"
asr_cfg = CODE / "configs/kaggle_asr.toml"
shutil.copy(v0, asr_cfg)
edit(asr_cfg, r'^backend = "gold".*$', 'backend = "asr"')
# T4 has 15 GB shared by vLLM and PhoWhisper. The longest smoke prompt is about 6.1K tokens, so an 8K
# context at 0.72 utilisation fits both. The A30 (24 GB) keeps the v0 values.
edit(asr_cfg, r"^max_model_len = .*$", "max_model_len = 8192")
edit(asr_cfg, r"^gpu_memory_utilization = .*$", "gpu_memory_utilization = 0.72")
asr_fp32 = CODE / "configs/kaggle_asr_fp32.toml"
shutil.copy(asr_cfg, asr_fp32)
edit(asr_fp32, r'^dtype = "auto"(\s+# auto: bfloat16.*)$', r'dtype = "float32"\1')

# 4. Dependencies. vLLM brings torch for CUDA 13.0; Kaggle's torchaudio is built for CUDA 12.8 and
# breaks the transformers import, and LifeMate reads WAV itself, so drop torchaudio.
sh("nvidia-smi")
sh("pip install -q -r ops/requirements/notebook.txt pytest")
sh("pip install -q vllm==0.32.0")
sh("pip uninstall -y -q torchaudio", check=False)
sh(
    'python -c "import torch, vllm, transformers; '
    'print(torch.__version__, vllm.__version__, transformers.__version__, torch.cuda.get_device_name(0))"'
)
sh("python -c \"from pathlib import Path; from lifemate.core.envinfo import detect; print(detect(Path('.')))\"")
sh('python -m pytest -q -p no:cacheprovider -m "not data and not tokenizer"', check=False)

# 5. ASR quality: float32 (old behaviour) vs auto (float16 on T4, bfloat16 on sm80+), same samples.
if flag("RUN_ASR_EVAL"):
    for name, cfg in (("fp32", asr_fp32), ("auto", asr_cfg)):
        sh(
            f"python -m lifemate.eval.eval_asr --config {cfg} --env kaggle --split train "
            f"--ids configs/splits/val_seen.txt --out {WORK}/asr_eval/val_seen_{name}.json",
            check=False,
        )

# 6. Agent runs on the 30 smoke samples.
smoke = "--split train --ids configs/splits/smoke.txt"
runs = {}
cli = "python -m lifemate.run.cli --env kaggle"
if flag("RUN_GOLD"):
    runs["gold"] = sh(f"{cli} --config configs/v0.toml {smoke} --name smoke_gold", check=False)
if flag("RUN_ASR"):
    runs["asr"] = sh(f"{cli} --config {asr_cfg} {smoke} --name smoke_asr", check=False)

# 7. Report; keep only runs/ and asr_eval/ in the output.
sh(f"python -m lifemate.run.report runs {WORK}/runs/index.csv", check=False)
for path in sorted(glob.glob(f"{WORK}/runs/*/summary.json")) + sorted(glob.glob(f"{WORK}/asr_eval/*.json")):
    print(f"\n== {path}\n" + Path(path).read_text(encoding="utf-8"), flush=True)
shutil.rmtree(CODE, ignore_errors=True)
shutil.rmtree(WORK / "hf_cache", ignore_errors=True)
print("exit codes:", runs)
