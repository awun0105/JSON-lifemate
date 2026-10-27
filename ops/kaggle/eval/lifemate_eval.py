"""Kaggle eval run for the prompt-tuning loop (private kernel quanlam/lifemate-eval).

Runs the agent with PhoWhisper transcripts on one id list and keeps runs/ and logs/ as output.
It fetches ops/kaggle/smoke/lifemate_smoke.py from the same git ref and execs it with these knobs,
so each loop iteration only needs a commit on main and `kaggle kernels push -p ops/kaggle/eval`.
Edit RUN below to change the split or the run name.
"""

import os
import subprocess

RUN = {
    "LIFEMATE_REF": "main",
    "RUN_IDS": "configs/splits/val_seen.txt",
    "RUN_TAG": "val_seen",
    "RUN_GOLD": "0",
    "RUN_ASR": "1",
    "RUN_ASR_EVAL": "0",
}
os.environ.update(RUN)
src = "/kaggle/working/_launcher"
subprocess.check_call(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        RUN["LIFEMATE_REF"],
        "https://github.com/awun0105/JSON-lifemate.git",
        src,
    ]
)
script = f"{src}/ops/kaggle/smoke/lifemate_smoke.py"
with open(script, encoding="utf-8") as f:
    code = f.read()
subprocess.check_call(["rm", "-rf", src])
exec(compile(code, script, "exec"))
