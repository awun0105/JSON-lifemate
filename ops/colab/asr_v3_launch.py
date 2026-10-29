"""Chạy nền một lệnh của benchmark ASR lần 3 trên máy Colab (`colab exec -s <phiên> -f ops/colab/asr_v3_launch.py`).

Sửa `JOB` và `CMD` trước mỗi lần chạy. Lệnh chạy bằng `nohup` trong `/content/lifemate`, ghi log ra
`/content/<JOB>.log` và kết quả ra `/content/out/<JOB>/`; `colab exec` trả về ngay, theo dõi bằng
`tail /content/<JOB>.log`.
"""

import subprocess

JOB = "k1"
CMD = (
    "python -m lifemate.eval.asr_v3 transcribe --env colabcli --ids configs/splits/asr_tune.txt "
    "--variants E0 E1 E2 --rnnt --rerun 50 --resume --out /content/out/k1"
)
# Thư viện CUDA 12 cho CTranslate2 (cài ở `asr_v3_setup.py`).
LIBS = (
    "$(python -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; "
    "print(list(a.__path__)[0] + \":\" + list(b.__path__)[0])')"
)
subprocess.Popen(
    f"cd /content/lifemate && export LD_LIBRARY_PATH={LIBS}:$LD_LIBRARY_PATH && "
    f"nohup {CMD} > /content/{JOB}.log 2>&1 &",
    shell=True,
    executable="/bin/bash",
)
print(f"started {JOB}: {CMD}", flush=True)
