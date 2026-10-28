"""Chạy thử (smoke) LifeMate trên GPU của Kaggle (kernel riêng tư quanlam/lifemate-smoke).

Nội dung: kịch bản chạy trọn một lượt kiểm tra trên Kaggle, từ lấy dữ liệu, cài môi trường
đến chạy agent với vLLM thật và in báo cáo.

Các bước: tìm dataset dữ liệu riêng tư, clone mã nguồn từ GitHub, cài các thư viện đã khoá phiên bản
cùng vLLM, chạy các bài kiểm thử offline, đo chất lượng ASR trên val_seen, chọn cấu hình ASR (tùy chọn),
sau đó chạy 30 mẫu smoke với vLLM thật (transcript gold và/hoặc ASR whisper-large-v3). Kết quả nằm trong
/kaggle/working/runs, asr_eval và asr_tune để tải về bằng `kaggle kernels output`.

Cách hoạt động, theo thứ tự trong file:
  1. Tìm thư mục dữ liệu dưới /kaggle/input, clone mã nguồn ở git ref đã chọn.
  2. Sửa profile Kaggle (đường dẫn dữ liệu thật, nguồn HF token).
  3. Sinh config chạy agent trên chữ ASR từ configs/v0.toml.
  4. Cài thư viện, gỡ torchaudio, dựng lại danh sách split, chạy pytest.
  5. Đo chất lượng ASR (RUN_ASR_EVAL), chọn cấu hình ASR (ASR_TUNE), dựng cache ASR (RUN_ASR_CACHE).
  6. Chạy agent (gold và/hoặc ASR), đếm các sự kiện deadline trong log.
  7. In báo cáo, xoá mã nguồn và cache để output chỉ còn kết quả.

Đẩy lên:  kaggle kernels push -p ops/kaggle/smoke --accelerator NvidiaTeslaT4
Dữ liệu:  dataset riêng tư quanlam/zac2026-lifemate (tuyệt đối không công khai).
Tham số (biến môi trường, đọc lúc chạy): LIFEMATE_REF (git ref, mặc định main),
RUN_GOLD / RUN_ASR_EVAL ("1" để chạy, mặc định "1"), RUN_ASR (agent trên chữ ASR, mặc định "0": T4 15 GB
không chứa nổi vLLM cùng whisper-large-v3 ở dtype auto), RUN_ASR_CACHE (mặc định "0"),
ASR_TUNE (danh sách mô hình cho lifemate.eval.asr_tune, ví dụ "large-v3,phowhisper,parakeet"; trống = bỏ qua),
ASR_EVAL_IDS (mặc định val_seen), RUN_IDS (file danh sách id,
mặc định là 30 mẫu smoke), RUN_TAG (hậu tố của tên lượt chạy). ops/kaggle/eval đặt các biến này rồi
exec file này.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Địa chỉ repo, git ref và các thư mục làm việc cố định của Kaggle.
REPO = "https://github.com/awun0105/JSON-lifemate.git"
REF = os.environ.get("LIFEMATE_REF", "main")
WORK = Path("/kaggle/working")
CODE = WORK / "lifemate"
INPUT = Path("/kaggle/input")


def flag(name: str, default: str = "1") -> bool:
    """Đọc cờ bật/tắt từ biến môi trường; thiếu biến thì dùng `default` (mặc định là bật, "1")."""
    return os.environ.get(name, default) == "1"


def sh(cmd: str, check: bool = True) -> int:
    """Chạy một lệnh shell (bash, pipefail) trong thư mục mã nguồn (hoặc /kaggle/working trước khi clone).

    In lệnh và thời gian chạy ra log. Khi `check` bật và lệnh lỗi thì dừng cả kịch bản với cùng
    mã thoát; khi `check` tắt thì trả về mã thoát để nơi gọi tự xử lý.
    """
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    code = subprocess.call(
        f"set -o pipefail; {cmd}", shell=True, executable="/bin/bash", cwd=CODE if CODE.is_dir() else WORK
    )
    # In mã thoát và thời gian chạy để dễ đọc log Kaggle.
    print(f"[exit {code}, {time.time() - t0:.0f}s]", flush=True)
    if check and code != 0:
        sys.exit(code)
    return code


def edit(path: Path, pattern: str, repl: str) -> None:
    """Thay đúng một dòng `key = value` trong file TOML; dừng ngay với thông báo lỗi nếu không thấy khoá.

    Dừng khi không khớp giúp phát hiện sớm việc config gốc đổi định dạng, thay vì âm thầm chạy với
    giá trị cũ.
    """
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{path}: pattern {pattern!r} not found")
    path.write_text(new, encoding="utf-8")


def find_data_root() -> Path:
    """Trả về thư mục chứa train/ và public_test/.

    Cách Kaggle gắn dataset khác nhau giữa các phiên bản và các mục có thể là symlink, nên duyệt
    với followlinks và in phần đầu của cây thư mục ra log. Chỉ duyệt tối đa 5 tầng để không đi quá sâu.
    """
    for root, dirs, _files in os.walk(INPUT, followlinks=True):
        depth = len(Path(root).relative_to(INPUT).parts)
        if depth <= 3:
            print("  " * depth + root, flush=True)
        if "train" in dirs and "public_test" in dirs:
            return Path(root)
        # Không đi sâu hơn 5 tầng.
        if depth >= 5:
            dirs[:] = []
    sys.exit("no folder with train/ and public_test/ under /kaggle/input")


# 1. Dữ liệu và mã nguồn: tìm thư mục dữ liệu, clone nông (depth 1) đúng git ref.
data_root = find_data_root()
print("data_root:", data_root, flush=True)
sh(f"git clone --depth 1 --branch {REF} {REPO} {CODE}")
sh("git log --oneline -1")

# 2. Profile Kaggle: ghi đường dẫn gắn dữ liệu thật; cả hai mô hình đều công khai nên không cần secret HF.
profile = CODE / "configs/env/kaggle.toml"
edit(profile, r"^data_root = .*$", f'data_root = "{data_root}"')
edit(profile, r"^hf_token_source = .*$", 'hf_token_source = "env:HF_TOKEN"')

# 3. Các config chạy sinh ra từ v0: bản dùng ASR thay cho transcript gold.
v0 = CODE / "configs/v0.toml"
asr_cfg = CODE / "configs/kaggle_asr.toml"
shutil.copy(v0, asr_cfg)
edit(asr_cfg, r'^backend = "gold".*$', 'backend = "asr"')
# T4 có 15 GB dùng chung cho vLLM và ASR. Prompt dài nhất đo được khoảng 7.7K token, nên context 8K.
# Mức 0.72 từng vừa cho vLLM + PhoWhisper-medium; với whisper-large-v3 ở float16 thì hết bộ nhớ
# (eval 28/10: 431/452 mẫu lỗi), nên agent trên chữ ASR cần GPU >= 24 GB. A30 giữ nguyên giá trị của v0.
edit(asr_cfg, r"^max_model_len = .*$", "max_model_len = 8192")
edit(asr_cfg, r"^gpu_memory_utilization = .*$", "gpu_memory_utilization = 0.72")

# 4. Thư viện. vLLM kéo theo torch cho CUDA 13.0; torchaudio có sẵn trên Kaggle được build cho CUDA 12.8
# và làm hỏng lệnh import transformers, trong khi LifeMate tự đọc WAV, nên gỡ torchaudio.
sh("nvidia-smi")
sh("pip install -q -r ops/requirements/notebook.txt pytest")
sh("pip install -q vllm==0.32.0")
sh("pip install -q faster-whisper==1.2.1")  # engine ASR của configs/v0.toml (CTranslate2 whisper-large-v3)
# check=False: torchaudio có thể không có sẵn, khi đó lệnh gỡ báo lỗi nhưng không ảnh hưởng.
sh("pip uninstall -y -q torchaudio", check=False)
# In phiên bản torch/vLLM/transformers và tên GPU để kiểm tra môi trường.
sh(
    'python -c "import torch, vllm, transformers; '
    'print(torch.__version__, vllm.__version__, transformers.__version__, torch.cuda.get_device_name(0))"'
)
# In thông tin môi trường mà LifeMate tự nhận diện.
sh("python -c \"from pathlib import Path; from lifemate.core.envinfo import detect; print(detect(Path('.')))\"")
# Danh sách id của các split bị git bỏ qua; dựng lại từ dữ liệu đã gắn (tất định, giống hệt file ở máy local).
sh(f"python -m lifemate.eval.make_splits --root {data_root} --smoke 30 --asr-tune 300")
# Chạy kiểm thử offline (bỏ qua nhóm cần dữ liệu và tokenizer); lỗi kiểm thử không dừng kịch bản.
sh('python -m pytest -q -p no:cacheprovider -m "not data and not tokenizer"', check=False)

# 5. Chất lượng ASR của cấu hình hiện tại (configs/v0.toml) trên cùng một bộ mẫu.
if flag("RUN_ASR_EVAL"):
    eval_ids = os.environ.get("ASR_EVAL_IDS", "configs/splits/val_seen.txt")
    out = f"{WORK}/asr_eval/{Path(eval_ids).stem}"
    sh(
        f"python -m lifemate.eval.eval_asr --config {asr_cfg} --env kaggle --split train "
        f"--ids {eval_ids} --out {out}.json --hyps {out}.hyps.jsonl",
        check=False,
    )

# 5a. Chọn mô hình và cấu hình ASR công bằng (lifemate/eval/asr_tune.py): dò từng bước trên asr_tune.txt, cùng hàm
# mục tiêu cho mọi mô hình, rồi đo cấu hình thắng một lần trên val_seen và val_unseen.
# Mỗi mô hình một lệnh. ChunkFormer và Parakeet đo sau cùng, thư viện riêng của chúng cài ngay trước khi đo:
# chunkformer cần torchaudio (cài bản khớp torch hiện tại, vì bản có sẵn trên Kaggle làm hỏng transformers), NeMo
# kéo theo nhiều thư viện và có thể đổi phiên bản torch. Cài lỗi thì chỉ mô hình đó hỏng, các mô hình trước đã xong.
tune_models = [m.strip() for m in os.environ.get("ASR_TUNE", "").split(",") if m.strip()]
LATE = {"chunkformer": 1, "parakeet": 2}
for model in sorted(tune_models, key=lambda m: LATE.get(m, 0)):
    if model == "chunkformer":
        sh(
            "TV=$(python -c \"import torch; print(torch.__version__.split('+')[0])\"); "
            "CU=$(python -c \"import torch; print('cu' + torch.version.cuda.replace('.', ''))\"); "
            'pip install -q "torchaudio==$TV" --index-url "https://download.pytorch.org/whl/$CU"',
            check=False,
        )
        sh('pip install -q "chunkformer==1.2.2"', check=False)
    if model == "parakeet":
        sh('pip install -q "nemo_toolkit[asr]==2.6.2"', check=False)
        sh("pip install -q flashlight-text", check=False)  # cho lựa chọn lm (beam + KenLM); thiếu thì chỉ lm lỗi
    sh(f"python -m lifemate.eval.asr_tune --env kaggle --models {model} --out {WORK}/asr_tune/{model}", check=False)

# 5b. ASR cache cho dữ liệu fine-tune (SPEC 6.2.1), tắt mặc định: chạy ASR một lần trên train, val_seen và
# val_unseen rồi lưu chữ vào output của kernel (riêng tư). Tải về bằng `kaggle kernels output` và đặt vào
# data/asr/train_asr.json, hoặc gắn output này làm dataset riêng tư cho notebook train.
if flag("RUN_ASR_CACHE", "0"):
    sh(
        f"python -m lifemate.train.asr_cache --config {asr_cfg} --env kaggle --out {WORK}/asr_cache/train_asr.json "
        "--ids configs/splits/train.txt configs/splits/val_seen.txt configs/splits/val_unseen.txt",
        check=False,
    )

# 6. Chạy agent. Giữ log CLI của từng lượt để kiểm tra nhánh xử lý deadline: `llm_step_aborted_at_deadline`
# nghĩa là một bước bị cắt tại deadline_s; `vllm_abortable_generate_unsupported` nghĩa là vLLM thiếu API
# step/abort nên không cắt được bước (phải bằng 0 trước khi nộp bài).
# RUN_IDS có thể là nhiều file cách nhau bằng dấu phẩy (ví dụ val_seen và val_unseen trong cùng một phiên, đỡ một
# lượt cài đặt). Với nhiều file, tên lượt chạy là `<RUN_TAG>_<tên file>_<gold|asr>`.
id_files = [f.strip() for f in os.environ.get("RUN_IDS", "configs/splits/smoke.txt").split(",") if f.strip()]
tag = os.environ.get("RUN_TAG", "smoke")
runs = {}
cli = "python -m lifemate.run.cli --env kaggle"
logs = WORK / "logs"
logs.mkdir(exist_ok=True)
for ids in id_files:
    split_args = f"--split train --ids {ids}"
    prefix = tag if len(id_files) == 1 else f"{tag}_{Path(ids).stem}"
    for kind, enabled, cfg in (("gold", flag("RUN_GOLD"), "configs/v0.toml"), ("asr", flag("RUN_ASR", "0"), asr_cfg)):
        if enabled:
            name = f"{prefix}_{kind}"
            # Ghi log ra file qua tee; mã thoát của lượt chạy được lưu để in ở cuối.
            runs[name] = sh(f"{cli} --config {cfg} {split_args} --name {name} 2>&1 | tee {logs}/{name}.log", False)
            text = (logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")
            # Đếm số lần xuất hiện của từng sự kiện quan trọng trong log.
            for key in ("llm_step_aborted_at_deadline", "vllm_abortable_generate_unsupported", "sample_exception"):
                print(f"[{name}] {key}: {text.count(key)}", flush=True)

# 7. Báo cáo; output chỉ giữ lại runs/ và asr_eval/ (xoá mã nguồn và cache HF cho gọn).
# Chỉ dựng báo cáo khi có lần chạy agent (kernel chỉ đo ASR thì không có index.csv).
if (WORK / "runs" / "index.csv").is_file():
    sh(f"python -m lifemate.run.report runs {WORK}/runs/index.csv", check=False)
for path in (
    sorted(glob.glob(f"{WORK}/runs/*/summary.json"))
    + sorted(glob.glob(f"{WORK}/asr_eval/*.json"))
    + sorted(glob.glob(f"{WORK}/asr_tune/*/summary.json"))
):
    print(f"\n== {path}\n" + Path(path).read_text(encoding="utf-8"), flush=True)
shutil.rmtree(CODE, ignore_errors=True)
shutil.rmtree(WORK / "hf_cache", ignore_errors=True)
print("exit codes:", runs)
