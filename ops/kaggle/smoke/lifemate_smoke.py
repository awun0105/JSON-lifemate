"""Chạy thử (smoke) LifeMate trên GPU của Kaggle (kernel riêng tư quanlam/lifemate-smoke).

Nội dung: kịch bản chạy trọn một lượt kiểm tra trên Kaggle, từ lấy dữ liệu, cài môi trường
đến chạy agent với vLLM thật và in báo cáo.

Các bước: tìm dataset dữ liệu riêng tư, clone mã nguồn từ GitHub, cài các thư viện đã khoá phiên bản
cùng vLLM, chạy các bài kiểm thử offline, đo chất lượng ASR ở float32 so với dtype auto trên val_seen,
sau đó chạy 30 mẫu smoke với vLLM thật (transcript gold và/hoặc PhoWhisper). Kết quả nằm trong
/kaggle/working/runs và /kaggle/working/asr_eval để tải về bằng `kaggle kernels output`.

Cách hoạt động, theo thứ tự trong file:
  1. Tìm thư mục dữ liệu dưới /kaggle/input, clone mã nguồn ở git ref đã chọn.
  2. Sửa profile Kaggle (đường dẫn dữ liệu thật, nguồn HF token).
  3. Sinh các config chạy từ configs/v0.toml (bản ASR và bản ASR float32).
  4. Cài thư viện, gỡ torchaudio, dựng lại danh sách split, chạy pytest.
  5. Đo chất lượng ASR (nếu RUN_ASR_EVAL=1).
  6. Chạy agent (gold và/hoặc ASR), đếm các sự kiện deadline trong log.
  7. In báo cáo, xoá mã nguồn và cache để output chỉ còn runs/ và asr_eval/.

Đẩy lên:  kaggle kernels push -p ops/kaggle/smoke --accelerator NvidiaTeslaT4
Dữ liệu:  dataset riêng tư quanlam/zac2026-lifemate (tuyệt đối không công khai).
Tham số (biến môi trường, đọc lúc chạy): LIFEMATE_REF (git ref, mặc định main),
RUN_GOLD / RUN_ASR / RUN_ASR_EVAL ("1" để chạy, mặc định đều là "1"), RUN_ASR_CACHE (mặc định "0"),
ASR_EVAL_DTYPES ("auto" hoặc "auto,fp32"), ASR_EVAL_IDS (mặc định val_seen), RUN_IDS (file danh sách id,
mặc định là 30 mẫu smoke), RUN_TAG (hậu tố của tên lượt chạy). ops/kaggle/eval đặt các biến này rồi
exec file này.
"""

import glob
import json
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
# T4 có 15 GB dùng chung cho vLLM và ASR. Prompt dài nhất đo được khoảng 7.7K token, nên context 8K
# với mức sử dụng bộ nhớ 0.72 vừa đủ cho vLLM + PhoWhisper-medium;
# whisper-large-v3 (CT2) chưa được đo ở đây. A30 (24 GB) giữ nguyên giá trị của v0.
edit(asr_cfg, r"^max_model_len = .*$", "max_model_len = 8192")
edit(asr_cfg, r"^gpu_memory_utilization = .*$", "gpu_memory_utilization = 0.72")
# ASR_DTYPE (tùy chọn) đổi dtype của ASR trong config chạy agent. Bản eval 28/10 với large-v3 float16 bị
# hết bộ nhớ GPU cạnh vLLM (253/274 mẫu lỗi); "int8_float16" của CTranslate2 cần khoảng một nửa bộ nhớ.
if os.environ.get("ASR_DTYPE"):
    edit(asr_cfg, r'^dtype = "auto"(\s+# auto: bfloat16.*)$', f'dtype = "{os.environ["ASR_DTYPE"]}"\\1')
# Bản float32 để so chất lượng ASR với dtype auto; regex giữ lại phần chú thích "# auto: ..." của dòng.
asr_fp32 = CODE / "configs/kaggle_asr_fp32.toml"
shutil.copy(asr_cfg, asr_fp32)
edit(asr_fp32, r'^dtype = "auto"(\s+# auto: bfloat16.*)$', r'dtype = "float32"\1')

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
sh(f"python -m lifemate.eval.make_splits --root {data_root} --smoke 30")
# Chạy kiểm thử offline (bỏ qua nhóm cần dữ liệu và tokenizer); lỗi kiểm thử không dừng kịch bản.
sh('python -m pytest -q -p no:cacheprovider -m "not data and not tokenizer"', check=False)

# 5. Chất lượng ASR trên cùng một bộ mẫu. ASR_EVAL_DTYPES chọn kiểu số: mặc định chỉ "auto" (float16 trên T4,
# bfloat16 trên sm80+); thêm "fp32" khi cần so lại với float32 (đã so với PhoWhisper ngày 27/10, chậm gấp ~3 lần).
if flag("RUN_ASR_EVAL"):
    eval_ids = os.environ.get("ASR_EVAL_IDS", "configs/splits/val_seen.txt")
    cfgs = {"fp32": asr_fp32, "auto": asr_cfg}
    for name in os.environ.get("ASR_EVAL_DTYPES", "auto").split(","):
        sh(
            f"python -m lifemate.eval.eval_asr --config {cfgs[name.strip()]} --env kaggle --split train "
            f"--ids {eval_ids} --out {WORK}/asr_eval/{Path(eval_ids).stem}_{name.strip()}.json",
            check=False,
        )

# 5a. So các biến thể cấu hình ASR trên cùng bộ mẫu (ASR_SWEEP="lead,norep,..."; trống = bỏ qua). Mỗi biến thể là
# một bản sao của cấu hình ASR với vài khóa [asr] được sửa; ghi WER/thực thể và từng câu nghe ra (--hyps).
# Lý do (28/10): large-v3 với tham số của lead cho WER 44% trên val_seen, có nhiều câu rỗng.
ASR_VARIANTS = {
    "lead": {},
    "norep": {"drop": ["repetition_penalty", "no_repeat_ngram_size"]},
    "norep_raw": {"drop": ["repetition_penalty", "no_repeat_ngram_size"], "set": {"sanitize": False}},
    "norep_noprompt": {"drop": ["repetition_penalty", "no_repeat_ngram_size", "initial_prompt"]},
    "norep_novad": {
        "drop": ["repetition_penalty", "no_repeat_ngram_size", "vad_parameters"],
        "gk": {"vad_filter": False},
    },
    "phowhisper": {
        "set": {
            "engine": "hf_seq2seq",
            "model": "vinai/PhoWhisper-medium",
            "revision": "55a7e3eb6c906de891f8f06a107754427dd3be79",
            "sanitize": False,
            "generate_kwargs": {"language": "vi", "task": "transcribe"},
        }
    },
}
sweep = [v.strip() for v in os.environ.get("ASR_SWEEP", "").split(",") if v.strip()]
if sweep:
    import tomllib

    sys.path.insert(0, str(CODE))
    from lifemate.train.llm_pipeline import dump_toml

    base = tomllib.loads(asr_cfg.read_text(encoding="utf-8"))
    eval_ids = os.environ.get("ASR_EVAL_IDS", "configs/splits/val_seen.txt")
    for name in sweep:
        spec = ASR_VARIANTS[name]
        raw = json.loads(json.dumps(base))  # bản sao sâu
        gk = dict(raw["asr"].get("generate_kwargs") or {})
        for k in spec.get("drop", []):
            gk.pop(k, None)
        gk.update(spec.get("gk", {}))
        raw["asr"]["generate_kwargs"] = gk
        raw["asr"].update(spec.get("set", {}))
        cfg_path = CODE / f"configs/kaggle_asr_{name}.toml"
        cfg_path.write_text(dump_toml(raw), encoding="utf-8")
        out = WORK / "asr_sweep" / name
        sh(
            f"python -m lifemate.eval.eval_asr --config {cfg_path} --env kaggle --split train --ids {eval_ids} "
            f"--out {out}.json --hyps {out}.hyps.jsonl",
            check=False,
        )

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
    for kind, enabled, cfg in (("gold", flag("RUN_GOLD"), "configs/v0.toml"), ("asr", flag("RUN_ASR"), asr_cfg)):
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
    + sorted(glob.glob(f"{WORK}/asr_sweep/*.json"))
):
    print(f"\n== {path}\n" + Path(path).read_text(encoding="utf-8"), flush=True)
shutil.rmtree(CODE, ignore_errors=True)
shutil.rmtree(WORK / "hf_cache", ignore_errors=True)
print("exit codes:", runs)
