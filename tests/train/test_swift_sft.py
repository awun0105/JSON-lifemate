"""Kiểm tra lớp nối với ms-swift (lifemate.train.swift_sft): tham số, xuất dòng, kiểm template, checkpoint.

Các test đọc cấu hình thật `configs/train/llm_v1.toml`, dựng checkpoint giả bằng `_checkpoint` trong `tmp_path` và
thay lệnh `swift` bằng runner giả. Không cần GPU, tokenizer hay dữ liệu cuộc thi.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from lifemate.train import swift_sft
from lifemate.train.sft import load_llm_train_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CFG = REPO_ROOT / "configs" / "train" / "llm_v1.toml"


def _flag(argv: Sequence[str], name: str) -> str:
    """Trả về giá trị đứng ngay sau cờ `name` trong argv."""
    return argv[list(argv).index(name) + 1]


def test_argv_follows_the_config() -> None:
    """Tham số dòng lệnh ms-swift lấy đúng từ cấu hình; chỉ thêm --resume_from_checkpoint khi có resume_from."""
    cfg = load_llm_train_config(CFG)
    argv = swift_sft.swift_argv(
        cfg, train_file=Path("t.jsonl"), val_file=Path("v.jsonl"), out_dir=Path("o"), dtype="float16"
    )
    assert _flag(argv, "--model") == cfg.base.model and _flag(argv, "--model_revision") == cfg.base.revision
    assert _flag(argv, "--lora_rank") == str(cfg.lora.r) and _flag(argv, "--lora_alpha") == str(cfg.lora.alpha)
    i = argv.index("--target_modules")
    assert argv[i + 1 : i + 1 + len(cfg.lora.target_modules)] == list(cfg.lora.target_modules)
    assert _flag(argv, "--val_dataset") == "v.jsonl"  # chỉ val_unseen: ms-swift gộp các file val thành một loss
    assert _flag(argv, "--max_length") == str(cfg.train.max_len) and _flag(argv, "--truncation_strategy") == "delete"
    assert _flag(argv, "--gradient_accumulation_steps") == str(cfg.train.grad_accum)
    assert _flag(argv, "--lr_scheduler_type") == cfg.train.lr_scheduler
    assert _flag(argv, "--save_steps") == str(cfg.train.save_every_steps)
    assert _flag(argv, "--add_version") == "false"
    assert _flag(argv, "--quant_bits") == "4" and _flag(argv, "--bnb_4bit_compute_dtype") == "float16"
    assert "--resume_from_checkpoint" not in argv
    resumed = swift_sft.swift_argv(
        cfg, train_file=Path("t"), val_file=Path("v"), out_dir=Path("o"), dtype="bfloat16", resume_from=Path("c")
    )
    assert _flag(resumed, "--resume_from_checkpoint") == "c"


def test_export_keeps_messages_and_loss_flags_only(tmp_path: Path) -> None:
    """File xuất cho ms-swift chỉ giữ `messages` (kèm cờ loss), bỏ các trường phụ như id, category."""
    src = tmp_path / "rows.jsonl"
    row = {
        "id": "x",
        "category": "execute",
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "bad", "loss": False},
            {"role": "assistant", "content": "ok"},
        ],
    }
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert swift_sft.export_rows(src, tmp_path / "out" / "t.jsonl") == 1
    out = json.loads((tmp_path / "out" / "t.jsonl").read_text(encoding="utf-8"))
    assert out == {"messages": row["messages"]}


def test_pick_check_rows_puts_error_recovery_first() -> None:
    """Khi chọn dòng để kiểm template, dòng hồi phục lỗi (có loss=False) được ưu tiên đứng đầu."""
    plain = [{"id": str(i), "messages": [{"role": "assistant", "content": "a"}]} for i in range(5)]
    special = {"id": "s", "messages": [{"role": "assistant", "content": "a", "loss": False}]}
    picked = swift_sft.pick_check_rows([*plain, special], 3)
    assert [r["id"] for r in picked] == ["s", "0", "1"]


def test_compare_encodings() -> None:
    """So sánh mã hóa của hai bên báo lỗi rõ vị trí token đầu tiên lệch ở input_ids hoặc labels."""
    swift_sft.compare_encodings("r", ([1, 2], [-100, 2]), ([1, 2], [-100, 2]), lambda ids: "")
    with pytest.raises(swift_sft.TemplateMismatchError, match="input_ids differ at token 1"):
        swift_sft.compare_encodings("r", ([1, 2], [-100, 2]), ([1, 3], [-100, 3]), lambda ids: "")
    with pytest.raises(swift_sft.TemplateMismatchError, match="labels differ at token 0"):
        swift_sft.compare_encodings("r", ([1, 2], [-100, 2]), ([1, 2], [1, 2]), lambda ids: "")


def _checkpoint(root: Path, step: int, history: list[dict]) -> None:
    """Tạo thư mục checkpoint-<step> giả với trainer_state.json chứa log_history cho trước."""
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"log_history": history}), encoding="utf-8")


def test_checkpoints_best_and_latest(tmp_path: Path) -> None:
    """Liệt kê checkpoint bỏ qua bản đang ghi dở; best_k theo eval_loss thấp nhất, latest theo step lớn nhất."""
    hist = [{"step": 50, "eval_loss": 0.9}, {"step": 50, "loss": 1.2}, {"step": 100, "eval_loss": 0.7}]
    _checkpoint(tmp_path, 50, hist[:2])
    _checkpoint(tmp_path, 100, hist)
    _checkpoint(tmp_path, 150, [*hist, {"step": 150, "eval_loss": 0.8}])
    (tmp_path / "checkpoint-200").mkdir()  # đang ghi dở: chưa có trainer_state.json
    assert [(c.step, c.eval_loss) for c in swift_sft.checkpoints(tmp_path)] == [(50, 0.9), (100, 0.7), (150, 0.8)]
    assert [c.step for c in swift_sft.best_k(tmp_path, 2)] == [100, 150]
    latest = swift_sft.latest(tmp_path)
    assert latest is not None and latest.step == 150


def test_train_exports_checks_and_runs_swift(tmp_path: Path) -> None:
    """train xuất dữ liệu, kiểm template, gọi `swift sft` tiếp từ checkpoint mới nhất, báo lỗi khi mã thoát khác 0."""
    cfg = load_llm_train_config(CFG)
    rows = {"id": "x", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    for rel in (cfg.data.train, cfg.data.eval_unseen):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    calls: list[list[str]] = []
    checked: list[int] = []

    def runner(cmd: Sequence[str], cwd: Path) -> int:
        """Runner giả: ghi lại lệnh và trả mã thoát 0."""
        calls.append(list(cmd))
        return 0

    out = tmp_path / "work"
    # Có sẵn checkpoint-50 nên lần chạy phải tiếp tục từ đó.
    _checkpoint(out / "swift", 50, [{"step": 50, "eval_loss": 1.0}])
    swift_dir = swift_sft.train(
        cfg,
        repo_root=tmp_path,
        out_dir=out,
        runner=runner,
        dtype="float16",
        check=lambda c, r: checked.append(len(r)) or len(r),
    )
    assert swift_dir == out / "swift" and checked == [1]
    assert (out / "metrics.jsonl").is_file()  # report-train và register đọc file này
    assert calls[0][:2] == ["swift", "sft"]
    assert _flag(calls[0], "--resume_from_checkpoint") == str(out / "swift" / "checkpoint-50")
    assert (out / "swift_data" / "train.jsonl").is_file() and (out / "swift_data" / "val_unseen.jsonl").is_file()
    with pytest.raises(RuntimeError, match="exit code 3"):
        swift_sft.train(
            cfg, repo_root=tmp_path, out_dir=out, runner=lambda c, d: 3, dtype="float16", check=lambda c, r: 0
        )


def test_write_metrics_renames_eval_loss(tmp_path: Path) -> None:
    """Bảo vệ: metrics.jsonl được dựng từ checkpoint mới nhất, `eval_loss` thành `eval_unseen_loss`."""
    assert swift_sft.write_metrics(tmp_path / "none", tmp_path / "m.jsonl") == 0
    hist = [{"step": 10, "loss": 1.5}, {"step": 50, "eval_loss": 0.9}]
    _checkpoint(tmp_path / "swift", 50, hist)
    assert swift_sft.write_metrics(tmp_path / "swift", tmp_path / "m.jsonl") == 2
    rows = [json.loads(x) for x in (tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"step": 10, "loss": 1.5}, {"step": 50, "eval_unseen_loss": 0.9}]
