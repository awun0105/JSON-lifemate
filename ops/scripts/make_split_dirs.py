"""Dựng thư mục dữ liệu riêng cho từng tập (`data/split/<tên>/`) theo đúng cấu trúc thư mục của BTC.

Cách dùng (từ gốc repo): python ops/scripts/make_split_dirs.py [--src data/train+public_test/train] [--out data/split]

Mỗi `data/split/<tên>/` gồm `<tên>.jsonl` (các dòng của `train.jsonl` thuộc tập, giữ thứ tự gốc), `audio/<id>/`,
`<tên>_environment/<id>/`, `<tên>_tools_catalog.json`, `zaic_tools/` và `ids.txt`; mọi lệnh của dự án chạy được bằng
`--root data/split --split <tên>`. Danh sách id lấy từ `configs/splits/` (`make splits`).

Cách hoạt động:
- File âm thanh, thư mục trạng thái app, catalog và `zaic_tools` là hard link tới bản gốc: không tốn thêm dung lượng,
  không mở đọc nội dung (trạng thái app chỉ được đọc qua `lifemate/io/zaic.py`). Hard link dùng chung dữ liệu với bản
  gốc, nên không sửa file trong `data/split/`.
- Chỉ file `<tên>.jsonl` và `ids.txt` là file mới.
- Dữ liệu thi của BTC: `data/` không lên git.
- Nhóm câu (`SUBSET_SPLITS`), chia theo lời chuẩn có hay không có từ tiếng Anh (`foreign_words.find_foreign_words`,
  đúng cách `eval_asr` tách `vi_only_*` và `en_mixed_*`): `<tên>/vi_only/`, `<tên>/en_mixed/` (câu người dùng ở mọi
  lượt) và `<tên>/final_vi_only/`, `<tên>/final_en_mixed/` (chỉ câu lượt cuối). Mỗi nhóm có `utterances.jsonl` (`id`,
  `turn` là "final" hoặc chỉ số lượt trong `history` như các file `hyps.jsonl`, `audio`, `ref`) và `audio/<id>/` (hard
  link). Ở mức mẫu: `<tên>/samples_all_vi.txt` (mọi câu người dùng thuần Việt) và `<tên>/samples_with_en.txt` (có ít
  nhất một câu chêm tiếng Anh).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# tên thư mục -> file id trong configs/splits/
SPLITS = {
    "train": "train",  # 1 093 mẫu: dữ liệu học (fine-tune; bộ chọn H-Anh, H-Việt học trên cả tập); = tune + learn
    "tune": "asr_tune",  # 300 mẫu của train: chọn cấu hình ASR
    "learn": "asr_learn",  # 793 mẫu còn lại của train; tập học H của lần 3, từ D54 không còn vai riêng
    "smoke": "smoke",  # 30 mẫu của train: chạy thử nhanh
    "val_seen": "val_seen",  # 274 mẫu: kiểm tra, mỗi ứng viên đo một lần
    "val_unseen": "val_unseen",  # 178 mẫu: KHOÁ, chỉ đo một lần cho cấu hình sắp nộp (D53)
}


SUBSET_SPLITS = ("val_seen",)


def write_subsets(d: Path, rows: list[dict[str, object]]) -> dict[str, int]:
    """Ghi nhóm câu thuần Việt / có tiếng Anh của một tập vào `d/vi_only/`, `d/en_mixed/` và hai danh sách mẫu."""
    from lifemate.eval.foreign_words import find_foreign_words

    names = ("vi_only", "en_mixed", "final_vi_only", "final_en_mixed")
    groups: dict[str, list[dict[str, object]]] = {g: [] for g in names}
    all_vi, with_en = [], []
    for r in rows:
        turns = [("final", r["audio"], r["audio_text"])] + [
            (str(i), h.get("audio"), h.get("audio_text"))
            for i, h in enumerate(r.get("history") or [])  # type: ignore[union-attr]
            if h.get("role") == "user" and h.get("audio_text")
        ]
        has_en = False
        for turn, audio, ref in turns:
            if not ref:
                continue
            g = "en_mixed" if find_foreign_words(str(ref)) else "vi_only"
            has_en |= g == "en_mixed"
            item = {"id": r["id"], "turn": turn, "audio": audio, "ref": ref}
            groups[g].append(item)
            if turn == "final":
                groups[f"final_{g}"].append(item)
        (with_en if has_en else all_vi).append(str(r["id"]))
    for g, items in groups.items():
        gd = d / g
        gd.mkdir()
        (gd / "utterances.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8"
        )
        for x in items:
            src = d / str(x["audio"])
            (gd / str(x["audio"])).parent.mkdir(parents=True, exist_ok=True)
            os.link(src, gd / str(x["audio"]))
    (d / "samples_all_vi.txt").write_text("".join(i + "\n" for i in all_vi), encoding="utf-8")
    (d / "samples_with_en.txt").write_text("".join(i + "\n" for i in with_en), encoding="utf-8")
    return {**{g: len(v) for g, v in groups.items()}, "samples_all_vi": len(all_vi), "samples_with_en": len(with_en)}


def link_tree(src: Path, dst: Path) -> int:
    """Hard link mọi file của `src` vào `dst` (giữ cấu trúc thư mục); trả số file."""
    n = 0
    for dirpath, _, files in os.walk(src):
        target = dst / Path(dirpath).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            os.link(Path(dirpath) / f, target / f)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/train+public_test/train"))
    ap.add_argument("--out", type=Path, default=Path("data/split"))
    args = ap.parse_args()
    rows = {}
    for line in (args.src / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows[json.loads(line)["id"]] = line
    order = list(rows)
    for name, ids_name in SPLITS.items():
        ids = {x.strip() for x in Path(f"configs/splits/{ids_name}.txt").read_text().splitlines() if x.strip()}
        missing = ids - set(rows)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} id không có trong train.jsonl")
        keep = [i for i in order if i in ids]
        d = args.out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / f"{name}.jsonl").write_text("".join(rows[i] + "\n" for i in keep), encoding="utf-8")
        (d / "ids.txt").write_text("".join(i + "\n" for i in keep), encoding="utf-8")
        n_audio = sum(link_tree(args.src / "audio" / i, d / "audio" / i) for i in keep)
        n_env = sum(link_tree(args.src / "train_environment" / i, d / f"{name}_environment" / i) for i in keep)
        os.link(args.src / "train_tools_catalog.json", d / f"{name}_tools_catalog.json")
        link_tree(args.src / "zaic_tools", d / "zaic_tools")
        print(f"{name}: {len(keep)} mẫu, {n_audio} file âm thanh, {n_env} file trạng thái")
        if name in SUBSET_SPLITS:
            print(f"  nhóm câu: {write_subsets(d, [json.loads(rows[i]) for i in keep])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
