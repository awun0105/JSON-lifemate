#!/usr/bin/env bash
# Entrypoint của container (SPEC 6.3): /data chứa một split có cấu trúc giống public_test; ToolEnv
# cần <root>/<split>/..., nên tạo liên kết <ZAIC_ROOT>/<split> -> /data rồi chạy CLI ghi vào result/.
#
# Cách hoạt động:
#   1. Tìm đúng một file *_tools_catalog.json trong /data; không đúng một file thì thoát mã 2.
#   2. Lấy tên split từ tên file đó (phần trước _tools_catalog.json).
#   3. Tạo symlink <ZAIC_ROOT>/<split> -> /data và thư mục kết quả.
#   4. Thư mục kết quả còn trace.jsonl của lần chạy trước thì đổi tên thành trace.<thời điểm>.bak:
#      mỗi lần BTC chạy phải là một lần chạy mới, độc lập (thể lệ: chạy 5 lần lấy trung bình), không
#      được chạy tiếp (--resume) trên kết quả cũ, cũng không được chết vì config cũ lệch config mới.
#   5. exec CLI của LifeMate, chuyển tiếp mọi tham số thêm.
#
# Biến môi trường: DATA_DIR (/data), RESULT_DIR (result), ZAIC_ROOT (/tmp/zaic_root),
# LIFEMATE_CONFIG (configs/docker.toml). LIFEMATE_DRY_RUN=1 in lệnh CLI thay vì chạy (để test).
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RESULT_DIR="${RESULT_DIR:-result}"
ZAIC_ROOT="${ZAIC_ROOT:-/tmp/zaic_root}"
CONFIG="${LIFEMATE_CONFIG:-configs/docker.toml}"

# nullglob: không khớp file nào thì mảng rỗng thay vì giữ nguyên chuỗi mẫu.
shopt -s nullglob
catalogs=("${DATA_DIR}"/*_tools_catalog.json)
shopt -u nullglob
if [[ ${#catalogs[@]} -ne 1 ]]; then
    echo "docker-entrypoint: expected exactly one ${DATA_DIR}/*_tools_catalog.json, found ${#catalogs[@]}" >&2
    for c in "${catalogs[@]}"; do echo "  ${c}" >&2; done
    exit 2
fi
# Tên split = tên file bỏ hậu tố _tools_catalog.json.
name="$(basename "${catalogs[0]}")"
split="${name%_tools_catalog.json}"
if [[ -z "${split}" ]]; then
    echo "docker-entrypoint: cannot read a split name from ${catalogs[0]}" >&2
    exit 2
fi

# -sfn: ghi đè liên kết cũ nếu container chạy lại.
mkdir -p "${ZAIC_ROOT}"
ln -sfn "${DATA_DIR}" "${ZAIC_ROOT}/${split}"
mkdir -p "${RESULT_DIR}"
if [[ -e "${RESULT_DIR}/trace.jsonl" ]]; then
    bak="${RESULT_DIR}/trace.$(date -u +%Y%m%dT%H%M%SZ).bak"
    echo "docker-entrypoint: ${RESULT_DIR}/trace.jsonl exists from an earlier run; moved to ${bak}" >&2
    mv "${RESULT_DIR}/trace.jsonl" "${bak}"
fi
echo "docker-entrypoint: split=${split} data=${DATA_DIR} root=${ZAIC_ROOT} result=${RESULT_DIR} config=${CONFIG}" >&2

cmd=(python -m lifemate.run.cli --config "${CONFIG}" --env docker --split "${split}" --out "${RESULT_DIR}" "$@")
if [[ "${LIFEMATE_DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "${cmd[@]}"
    exit 0
fi
exec "${cmd[@]}"
