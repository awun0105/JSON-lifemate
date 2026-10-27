#!/usr/bin/env bash
# Entrypoint của container (SPEC 6.3): /data chứa một split có cấu trúc giống public_test; ToolEnv
# cần <root>/<split>/..., nên tạo liên kết <ZAIC_ROOT>/<split> -> /data rồi chạy CLI ghi vào result/.
#
# Cách hoạt động:
#   1. Tìm đúng một file *_tools_catalog.json trong /data; không đúng một file thì thoát mã 2.
#   2. Lấy tên split từ tên file đó (phần trước _tools_catalog.json).
#   3. Tạo symlink <ZAIC_ROOT>/<split> -> /data và thư mục result/.
#   4. exec CLI của LifeMate với --resume, chuyển tiếp mọi tham số thêm.
set -euo pipefail

# Đường dẫn có thể ghi đè qua biến môi trường.
DATA_DIR="${DATA_DIR:-/data}"
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
mkdir -p result
echo "docker-entrypoint: split=${split} data=${DATA_DIR} root=${ZAIC_ROOT} config=${CONFIG}" >&2

# --resume an toàn khi result/ còn rỗng: RunStore tạo thư mục và không thấy id nào đã xong.
exec python -m lifemate.run.cli --config "${CONFIG}" --env docker --split "${split}" --out result --resume "$@"
