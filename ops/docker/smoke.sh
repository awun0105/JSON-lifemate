#!/usr/bin/env bash
# Build image và chạy theo đúng cách ban tổ chức chạy (SPEC 6.3). Chạy trên máy GPU thuê có
# docker + nvidia-container-toolkit, không chạy trên máy phát triển không có GPU.
#
# Cách hoạt động:
#   1. Xác định thư mục split và tên split từ file *_tools_catalog.json duy nhất.
#   2. docker build image kèm trọng số mô hình.
#   3. docker run với GPU, không mạng (--network none), gắn dữ liệu chỉ đọc vào /data.
#   4. Kiểm tra submission.jsonl tồn tại, không rỗng và hợp lệ theo zaic_tools.
#
#   DATA_SPLIT_DIR=data/train+public_test/public_test IMAGE=lifemate:dev ops/docker/smoke.sh
set -euo pipefail

# Tham số, có thể ghi đè qua biến môi trường.
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-data/train+public_test/public_test}"
IMAGE="${IMAGE:-lifemate:dev}"
MODEL_DIR="${MODEL_DIR:-artifacts/models}"
OUT_DIR="${OUT_DIR:-$PWD/result}"

# Chuẩn hoá thành đường dẫn tuyệt đối (docker -v cần đường dẫn tuyệt đối).
DATA_SPLIT_DIR="$(cd "${DATA_SPLIT_DIR}" && pwd)"
shopt -s nullglob
catalogs=("${DATA_SPLIT_DIR}"/*_tools_catalog.json)
shopt -u nullglob
[[ ${#catalogs[@]} -eq 1 ]] || { echo "need exactly one *_tools_catalog.json in ${DATA_SPLIT_DIR}" >&2; exit 2; }
# Tên split = tên file catalog bỏ hậu tố _tools_catalog.json.
split="$(basename "${catalogs[0]}" _tools_catalog.json)"

docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR="${MODEL_DIR}" -t "${IMAGE}" .

mkdir -p "${OUT_DIR}"
# uid 10001 trong image phải ghi được vào thư mục result đã gắn.
chmod a+rwx "${OUT_DIR}"
# Không mạng giống môi trường chấm; dữ liệu gắn chỉ đọc.
docker run --rm --gpus all --network none \
    -v "${DATA_SPLIT_DIR}:/data:ro" \
    -v "${OUT_DIR}:/app/result" \
    "${IMAGE}"

# Kiểm tra file nộp: tồn tại, không rỗng, rồi xác thực định dạng bằng zaic_tools.
sub="${OUT_DIR}/submission.jsonl"
[[ -s "${sub}" ]] || { echo "missing or empty ${sub}" >&2; exit 1; }
root="$(dirname "${DATA_SPLIT_DIR}")"
(cd "${DATA_SPLIT_DIR}" && python3 -m zaic_tools --split "${split}" --root "${root}" --validate "${sub}")
echo "lines: $(wc -l < "${sub}")"
