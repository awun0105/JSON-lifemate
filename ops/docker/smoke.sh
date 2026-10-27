#!/usr/bin/env bash
# Build the image and run it the organisers' way (SPEC 6.3). Run on a rented GPU box with
# docker + nvidia-container-toolkit, not on a dev machine without a GPU.
#
#   DATA_SPLIT_DIR=data/train+public_test/public_test IMAGE=lifemate:dev ops/docker/smoke.sh
set -euo pipefail

DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-data/train+public_test/public_test}"
IMAGE="${IMAGE:-lifemate:dev}"
MODEL_DIR="${MODEL_DIR:-artifacts/models}"
OUT_DIR="${OUT_DIR:-$PWD/result}"

DATA_SPLIT_DIR="$(cd "${DATA_SPLIT_DIR}" && pwd)"
shopt -s nullglob
catalogs=("${DATA_SPLIT_DIR}"/*_tools_catalog.json)
shopt -u nullglob
[[ ${#catalogs[@]} -eq 1 ]] || { echo "need exactly one *_tools_catalog.json in ${DATA_SPLIT_DIR}" >&2; exit 2; }
split="$(basename "${catalogs[0]}" _tools_catalog.json)"

docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR="${MODEL_DIR}" -t "${IMAGE}" .

mkdir -p "${OUT_DIR}"
# uid 10001 in the image must be able to write the mounted result folder.
chmod a+rwx "${OUT_DIR}"
docker run --rm --gpus all --network none \
    -v "${DATA_SPLIT_DIR}:/data:ro" \
    -v "${OUT_DIR}:/app/result" \
    "${IMAGE}"

sub="${OUT_DIR}/submission.jsonl"
[[ -s "${sub}" ]] || { echo "missing or empty ${sub}" >&2; exit 1; }
root="$(dirname "${DATA_SPLIT_DIR}")"
(cd "${DATA_SPLIT_DIR}" && python3 -m zaic_tools --split "${split}" --root "${root}" --validate "${sub}")
echo "lines: $(wc -l < "${sub}")"
