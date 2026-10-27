#!/usr/bin/env bash
# Container entrypoint (SPEC 6.3): /data holds one split laid out like public_test; ToolEnv
# wants <root>/<split>/..., so link <ZAIC_ROOT>/<split> -> /data and run the CLI into result/.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
ZAIC_ROOT="${ZAIC_ROOT:-/tmp/zaic_root}"
CONFIG="${LIFEMATE_CONFIG:-configs/docker.toml}"

shopt -s nullglob
catalogs=("${DATA_DIR}"/*_tools_catalog.json)
shopt -u nullglob
if [[ ${#catalogs[@]} -ne 1 ]]; then
    echo "docker-entrypoint: expected exactly one ${DATA_DIR}/*_tools_catalog.json, found ${#catalogs[@]}" >&2
    for c in "${catalogs[@]}"; do echo "  ${c}" >&2; done
    exit 2
fi
name="$(basename "${catalogs[0]}")"
split="${name%_tools_catalog.json}"
if [[ -z "${split}" ]]; then
    echo "docker-entrypoint: cannot read a split name from ${catalogs[0]}" >&2
    exit 2
fi

mkdir -p "${ZAIC_ROOT}"
ln -sfn "${DATA_DIR}" "${ZAIC_ROOT}/${split}"
mkdir -p result
echo "docker-entrypoint: split=${split} data=${DATA_DIR} root=${ZAIC_ROOT} config=${CONFIG}" >&2

# --resume is safe on an empty result/: RunStore creates the folder and finds no done ids.
exec python -m lifemate.cli --config "${CONFIG}" --env docker --split "${split}" --out result --resume "$@"
