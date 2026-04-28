#!/usr/bin/env bash
# Apply third-party patches required for the ROCm build of mini-sglang.
#
# Idempotent — re-running is safe. Each patch is probed via `patch -R --dry-run`
# first; if it applies cleanly in reverse, the patch is already in place and
# gets skipped.
#
# Usage:
#   scripts/apply_rocm_patches.sh             # uses `python` on PATH
#   scripts/apply_rocm_patches.sh /app/.venv/bin/python
#
# Invoked by Dockerfile.rocm after `uv sync`. For bare-metal dev, run it
# manually after `uv sync --extra rocm --frozen` (or any re-sync that could
# reinstall torch-c-dlpack-ext).

set -euo pipefail

PY="${1:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES_DIR="${REPO_ROOT}/patches"

apply_patch_at() {
    local patch_file="$1"
    local target_dir="$2"

    if [[ ! -d "${target_dir}" ]]; then
        echo "[fail] target dir not found: ${target_dir}" >&2
        return 1
    fi

    if patch --dry-run -R -s -p1 -d "${target_dir}" < "${patch_file}" >/dev/null 2>&1; then
        echo "[skip] $(basename "${patch_file}") already applied under ${target_dir}"
        return 0
    fi

    echo "[apply] $(basename "${patch_file}") -> ${target_dir}"
    patch -p1 -d "${target_dir}" < "${patch_file}"
}

# ---------------------------------------------------------------------------
# torch_c_dlpack_ext: fall back to -cpu.so on HIP (see patch header).
# ---------------------------------------------------------------------------
TORCH_C_DIR="$("${PY}" -c 'import importlib.util, os; s = importlib.util.find_spec("torch_c_dlpack_ext"); print(os.path.dirname(s.origin))')"
apply_patch_at "${PATCHES_DIR}/torch_c_dlpack_ext_rocm.patch" "${TORCH_C_DIR}"

echo "ROCm patches applied."
