#!/usr/bin/env bash
# dev-setup.sh — set up a bare-metal development environment for mini-sglang
#
# Usage:
#   ./scripts/dev-setup.sh                 # auto-detect platform
#   ./scripts/dev-setup.sh --platform rocm
#   ./scripts/dev-setup.sh --platform cuda
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Python 3.12 (or override via PYTHON_VERSION env var)
#
# The script creates .venv/ in the repo root and installs the package in
# editable mode with the dev extra and the appropriate platform extra.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-python3.12}"
VENV_DIR="${REPO_ROOT}/.venv"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="$2"; shift 2 ;;
        --platform=*)
            PLATFORM="${1#*=}"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--platform cuda|rocm]" >&2
            exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Auto-detect platform when not specified
# ---------------------------------------------------------------------------
if [[ -z "$PLATFORM" ]]; then
    if command -v rocm-smi &>/dev/null && rocm-smi --showid &>/dev/null 2>&1; then
        PLATFORM="rocm"
    else
        PLATFORM="cuda"
    fi
    echo "Auto-detected platform: ${PLATFORM}"
fi

if [[ "$PLATFORM" != "cuda" && "$PLATFORM" != "rocm" ]]; then
    echo "Error: --platform must be 'cuda' or 'rocm', got '${PLATFORM}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
echo "Creating venv at ${VENV_DIR} with ${PYTHON_VERSION}..."
uv venv --python="${PYTHON_VERSION}" "${VENV_DIR}"

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

if [[ "$PLATFORM" == "rocm" ]]; then
    echo "Installing ROCm dependencies..."
    # Step 1: install torch from the ROCm 6.2 wheel index first so that the
    # subsequent package install does not pull the CUDA build from PyPI.
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        "torch==2.5.1+rocm6.2" \
        --index-url https://download.pytorch.org/whl/rocm6.2

    # Step 2: install the package; torch constraint is already satisfied by
    # the ROCm wheel above, so uv will not reinstall from PyPI.
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        -e ".[rocm,dev]"

    # torch-c-dlpack-ext is a manylinux wheel on PyPI
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        torch-c-dlpack-ext

    # torch-c-dlpack-ext ships a prebuilt .so that links against
    # libtorch_cuda.so, which doesn't exist on ROCm (there's libtorch_hip.so).
    # Patch core.py to use the -cpu variant when torch.version.hip is set;
    # the cpu variant resolves all deps through libtorch_python.so and works
    # correctly for ROCm tensors.
    DLPACK_CORE=$(${VENV_DIR}/bin/python -c \
        "import torch_c_dlpack_ext, os; print(os.path.join(os.path.dirname(torch_c_dlpack_ext.__file__), 'core.py'))")
    sed -i 's/suffix = "cuda" if torch.cuda.is_available() else "cpu"/suffix = "cuda" if (torch.cuda.is_available() and not getattr(torch.version, "hip", None)) else "cpu"/' \
        "${DLPACK_CORE}"
    echo "Patched torch_c_dlpack_ext for ROCm: ${DLPACK_CORE}"

    # ---------------------------------------------------------------------------
    # ROCm environment hints
    # ---------------------------------------------------------------------------
    cat << 'EOF'

ROCm environment variables — add these to your shell profile or source this
block before running minisgl:

    export ROCM_HOME=/opt/rocm
    export PATH="${ROCM_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${LD_LIBRARY_PATH}"
    export HSA_OVERRIDE_GFX_VERSION=9.0.0   # CDNA2 / gfx90a
    export TVM_FFI_CACHE_DIR="${HOME}/.cache/tvm-ffi"
    export HF_HOME="${HOME}/.cache/huggingface"

Or activate the venv and run:
    source .venv/bin/activate
    python -m minisgl --help
EOF

else
    echo "Installing CUDA dependencies (torch from PyPI)..."
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        -e ".[cuda,dev]"

    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        torch-c-dlpack-ext

    cat << 'EOF'

CUDA environment variables — add to your shell profile if needed:

    export CUDA_HOME=/usr/local/cuda
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    export TVM_FFI_CACHE_DIR="${HOME}/.cache/tvm-ffi"
    export HF_HOME="${HOME}/.cache/huggingface"
    export FLASHINFER_WORKSPACE_BASE="${HOME}/.cache/flashinfer"

Or activate the venv and run:
    source .venv/bin/activate
    python -m minisgl --help
EOF
fi

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
