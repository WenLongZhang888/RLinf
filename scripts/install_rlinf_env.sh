#!/usr/bin/env bash
# Install a named RLinf environment with all caches kept under this repository.
#
# Usage:
#   source scripts/install_rlinf_env.sh <env_name> embodied --model openpi --env maniskill_libero
#
# Examples:
#   source scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero --no-root
#   bash scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero --no-root
#
# Directory layout:
#   .venvs/<env_name>/                 Python virtual environment
#   .cache/<env_name>/uv/              uv package/build cache
#   .cache/<env_name>/pip/             pip cache
#   .cache/<env_name>/downloads/       ManiSkill / SAPIEN / OpenPI downloads
#   .cache/<env_name>/huggingface/     Hugging Face cache
#   .cache/<env_name>/torch/           Torch cache
#   .cache/<env_name>/xdg/             XDG cache
#   .cache/openpi/                     OpenPI data home
#   .checkpoints/<env_name>/           User-managed model checkpoints
#
# Optional:
#   export LIBERO_PATH=/path/to/LIBERO
#   source scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero
#
# If system dependency installation blocks on sudo:
#   source scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero --no-root

set -euo pipefail

is_sourced() {
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

print_usage() {
    cat <<'EOF'
Usage:
  source scripts/install_rlinf_env.sh <env_name> <target> [install.sh options]
  bash scripts/install_rlinf_env.sh <env_name> <target> [install.sh options]

Examples:
  source scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero --no-root
  bash scripts/install_rlinf_env.sh pirl embodied --model openpi --env maniskill_libero --no-root

Notes:
  Use "source" if you want the environment activated in the current shell after installation.
EOF
}

strip_venv_args() {
    local -n _input_args=$1
    local -n _output_args=$2

    _output_args=()
    local skip_next=0
    local arg
    for arg in "${_input_args[@]}"; do
        if (( skip_next )); then
            skip_next=0
            continue
        fi

        case "$arg" in
            --venv)
                skip_next=1
                ;;
            --venv=*)
                ;;
            *)
                _output_args+=("$arg")
                ;;
        esac
    done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    print_usage
    if is_sourced; then
        return 0
    fi
    exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMED_ENV="${1:?missing environment name}"
shift

RLINF_ROOT="${RLINF_ROOT:-$ROOT}"
if [[ ! -f "$RLINF_ROOT/requirements/install.sh" ]]; then
    echo "[install_rlinf_env] RLinf install script not found: $RLINF_ROOT/requirements/install.sh" >&2
    echo "[install_rlinf_env] Set RLINF_ROOT to the RLinf repository root if needed." >&2
    if is_sourced; then
        return 1
    fi
    exit 1
fi

RAW_INSTALL_ARGS=("$@")
INSTALL_ARGS=()
strip_venv_args RAW_INSTALL_ARGS INSTALL_ARGS

VENV_PATH="$ROOT/.venvs/$NAMED_ENV"
CACHE_ROOT="$ROOT/.cache/$NAMED_ENV"
CHECKPOINT_ROOT="$ROOT/.checkpoints/$NAMED_ENV"

mkdir -p \
    "$ROOT/.venvs" \
    "$CHECKPOINT_ROOT" \
    "$CACHE_ROOT/uv" \
    "$CACHE_ROOT/pip" \
    "$CACHE_ROOT/downloads" \
    "$CACHE_ROOT/huggingface" \
    "$CACHE_ROOT/torch" \
    "$CACHE_ROOT/xdg" \
    "$ROOT/.cache/openpi"

# Keep package managers and heavyweight data caches inside the workspace.
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export DOWNLOAD_DIR="$CACHE_ROOT/downloads"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
export RLINF_CHECKPOINT_HOME="${RLINF_CHECKPOINT_HOME:-$CHECKPOINT_ROOT}"

# Mainland China friendly defaults. Users can export their own values before
# running this script to override them.
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

mkdir -p \
    "$UV_CACHE_DIR" \
    "$PIP_CACHE_DIR" \
    "$DOWNLOAD_DIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME" \
    "$OPENPI_DATA_HOME" \
    "$RLINF_CHECKPOINT_HOME"

echo "[install_rlinf_env] workspace=$ROOT"
echo "[install_rlinf_env] rlinf_root=$RLINF_ROOT"
echo "[install_rlinf_env] named_env=$NAMED_ENV"
echo "[install_rlinf_env] venv=$VENV_PATH"
echo "[install_rlinf_env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[install_rlinf_env] PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "[install_rlinf_env] DOWNLOAD_DIR=$DOWNLOAD_DIR"
echo "[install_rlinf_env] HF_HOME=$HF_HOME"
echo "[install_rlinf_env] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[install_rlinf_env] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[install_rlinf_env] TORCH_HOME=$TORCH_HOME"
echo "[install_rlinf_env] XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "[install_rlinf_env] OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "[install_rlinf_env] RLINF_CHECKPOINT_HOME=$RLINF_CHECKPOINT_HOME"
echo "[install_rlinf_env] UV_INDEX_URL=$UV_INDEX_URL"
echo "[install_rlinf_env] HF_ENDPOINT=$HF_ENDPOINT"
echo "[install_rlinf_env] GIT_LFS_SKIP_SMUDGE=$GIT_LFS_SKIP_SMUDGE"
if [[ -n "${LIBERO_PATH:-}" ]]; then
    echo "[install_rlinf_env] LIBERO_PATH=$LIBERO_PATH"
fi

(
    cd "$RLINF_ROOT"
    bash requirements/install.sh "${INSTALL_ARGS[@]}" --venv "$VENV_PATH"
)

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "[install_rlinf_env] install finished, but activate script was not found: $VENV_PATH/bin/activate" >&2
    if is_sourced; then
        return 1
    fi
    exit 1
fi

if is_sourced; then
    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
    echo "[install_rlinf_env] activated: $VENV_PATH"
else
    echo "[install_rlinf_env] installed: $VENV_PATH"
    echo "[install_rlinf_env] To activate in your current shell, run:"
    echo "  source $VENV_PATH/bin/activate"
    echo "[install_rlinf_env] Or install+activate in one command next time:"
    echo "  source scripts/install_rlinf_env.sh $NAMED_ENV ${INSTALL_ARGS[*]}"
fi
