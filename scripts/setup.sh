#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/' >&2; exit 1; }
python3 -m hint_searchqa.setup --framework
export UV_PROJECT_ENVIRONMENT="$ROOT/.deps/hint-sd/.venv"
uv sync --frozen --project .deps/hint-sd
uv venv --python 3.12 .env-searchqa
uv pip install --python .env-searchqa/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .env-searchqa/bin/python -r configs/requirements-cpu.txt
echo 'HINT-SD and CPU QA runtimes installed. Prepare the separate native FAISS/E5 runtime as documented.'
