#!/usr/bin/env bash
# An isolated Python for the Stable-Baselines3 tests, so that the main environment's packages are not touched.
#   scripts/make_sb3_env.sh [/path/to/venv]        (default /tmp/sb3env)
# The venv sees the main environment's packages (PyTorch, NumPy, pandas ...) and adds stable-baselines3, sb3-contrib and gymnasium on top. The tests find it through $QUANT_SB3_PYTHON or /tmp/sb3env/bin/python;
# without it they are skipped as dependency-blocked.
set -euo pipefail
dir="${1:-/tmp/sb3env}"
python -m venv --system-site-packages "$dir"
"$dir/bin/pip" install "stable-baselines3" "sb3-contrib" "gymnasium"
"$dir/bin/python" -c "import stable_baselines3, sb3_contrib, gymnasium; print('ready:', stable_baselines3.__version__, sb3_contrib.__version__, gymnasium.__version__)"
