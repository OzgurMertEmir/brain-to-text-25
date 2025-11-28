# echo "[info] Installing system build dependencies (cmake, build-essential) ..."
# apt-get update -y
# apt-get install -y cmake build-essential

# echo "[info] Installing Redis Server ..."
# apt-get install lsb-release curl gpg
# curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
# chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
# echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
# apt-get update
# apt-get install redis

# echo "[info] Removing previous LM build directories (if any) ..."
# rm -rf language_model/runtime/server/x86/build
# rm -rf language_model/runtime/server/x86/fc_base

# echo "[info] Installing Python Env Management Tool: uv ..."
# curl -LsSf https://astral.sh/uv/install.sh | sh

# echo "[info] Creating LM Python 3.9 Env under .venv-lm ..."
# uv venv --python 3.9 .venv-lm
# source .venv-lm/bin/activate
# uv sync --active --python=3.9

# echo "[info] Creating Python 3.11 Env under .venv ..."
# uv venv --python 3.11 .venv
# source .venv/bin/activate
# uv sync --active --python=3.11
# uv pip install -e .


