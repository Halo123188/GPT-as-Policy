#!/usr/bin/env bash
# One frozen RoboDojo case driven by a locally served Qwen VLM instead of Codex.
# Same simulator session, case identity, action contract and native scoring as
# the Codex methods; the only substitution is the controlling model.
#   ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt  hybrid: Qwen reviews/corrects pi05
#   ROLLOUT_EVALUATION_METHOD=gpt_only       direct: Qwen acts from observations
# No credentials: the endpoint is an operator-started OpenAI-compatible server.
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT (contains src/RoboDojo, sim-venv, checkpoints)}"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-$RUNTIME_ROOT/src/RoboDojo}"
OPENPI_SOURCE="${OPENPI_SOURCE:-$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$OPENPI_SOURCE/.venv/bin/python}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-$RUNTIME_ROOT/sim-venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999}"
ROLLOUT_CASE_ID="${ROLLOUT_CASE_ID:?Set a frozen case ID, e.g. build_tower__standard__g0__l0}"
ROLLOUT_EVALUATION_METHOD="${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}"
ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:-qwen_hybrid_panel50}"
ROLLOUT_ATTEMPT="${ROLLOUT_ATTEMPT:-0}"
ROLLOUT_EVAL_MANIFEST="${ROLLOUT_EVAL_MANIFEST:-$CODE_ROOT/hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUNTIME_ROOT/results/$ROLLOUT_EXPERIMENT_ID}"
POLICY_GPU="${POLICY_GPU:-0}"
SIM_GPU="${SIM_GPU:-0}"
STARTUP_SECONDS="${STARTUP_SECONDS:-1800}"
MAX_DECISIONS="${MAX_DECISIONS:-0}"  # 0 = native termination only, as in the report's campaign
case "$ROLLOUT_EVALUATION_METHOD" in
    pi05_plus_gpt|gpt_only) ;;
    *) echo 'Invalid evaluation method' >&2; exit 2 ;;
esac
ROLLOUT_QWEN_BASE_URL="${ROLLOUT_QWEN_BASE_URL:?Set ROLLOUT_QWEN_BASE_URL to an OpenAI-compatible server, e.g. http://host:8000/v1}"
ROLLOUT_QWEN_MODEL="${ROLLOUT_QWEN_MODEL:-Qwen/Qwen3.8-27B}"
export ROLLOUT_QWEN_BASE_URL ROLLOUT_QWEN_MODEL
for identifier in "$ROLLOUT_CASE_ID" "$ROLLOUT_EXPERIMENT_ID" "$ROLLOUT_ATTEMPT"; do
    [[ "$identifier" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid identifier: $identifier" >&2; exit 2; }
done
required=("$ROBODOJO_PYTHON" "$ROBODOJO_SOURCE/Assets" "$ROLLOUT_EVAL_MANIFEST")
if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
    required+=("$OPENPI_PYTHON" "$CHECKPOINT/params"
        "$CHECKPOINT/assets/arx_x5_sim/norm_stats.json" "$OPENPI_SOURCE/src/openpi")
fi
for input in "${required[@]}"; do
    [[ -e "$input" ]] || { echo "Missing input: $input" >&2; exit 2; }
done
"$ROBODOJO_PYTHON" -c 'import importlib.metadata as m; assert m.version("isaacsim").startswith("5.1."), "RoboDojo requires Isaac Sim 5.1"'
# Fail before any physics if the model server is not actually answering.
"$ROBODOJO_PYTHON" - "$ROLLOUT_QWEN_BASE_URL" "$ROLLOUT_QWEN_MODEL" <<'PY'
import json, sys, urllib.request
base, wanted = sys.argv[1].rstrip('/'), sys.argv[2]
with urllib.request.urlopen(base + '/models', timeout=60) as response:
    served = [row['id'] for row in json.load(response)['data']]
assert wanted in served, f'{wanted} not served at {base}; found {served}'
print(f'model server ready: {wanted} at {base}')
PY
free_port() { "$ROBODOJO_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
POLICY_PORT="${POLICY_PORT:-$(free_port)}"
SIM_PORT="${SIM_PORT:-$(free_port)}"
[[ "$POLICY_PORT" != "$SIM_PORT" ]] || { echo 'Policy and simulator ports collide' >&2; exit 2; }

ARCHIVE="$RESULTS_ROOT/$ROLLOUT_CASE_ID/attempt_$ROLLOUT_ATTEMPT"
SNAPSHOT="$ARCHIVE/source_snapshot"
mkdir -p "$(dirname "$ARCHIVE")"
mkdir "$ARCHIVE" || { echo "Refusing existing archive: $ARCHIVE" >&2; exit 2; }
mkdir -p "$SNAPSHOT" "$ARCHIVE/logs" "$ARCHIVE/sim"
cp -a "$CODE_ROOT/hybrid_rollout" "$SNAPSHOT/hybrid_rollout"
cp "$ROLLOUT_EVAL_MANIFEST" "$ARCHIVE/evaluation_manifest.json"
eval "$(PYTHONPATH="$SNAPSHOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.pi05_only prepare-case \
    --manifest "$ARCHIVE/evaluation_manifest.json" --case-id "$ROLLOUT_CASE_ID" \
    --output "$ARCHIVE/evaluation_case.json")"
export ROLLOUT_EVAL_MANIFEST_SHA256="$PANEL_SHA256"
git -C "$ROBODOJO_SOURCE" rev-parse HEAD > "$ARCHIVE/robodojo_head.txt"
[[ "$ROLLOUT_EVALUATION_METHOD" != pi05_plus_gpt ]] || \
    git -C "$OPENPI_SOURCE" rev-parse HEAD > "$ARCHIVE/openpi_head.txt"

TASK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/robodojo_${ROLLOUT_CASE_ID}.XXXXXX")"
export TASK_ROOT
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RUNTIME_ROOT/openpi-cache}"
export PYTHONUNBUFFERED=1
# A local 27B model at ~25 tok/s can think for well over the simulator's default
# 900 s inter-request timeout, especially in direct mode where one decision can
# run to the full output budget. Without this the simulator closes the episode
# as `server_close` with zero control steps.
export ROBODOJO_RPC_TIMEOUT_SECONDS="${ROBODOJO_RPC_TIMEOUT_SECONDS:-7200}"
MANIFEST_KEYS='CODE_ROOT RUNTIME_ROOT ROBODOJO_SOURCE OPENPI_SOURCE OPENPI_PYTHON ROBODOJO_PYTHON CHECKPOINT
    ROLLOUT_CASE_ID ROLLOUT_EXPERIMENT_ID ROLLOUT_ATTEMPT ROLLOUT_EVAL_MANIFEST ROLLOUT_EVAL_MANIFEST_SHA256
    ROLLOUT_EVALUATION_METHOD ROLLOUT_QWEN_BASE_URL ROLLOUT_QWEN_MODEL MAX_DECISIONS
    ROBODOJO_RPC_TIMEOUT_SECONDS
    RESULTS_ROOT POLICY_GPU SIM_GPU POLICY_PORT SIM_PORT TASK_ROOT OPENPI_DATA_HOME
    RUNTIME_TASK BASE_TASK VARIANT LAYOUT_ID EVAL_SEED SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID'
export MANIFEST_KEYS
export $MANIFEST_KEYS  # Word splitting intended; names without a value stay out of the environment.
"$ROBODOJO_PYTHON" - "$ARCHIVE/launch_manifest.json" <<'PY'
import json, os, platform, sys
from datetime import datetime, timezone
values = {key: os.environ[key] for key in os.environ['MANIFEST_KEYS'].split() if key in os.environ}
json.dump(dict(schema='hybrid_rollout.robodojo.launch.v1',
    evaluation_method=values['ROLLOUT_EVALUATION_METHOD'], controller_backend='qwen',
    created_utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(),
    runtime_task=values['RUNTIME_TASK'], layout_id=int(values['LAYOUT_ID']),
    eval_seed=int(values['EVAL_SEED']), panel_sha256=values['ROLLOUT_EVAL_MANIFEST_SHA256'],
    resolved_environment=values), open(sys.argv[1], 'x'), indent=2)
PY

policy_pid=''; sim_pid=''; controller_pid=''
stop_child() {
    local child="$1"
    [[ -n "$child" ]] || return 0
    if kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
        for _ in {1..40}; do kill -0 "$child" 2>/dev/null || break; sleep .5; done
        kill -KILL "$child" 2>/dev/null || true
    fi
    wait "$child" 2>/dev/null || true
}
cleanup() {
    local status=$?
    trap - EXIT
    set +e
    stop_child "$controller_pid"; stop_child "$sim_pid"; stop_child "$policy_pid"
    printf '%s\n' "$status" > "$ARCHIVE/job_exit_status.txt"
    rm -rf -- "$TASK_ROOT"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
(
    cd "$SNAPSHOT"
    exec env CUDA_VISIBLE_DEVICES="$POLICY_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="$SNAPSHOT:$OPENPI_SOURCE/src:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$OPENPI_PYTHON" -m hybrid_rollout.robodojo.pi05_server.server \
        --checkpoint "$CHECKPOINT" --port "$POLICY_PORT" --identity-output "$ARCHIVE/policy_identity.json"
) > "$ARCHIVE/logs/policy.log" 2>&1 &
policy_pid=$!
fi
(
    export CUDA_VISIBLE_DEVICES="$SIM_GPU"
    export PYTHONPATH="$SNAPSHOT:$ROBODOJO_SOURCE:$ROBODOJO_SOURCE/XPolicyLab:$ROBODOJO_SOURCE/third_party/curobo${SIM_EXTRA_PYTHONPATH:+:$SIM_EXTRA_PYTHONPATH}"
    source "$SNAPSHOT/hybrid_rollout/robodojo/robodojo_server/runtime.sh"
    mkdir -p "$ARCHIVE/sim/native_runtime"
    cd "$ARCHIVE/sim/native_runtime"
    exec "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.robodojo_server.server \
        --task "$RUNTIME_TASK" --output "$ARCHIVE/sim" --port "$SIM_PORT" --eval-seed "$EVAL_SEED" \
        --eval-manifest "$ARCHIVE/evaluation_manifest.json" --case-file "$ARCHIVE/evaluation_case.json"
) > "$ARCHIVE/logs/sim.log" 2>&1 &
sim_pid=$!
deadline=$((SECONDS+STARTUP_SECONDS))
until { [[ "$ROLLOUT_EVALUATION_METHOD" == gpt_only || -f "$ARCHIVE/policy_identity.json" ]]; } \
        && grep -q '"event": "ready"' "$ARCHIVE/logs/sim.log"; do
    if [[ -n "$policy_pid" ]]; then
        kill -0 "$policy_pid" 2>/dev/null || { echo "Policy exited: $ARCHIVE/logs/policy.log" >&2; exit 3; }
    fi
    kill -0 "$sim_pid" 2>/dev/null || { echo "Simulator exited: $ARCHIVE/logs/sim.log" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo 'Service startup timeout' >&2; exit 3; }
    sleep 2
done
(
    cd "$SNAPSHOT"
    controller_args=(--evaluation-method "$ROLLOUT_EVALUATION_METHOD")
    controller_pythonpath="$SNAPSHOT"
    if [[ "$ROLLOUT_EVALUATION_METHOD" == pi05_plus_gpt ]]; then
        controller_args+=(--checkpoint "$CHECKPOINT" --student-port "$POLICY_PORT")
        controller_pythonpath="$SNAPSHOT:$OPENPI_SOURCE/packages/openpi-client/src"
    fi
    exec env PYTHONPATH="$controller_pythonpath" \
        "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.skill.run \
        --output "$ARCHIVE/controller" --task "$RUNTIME_TASK" "${controller_args[@]}" \
        --backend qwen --base-url "$ROLLOUT_QWEN_BASE_URL" --model "$ROLLOUT_QWEN_MODEL" \
        --sim-port "$SIM_PORT" --seed "$LAYOUT_ID" --max-decisions "$MAX_DECISIONS"
) > "$ARCHIVE/logs/controller.log" 2>&1 &
controller_pid=$!
wait "$controller_pid"
controller_pid=''
[[ -f "$ARCHIVE/controller/result.json" ]] || { echo 'No rollout result' >&2; exit 4; }
for _ in {1..120}; do kill -0 "$sim_pid" 2>/dev/null || break; [[ -f "$ARCHIVE/sim/evaluation_outcome.json" ]] && break; sleep 1; done
cat "$ARCHIVE/sim/evaluation_outcome.json"
