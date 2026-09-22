#!/usr/bin/env bash
# pi05-only RoboDojo baseline for ONE frozen panel case: OpenPI server + native simulator
# + pi05_only controller. No Codex, credentials or /mnt layout required; one GPU suffices.
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT (contains src/RoboDojo, sim-venv, checkpoints)}"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-$RUNTIME_ROOT/src/RoboDojo}"
OPENPI_SOURCE="${OPENPI_SOURCE:-$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$OPENPI_SOURCE/.venv/bin/python}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-$RUNTIME_ROOT/sim-venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$RUNTIME_ROOT/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999}"
ROLLOUT_CASE_ID="${ROLLOUT_CASE_ID:?Set a frozen case ID, e.g. build_tower__standard__g0__l0}"
ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:-pi05_only_panel50}"
ROLLOUT_ATTEMPT="${ROLLOUT_ATTEMPT:-0}"
ROLLOUT_EVAL_MANIFEST="${ROLLOUT_EVAL_MANIFEST:-$CODE_ROOT/hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$RUNTIME_ROOT/results/$ROLLOUT_EXPERIMENT_ID}"
POLICY_GPU="${POLICY_GPU:-0}"
SIM_GPU="${SIM_GPU:-0}"
EXECUTE_STEPS="${EXECUTE_STEPS:-50}"
STARTUP_SECONDS="${STARTUP_SECONDS:-1800}"
for identifier in "$ROLLOUT_CASE_ID" "$ROLLOUT_EXPERIMENT_ID" "$ROLLOUT_ATTEMPT"; do
    [[ "$identifier" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid identifier: $identifier" >&2; exit 2; }
done
for input in "$ROBODOJO_PYTHON" "$OPENPI_PYTHON" "$ROBODOJO_SOURCE/Assets" "$CHECKPOINT/params" \
    "$CHECKPOINT/assets/arx_x5_sim/norm_stats.json" "$OPENPI_SOURCE/src/openpi" "$ROLLOUT_EVAL_MANIFEST"; do
    [[ -e "$input" ]] || { echo "Missing input: $input" >&2; exit 2; }
done
"$ROBODOJO_PYTHON" -c 'import importlib.metadata as m; assert m.version("isaacsim").startswith("5.1."), "RoboDojo requires Isaac Sim 5.1"'
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
git -C "$OPENPI_SOURCE" rev-parse HEAD > "$ARCHIVE/openpi_head.txt"

# Per-episode private runtime state (XDG runtime dir, temp files) on node-local storage.
TASK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/robodojo_${ROLLOUT_CASE_ID}.XXXXXX")"
export TASK_ROOT
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RUNTIME_ROOT/openpi-cache}"
export PYTHONUNBUFFERED=1
# Secret-free record of the resolved environment; runtime_manifest.py is Codex-specific.
MANIFEST_KEYS='CODE_ROOT RUNTIME_ROOT ROBODOJO_SOURCE OPENPI_SOURCE OPENPI_PYTHON ROBODOJO_PYTHON CHECKPOINT
    ROLLOUT_CASE_ID ROLLOUT_EXPERIMENT_ID ROLLOUT_ATTEMPT ROLLOUT_EVAL_MANIFEST ROLLOUT_EVAL_MANIFEST_SHA256
    RESULTS_ROOT POLICY_GPU SIM_GPU EXECUTE_STEPS POLICY_PORT SIM_PORT TASK_ROOT OPENPI_DATA_HOME
    RUNTIME_TASK BASE_TASK VARIANT LAYOUT_ID EVAL_SEED SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID
    ROBODOJO_TACTILE ROBODOJO_TACTILE_CALIBRATION'
export MANIFEST_KEYS
export $MANIFEST_KEYS  # Word splitting intended; names without a value stay out of the environment.
"$ROBODOJO_PYTHON" - "$ARCHIVE/launch_manifest.json" <<'PY'
import json, os, platform, sys
from datetime import datetime, timezone
values = {key: os.environ[key] for key in os.environ['MANIFEST_KEYS'].split() if key in os.environ}
json.dump(dict(schema='hybrid_rollout.robodojo.launch.v1', evaluation_method='pi05_only',
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

(
    cd "$SNAPSHOT"
    exec env CUDA_VISIBLE_DEVICES="$POLICY_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="$SNAPSHOT:$OPENPI_SOURCE/src:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$OPENPI_PYTHON" -m hybrid_rollout.robodojo.pi05_server.server \
        --checkpoint "$CHECKPOINT" --port "$POLICY_PORT" --identity-output "$ARCHIVE/policy_identity.json"
) > "$ARCHIVE/logs/policy.log" 2>&1 &
policy_pid=$!
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
until [[ -f "$ARCHIVE/policy_identity.json" ]] && grep -q '"event": "ready"' "$ARCHIVE/logs/sim.log"; do
    kill -0 "$policy_pid" 2>/dev/null || { echo "Policy exited: $ARCHIVE/logs/policy.log" >&2; exit 3; }
    kill -0 "$sim_pid" 2>/dev/null || { echo "Simulator exited: $ARCHIVE/logs/sim.log" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo 'Service startup timeout' >&2; exit 3; }
    sleep 2
done
(
    cd "$SNAPSHOT"
    exec env PYTHONPATH="$SNAPSHOT:$OPENPI_SOURCE/packages/openpi-client/src" \
        "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.pi05_only run \
        --output "$ARCHIVE/controller" --task "$RUNTIME_TASK" --checkpoint "$CHECKPOINT" \
        --sim-port "$SIM_PORT" --student-port "$POLICY_PORT" --seed "$LAYOUT_ID" \
        --execute-steps "$EXECUTE_STEPS"
) > "$ARCHIVE/logs/controller.log" 2>&1 &
controller_pid=$!
wait "$controller_pid"
controller_pid=''
[[ -f "$ARCHIVE/controller/result.json" ]] || { echo 'No rollout result' >&2; exit 4; }
# Let the simulator finalize sensors.mp4/evaluation_outcome.json after the controller disconnects.
for _ in {1..120}; do kill -0 "$sim_pid" 2>/dev/null || break; [[ -f "$ARCHIVE/sim/evaluation_outcome.json" ]] && break; sleep 1; done
cat "$ARCHIVE/sim/evaluation_outcome.json"
