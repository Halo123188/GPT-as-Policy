"""pi05-only RoboDojo baseline: no GPT/Codex review, native XPolicyLab chunk execution.

Mirrors XPolicyLab/policy/Pi_05/deploy.py: observe, infer one 50-step chunk, execute
every action of that chunk (one native take_action + observation per step), repeat
until the native episode ends. The simulator session, frozen case identity, scoring
and recorded artifacts are the same ones used by the hybrid and GPT-only methods.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .evaluation import case_identity, read_panel
from .io import require, write_json
from .robodojo_server.protocol import RPCClient

METHOD = 'pi05_only'
RPC_CHUNK_LIMIT = 15  # RoboDojoSession.chunk_step accepts 1..15 actions per request.


class Pi05OnlyRollout:
    def __init__(self, output, task, student, *, sim_port=19113, seed=0,
                 execute_steps=50, rpc_factory=RPCClient):
        require(1 <= execute_steps <= 50, 'execute_steps must be in 1..50')
        self.output = Path(output).resolve()
        self.task, self.student, self.seed = task, student, seed
        self.sim_port, self.execute_steps, self.rpc_factory = sim_port, execute_steps, rpc_factory
        self.sim = self.episode = None
        self.tick, self.history, self.done = 0, [], False
        self.started = time.monotonic()

    def _rpc(self, op, **kwargs):
        return self.sim.request(op, episode_id=self.episode, step_id=self.tick, **kwargs)

    def start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=900)
        self.meta = self.sim.request('metadata')
        require(self.meta['task'] == self.task, 'Launched RoboDojo task differs from requested task')
        identity = self.student.metadata['checkpoint_sha256']
        config = self.student.metadata['config']
        version = f'OpenPI-JAX/{config}/{identity[:16]}'
        reset = self.sim.request('reset', seed=self.seed, source='student', policy_version=version)
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        self._rpc('begin_combination', evaluation_method=METHOD, teacher_model=None,
                  student_policy_version=version, student_policy_sha256=identity)
        run = dict(schema='robodojo_rollout.run.v1', evaluation_method=METHOD, teacher=None,
            task=self.task, instruction=reset.get('instruction'), seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'),
            config=config, checkpoint=self.student.metadata['checkpoint'],
            student_backend='OpenPI/JAX', student_identity_sha256=identity,
            action_horizon=self.student.metadata['action_horizon'], action_dim=self.student.metadata['action_dim'],
            executed_steps_per_chunk=self.execute_steps, execution_reference='XPolicyLab/policy/Pi_05/deploy.py',
            initial_state_hash=reset.get('initial_state_hash'), reset_metadata=reset.get('metadata'),
            max_episode_steps=self.meta['max_episode_steps'], control_dt=self.meta['control_dt'],
            no_rollback=True)
        write_json(self.output/'run.json', run)
        return reset

    def step_chunk(self):
        require(not self.done, 'Episode already finished')
        decision = len(self.history)
        observation = self._rpc('teacher_observation')  # Cached post-ACK observation.
        proposals = self.output/'proposals'
        proposals.mkdir(exist_ok=True)
        actions, prediction = self.student.infer(observation, proposals/f'{decision:04d}.npz')
        chunk = np.asarray(actions[:self.execute_steps], np.float32)
        start_tick, rows = self.tick, []
        for begin in range(0, len(chunk), RPC_CHUNK_LIMIT):
            result = self._rpc('chunk_step', actions=chunk[begin:begin+RPC_CHUNK_LIMIT],
                               student_prediction_id=prediction['prediction_id'])
            rows.extend(result['steps'])
            self.tick = result['step_id']
            if any(r['terminated'] or r['truncated'] for r in result['steps']):
                self.done = True
                break
        require(rows and all(r['valid'] for r in rows), 'Simulator reported an invalid control step')
        last = rows[-1]
        self.history.append(dict(decision=decision, start_tick=start_tick, end_tick=self.tick,
            prediction_id=prediction['prediction_id'], inference_index=prediction['inference_index'],
            inference_seconds=prediction['inference_seconds'], executed_steps=len(rows),
            terminal=self.done, native_success=bool(last['success'])))
        write_json(self.output/'history.json', self.history)
        if not self.done:
            require(self.tick < self.meta['max_episode_steps'],
                    'Native step limit passed without an episode end')
        return self.history[-1]

    def run_episode(self):
        self.start()
        while not self.done:
            record = self.step_chunk()
            print(json.dumps(dict(event='chunk', **record)), flush=True)
        return self.finish('terminal')

    def finish(self, reason):
        final = self._rpc('finish_pilot', reason=reason)
        final.update(evaluation_method=METHOD, decisions=len(self.history),
            pi05_inference_calls=len(self.history), student_steps=self.tick,
            complete=bool(final['terminated'] or final['truncated']),
            wall_seconds=time.monotonic()-self.started, history_path=str(self.output/'history.json'),
            trajectory_path=str(self.output.parent/'sim'))
        write_json(self.output/'result.json', final)
        self.done = True
        return final

    def close(self):
        if self.sim is not None:
            self.sim.close()


def scope_case_ids(scope_path):
    scope = json.loads(Path(scope_path).read_text())
    ids = [case['case_id'] for case in scope.get('cases') or []]
    require(len(ids) == len(set(ids)) == scope['target_total_cases'], 'Scope has no confirmed case list')
    return ids


def write_case_file(manifest, case_id, output):
    """Same {identity, case} contract that batch.py hands to the simulator server."""
    panel = read_panel(manifest)
    matches = [c for c in panel['cases'] if c['case_id'] == case_id]
    require(len(matches) == 1, f'Unknown frozen case: {case_id}')
    case = matches[0]
    write_json(output, dict(identity=case_identity(panel, case), case=case))
    return panel, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('cases', help='Print the frozen case IDs of a campaign scope')
    listing.add_argument('--scope', type=Path, required=True)
    prepare = commands.add_parser('prepare-case', help='Write a case file; print shell assignments')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--case-id', required=True)
    prepare.add_argument('--output', type=Path, required=True)
    parser_run = commands.add_parser('run', help='Run one pi05-only episode')
    parser_run.add_argument('--output', type=Path, required=True)
    parser_run.add_argument('--task', required=True, help='Runtime task, e.g. fold_clothes_random')
    parser_run.add_argument('--checkpoint', type=Path, required=True)
    parser_run.add_argument('--sim-port', type=int, default=19113)
    parser_run.add_argument('--student-port', type=int, default=18830)
    parser_run.add_argument('--seed', type=int, default=0, help='Native layout_id passed to reset')
    parser_run.add_argument('--execute-steps', type=int, default=50,
                            help='Actions executed per 50-step chunk; native XPolicyLab executes all 50')
    args = parser.parse_args()
    if args.command == 'cases':
        print('\n'.join(scope_case_ids(args.scope)))
        return
    if args.command == 'prepare-case':
        panel, case = write_case_file(args.manifest, args.case_id, args.output)
        for key, value in (('RUNTIME_TASK', case['runtime_task']), ('BASE_TASK', case['task']),
                           ('VARIANT', case['variant']), ('LAYOUT_ID', case['layout_id']),
                           ('EVAL_SEED', case['eval_seed']), ('PANEL_SHA256', panel['panel_sha256'])):
            print(f'{key}={value}')
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    student = rollout = None
    try:
        from .pi05_server.client import Pi05Client
        student = Pi05Client(args.student_port, args.checkpoint)
        rollout = Pi05OnlyRollout(args.output, args.task, student, sim_port=args.sim_port,
                                  seed=args.seed, execute_steps=args.execute_steps)
        result = rollout.run_episode()
        print(json.dumps(dict(event='result', success=result['success'], step_id=result['step_id'],
                              decisions=result['decisions'])), flush=True)
    except BaseException:
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(), evaluation_method=METHOD,
            episode_id=rollout.episode if rollout else None, step_id=rollout.tick if rollout else 0,
            completed=False))
        if rollout is not None and rollout.episode is not None and not rollout.done:
            try:
                rollout.finish('controller_error')
            except Exception:
                pass  # Never retry an action with an uncertain ACK.
        raise
    finally:
        if rollout is not None:
            rollout.close()
        if student is not None:
            student.close()


if __name__ == '__main__':
    main()
