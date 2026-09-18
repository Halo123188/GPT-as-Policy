import json
from pathlib import Path

import numpy as np
import pytest

from .pi05_only import Pi05OnlyRollout, scope_case_ids, write_case_file
from .test_evaluation import fake_panel_file

PANELS = Path(__file__).parent/'eval_panels'


class Student:
    metadata = dict(checkpoint_sha256='a'*64, config='pi05_base_aloha_full_sim_arx-x5_seed_0',
                    checkpoint='/ckpt', action_horizon=50, action_dim=14)

    def __init__(self):
        self.calls = 0

    def infer(self, observation, output):
        self.calls += 1
        actions = np.full((50, 14), self.calls, np.float32)
        actions[:, [6, 13]] = .5
        return actions, dict(prediction_id=f'p{self.calls}', inference_index=self.calls-1, inference_seconds=.1)


class Sim:
    """Native session stand-in: at most 15 actions per chunk_step, ends at step_lim."""
    def __init__(self, *args, step_lim=120, succeed_at=None, **kwargs):
        self.tick, self.calls, self.step_lim, self.succeed_at = 0, [], step_lim, succeed_at
        self.chunk_sizes, self.executed = [], []

    def request(self, op, **args):
        self.calls.append(op)
        if op == 'metadata':
            return dict(task='build_tower', max_episode_steps=self.step_lim, control_dt=.04)
        if op == 'reset':
            assert args['seed'] == 3 and args['source'] == 'student'
            return dict(episode_id='episode', step_id=0, instruction='Build a tower.',
                        metadata=dict(evaluation_case=dict(case_id='build_tower__standard__g0__l3')))
        assert args.pop('episode_id') == 'episode' and args.pop('step_id') == self.tick
        if op == 'teacher_observation':
            return dict(states=np.zeros(14, np.float32), instruction='Build a tower.')
        if op == 'chunk_step':
            assert 1 <= len(args['actions']) <= 15
            self.chunk_sizes.append(len(args['actions']))
            rows = []
            for action in args['actions']:
                self.tick += 1
                self.executed.append(action)
                success = self.succeed_at is not None and self.tick >= self.succeed_at
                ended = success or self.tick >= self.step_lim
                rows.append(dict(valid=True, executed_action=action, terminated=success,
                                 truncated=ended and not success, success=success))
                if ended:
                    break
            return dict(step_id=self.tick, steps=rows)
        if op == 'finish_pilot':
            return dict(step_id=self.tick, success=self.tick == self.succeed_at, terminated=self.tick == self.succeed_at,
                        truncated=self.tick >= self.step_lim, reason=args['reason'])
        return {}


def test_executes_full_native_chunks_until_truncation(tmp_path):
    sims = []
    rollout = Pi05OnlyRollout(tmp_path, 'build_tower', Student(), seed=3,
                              rpc_factory=lambda *a, **k: sims.append(Sim()) or sims[-1])
    result = rollout.run_episode()
    sim = sims[0]
    # Two full 50-step chunks plus 20 of the third, split into <=15-action RPCs.
    assert sim.tick == 120 and rollout.student.calls == 3
    assert sim.chunk_sizes == [15, 15, 15, 5]*2 + [15, 15]
    assert [float(a[0]) for a in sim.executed[48:52]] == [1, 1, 2, 2]
    assert result['complete'] and not result['success'] and result['decisions'] == 3
    assert [h['executed_steps'] for h in json.loads((tmp_path/'history.json').read_text())] == [50, 50, 20]
    assert json.loads((tmp_path/'run.json').read_text())['evaluation_method'] == 'pi05_only'
    assert sim.calls.count('teacher_observation') == 3 and 'fk_preview' not in sim.calls


def test_stops_at_native_success_mid_chunk(tmp_path):
    rollout = Pi05OnlyRollout(tmp_path, 'build_tower', Student(), seed=3,
                              rpc_factory=lambda *a, **k: Sim(succeed_at=57))
    result = rollout.run_episode()
    assert result['success'] and rollout.tick == 57 and rollout.student.calls == 2
    assert json.loads((tmp_path/'result.json').read_text())['step_id'] == 57


def test_partial_chunk_execution_option(tmp_path):
    rollout = Pi05OnlyRollout(tmp_path, 'build_tower', Student(), seed=3, execute_steps=10,
                              rpc_factory=lambda *a, **k: Sim(step_lim=25))
    rollout.run_episode()
    assert rollout.student.calls == 3 and [h['executed_steps'] for h in rollout.history] == [10, 10, 5]
    with pytest.raises(ValueError):
        Pi05OnlyRollout(tmp_path, 'build_tower', Student(), execute_steps=51)


def test_rejects_wrong_task(tmp_path):
    rollout = Pi05OnlyRollout(tmp_path, 'make_kong', Student(), seed=3, rpc_factory=Sim)
    with pytest.raises(ValueError):
        rollout.start()


def test_scope_matches_public_fifty_cases():
    ids = scope_case_ids(PANELS/'robodojo_panel50_scope_v2.json')
    public = json.loads((PANELS.parents[2]/'public_results/evaluation_cases.json').read_text())
    assert sorted(ids) == sorted({c['case_id'] for c in public['cases']})
    assert len({i.split('__')[0] for i in ids}) == 10


def test_case_file_matches_batch_contract(tmp_path):
    _, manifest, panel = fake_panel_file(tmp_path)
    case = panel['cases'][7]
    write_case_file(manifest, case['case_id'], tmp_path/'case.json')
    written = json.loads((tmp_path/'case.json').read_text())
    assert written['case'] == case and written['identity']['panel_sha256'] == panel['panel_sha256']
    assert written['identity']['layout_id'] == case['layout_id']
    with pytest.raises(ValueError):
        write_case_file(manifest, 'missing__standard__g0__l0', tmp_path/'other.json')


def test_summary_counts_invalid_layouts_as_final_non_successes(tmp_path):
    from .pi05_only_summary import summarize
    ids = scope_case_ids(PANELS/'robodojo_panel50_scope_v2.json')
    def outcome(case_id, attempt, **values):
        path = tmp_path/case_id/f'attempt_{attempt}'/'sim'
        path.mkdir(parents=True)
        (path/'evaluation_outcome.json').write_text(json.dumps(values))
    outcome(ids[0], 0, status='native_completed', valid_for_success_rate=True, native_success=True, native_score=1.0)
    outcome(ids[1], 0, status='incomplete', valid_for_success_rate=False)  # infrastructure attempt
    outcome(ids[1], 1, status='native_completed', valid_for_success_rate=True, native_success=False, native_score=.5)
    outcome(ids[2], 0, status='invalid_native_layout', valid_for_success_rate=False)
    summary = summarize(tmp_path, PANELS/'robodojo_panel50_scope_v2.json')
    assert summary['complete_cases'] == 2 and summary['invalid_cases'] == 1 and len(summary['missing']) == 47
    assert summary['success_rate_percent_over_planned'] == 2.0 and summary['mean_score_100_over_complete'] == 75.0
    assert [c['attempt'] for c in summary['cases']] == ['attempt_0', 'attempt_1']
    assert summary['invalid'][0]['case_id'] == ids[2]
