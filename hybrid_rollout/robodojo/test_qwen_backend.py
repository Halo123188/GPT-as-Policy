"""Offline checks for the Qwen chat-completions backend; no model, network or physics.

The fake server reads `next_call` out of the conversation exactly as a real
model must, so these tests fail if the loop ever stops telling the model what to
call next.
"""
import copy
import json

import pytest

from .qwen_backend import settings as qwen_settings
from .qwen_backend.policy import ModelTransportError, QwenPolicy, _openai_tools
from .robodojo_server.client import RoboDojoTools
from .skill.run import tool_specs
from .test_contract import POSE, Sim, Student


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _latest(messages, key):
    """Newest JSON tool payload carrying `key`, as a model would have to find it."""
    for message in reversed(messages):
        content = message.get('content')
        if not isinstance(content, str):
            continue
        try:
            packet = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(packet, dict) and packet.get(key) is not None:
            return packet[key]
    return None


def direct_decision(request_id):
    """gpt_only authorises bounded EEF targets only; no gate, edit or student mode."""
    return dict(request_id=request_id, mode='eef', steps=2, reason='Synthetic backend test.',
        target={arm: copy.deepcopy(POSE) for arm in ('left', 'right')})


def decision(request_id, mode='student'):
    return dict(request_id=request_id, mode=mode, steps=2, reason='Synthetic backend test.',
        target={arm: copy.deepcopy(POSE) for arm in ('left', 'right')},
        edit={arm: dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='keep')
              for arm in ('left', 'right')},
        assessment=dict(
            task_progress=dict(verified_completed=[], currently_attempting='Arrange digits',
                               remaining=['Arrange']),
            current_subgoal='Arrange digits', execution_status='not_started',
            execution_evidence='Synthetic image.', expected_next_intent='Arrange',
            predicted_next_intent='Arrange', intent_status='aligned',
            intent_evidence='Synthetic FK.'))


class FakeModel:
    """Scripted server that answers with the tool call the host just requested."""

    def __init__(self, *, corrupt_call=None, fail_times=0, direct=False, always_invalid=False,
                 truncate_calls=()):
        self.corrupt_call = corrupt_call
        self.fail_times = fail_times
        self.direct = direct
        self.always_invalid = always_invalid
        self.truncate_calls = set(truncate_calls)
        self.nudges = []
        self.requests = []
        self.calls = 0

    def __call__(self, request, timeout=None):
        payload = json.loads(request.data.decode())
        self.requests.append(payload)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError('synthetic transport failure')
        messages = payload['messages']
        self.calls += 1
        text = messages[-1]['content']
        if isinstance(text, str) and 'Do not call any tool' in text:
            return FakeResponse(dict(choices=[dict(message=dict(
                role='assistant', content='Episode terminated natively; artifacts saved.'))]))
        if self.calls in self.truncate_calls:
            # Whole budget spent on reasoning: no content, no tool call.
            self.nudges.append(text if isinstance(text, str) else '')
            return FakeResponse(dict(choices=[dict(
                message=dict(role='assistant', content=''), finish_reason='length')],
                usage=dict(prompt_tokens=10, completion_tokens=8192, total_tokens=8202)))
        next_call = _latest(messages, 'next_call')
        if next_call is None:
            # Opening user turn carries next_call as trailing JSON text.
            next_call = json.loads(text[text.index('{'):])
        arguments = {k: v for k, v in next_call.items() if k != 'tool'}
        if 'response' in arguments:
            request_id = _latest(messages, 'request_id')
            if self.always_invalid:
                arguments['response'] = decision('stale-request-id')
            elif self.direct:
                arguments['response'] = direct_decision(request_id)
            else:
                arguments['response'] = decision(request_id)
        encoded = json.dumps(arguments)
        if self.corrupt_call == self.calls:
            encoded = encoded[:-3] + 'not json'
        return FakeResponse(dict(choices=[dict(message=dict(role='assistant', content='',
            tool_calls=[dict(id=f'call_{self.calls}', type='function',
                             function=dict(name=next_call['tool'], arguments=encoded))]))],
            usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)))


def policy(tmp_path, model, *, method='pi05_plus_gpt'):
    return QwenPolicy(tmp_path/'agent_workspace', method=method,
                      base_url='http://127.0.0.1:1/v1', opener=model)


def controller(tmp_path, name='controller'):
    """The entry point creates the rollout root before the handlers run."""
    root = tmp_path/name
    root.mkdir(parents=True)
    return root


def test_tool_schemas_are_the_codex_schemas_unchanged():
    for method in ('pi05_plus_gpt', 'gpt_only'):
        specs = tool_specs(method)
        wire = _openai_tools(specs)
        assert [w['function']['name'] for w in wire] == [s['name'] for s in specs]
        for spec, item in zip(specs, wire):
            assert item['function']['parameters'] is spec['inputSchema']
            assert item['function']['description'] == spec['description']


def test_hybrid_episode_runs_to_native_termination(tmp_path):
    model = FakeModel()
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256,
                            teacher=dict(name='qwen_tools', model='Qwen/Qwen3.8-27B',
                                         provider='vllm_openai_compatible', effort='xhigh'))
    worker.run(rollout)
    assert rollout.phase == 'done'
    assert (tmp_path/'controller/result.json').is_file()
    # The recorded identity follows the backend, not the pinned Codex settings.
    assert rollout.run['teacher_model'] == 'Qwen/Qwen3.8-27B'
    assert rollout.run['teacher_reasoning_effort'] == 'xhigh'
    worker_record = json.loads((tmp_path/'agent_workspace/worker.json').read_text())
    assert worker_record['backend'] == 'qwen_vllm'
    assert worker_record['reasoning_effort'] == 'xhigh'
    assert worker_record['sampling']['temperature'] == 1.0
    assert worker_record['tools'] == ['robodojo_start', 'pi05_infer', 'robodojo_execute']
    assert (tmp_path/'agent_workspace/call_0000_result.json').is_file()


def test_direct_episode_uses_two_tools_and_no_student(tmp_path):
    model = FakeModel(direct=True)
    worker = policy(tmp_path, model, method='gpt_only')
    from .robodojo_server.gpt_only_client import GPTOnlyTools
    rollout = GPTOnlyTools(controller(tmp_path), 'arrange_largest_number', rpc_factory=Sim,
                           prompt_sha256=worker.prompt_sha256,
                           teacher=dict(name='qwen_tools', model='Qwen/Qwen3.8-27B',
                                        provider='vllm_openai_compatible', effort='xhigh'))
    worker.run(rollout)
    assert rollout.phase == 'done'
    assert rollout.run['pi05_enabled'] is False
    assert json.loads((tmp_path/'agent_workspace/tools.json').read_text())[1]['name'] == 'robodojo_act'


def test_malformed_tool_arguments_are_recoverable(tmp_path):
    model = FakeModel(corrupt_call=2)
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    rejected = json.loads((tmp_path/'agent_workspace/call_0001_result.json').read_text())
    assert rejected['error_type'] == 'recoverable_tool_input'
    assert rejected['no_execution'] is True and rejected['retryable'] is True
    assert rejected['next_call']['tool'] == 'pi05_infer'
    # The rejection did not reset the episode or lose the rollout.
    assert rollout.phase == 'done'


def test_transport_failures_retry_then_give_up(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_settings, 'RETRY_DELAYS', (0, 0))
    model = FakeModel(fail_times=2)
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)  # Two failures, then the third attempt succeeds.
    assert rollout.phase == 'done'

    monkeypatch.setattr(qwen_settings, 'RETRY_DELAYS', (0,))
    dead = policy(tmp_path/'dead', FakeModel(fail_times=99))
    other = RoboDojoTools(controller(tmp_path, 'dead_controller'), 'arrange_largest_number', Student(),
                          rpc_factory=Sim, prompt_sha256=dead.prompt_sha256)
    with pytest.raises(ModelTransportError):
        dead.run(other)


def test_old_cycles_release_images_but_keep_tool_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_settings, 'FULL_DETAIL_CYCLES', 1)
    model = FakeModel()
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    payloads = [p for p in model.requests if len(p['messages']) > 6]
    assert payloads, 'expected the conversation to grow past the detail window'
    wire = payloads[-1]['messages']
    images = [m for m in wire if isinstance(m.get('content'), list)
              and any(p.get('type') == 'image_url' for p in m['content'])]
    assert len(images) <= 1, 'stale observations must not keep their attachments'
    # Every tool reply still answers exactly one assistant tool call.
    ids = [c['id'] for m in wire if m.get('role') == 'assistant' for c in m.get('tool_calls', [])]
    replies = [m['tool_call_id'] for m in wire if m.get('role') == 'tool']
    assert replies == ids[:len(replies)]
    assert all(isinstance(m['content'], str) for m in wire if m.get('role') == 'tool')


def test_prompt_keeps_skill_and_states_runtime_limits(tmp_path):
    worker = policy(tmp_path, FakeModel())
    prompt = (tmp_path/'agent_workspace/PROMPT.md').read_text()
    assert 'RoboDojo hybrid policy agent' in prompt
    assert 'Unchanged baseline gate prompt' in prompt
    assert 'there is no shell' in prompt
    assert worker.prompt_sha256 and len(worker.prompt_sha256) == 64


def test_endless_rejections_abandon_the_episode(tmp_path, monkeypatch):
    """An unattended local server must not hold a GPU re-sending one bad action."""
    monkeypatch.setattr(qwen_settings, 'MAX_CONSECUTIVE_REJECTIONS', 5)
    worker = policy(tmp_path, FakeModel(always_invalid=True))
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    with pytest.raises(RuntimeError, match='consecutive rejected tool calls'):
        worker.run(rollout)
    assert rollout.phase != 'done'


def test_long_episodes_drop_whole_cycles_and_stay_bounded(tmp_path, monkeypatch):
    """Compaction must bound the wire size, not merely shrink each message."""
    monkeypatch.setattr(qwen_settings, 'MAX_RETAINED_CYCLES', 2)
    monkeypatch.setattr(qwen_settings, 'FULL_DETAIL_CYCLES', 1)
    worker = policy(tmp_path, FakeModel())
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    # Simulate a long episode by replaying the recorded turns far into the future.
    worker.cycle += 50
    wire = worker._wire_messages()
    assert wire[0]['role'] == 'system' and wire[1]['role'] == 'user'
    assert any('no longer in context' in m['content'] for m in wire
               if isinstance(m.get('content'), str)), 'elision must be announced'
    # Every surviving tool reply still answers a visible tool call.
    ids = {c['id'] for m in wire if m.get('role') == 'assistant' for c in m.get('tool_calls', [])}
    assert all(m['tool_call_id'] in ids for m in wire if m.get('role') == 'tool')
    assert len(wire) < len(worker.messages)


def test_compaction_does_not_invent_elision_notices(tmp_path, monkeypatch):
    """Releasing a message's images must not be counted as a dropped cycle."""
    monkeypatch.setattr(qwen_settings, 'FULL_DETAIL_CYCLES', 1)
    monkeypatch.setattr(qwen_settings, 'MAX_RETAINED_CYCLES', 10_000)
    worker = policy(tmp_path, FakeModel())
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    wire = worker._wire_messages()
    notices = [m for m in wire if isinstance(m.get('content'), str)
               and 'no longer in context' in m['content']]
    assert notices == [], 'nothing aged out, so no cycle-elision notice belongs here'
    assert any('released from context' in m['content'] for m in wire
               if isinstance(m.get('content'), str)), 'image release should still be noted'


def test_replayed_tool_arguments_stay_strings(tmp_path):
    """The request schema validates `arguments` as a string; an object is a 400."""
    from .qwen_backend.policy import _replayable_tool_calls
    replay = _replayable_tool_calls([
        dict(id='a', type='function', function=dict(name='t', arguments='{"x": 1}')),
        dict(id='b', type='function', function=dict(name='t', arguments={'x': 1})),
        dict(id='c', type='function', function=dict(name='t', arguments='not json')),
    ])
    assert [c['function']['arguments'] for c in replay] == ['{"x": 1}', '{"x": 1}', 'not json']

    model = FakeModel()
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    for payload in model.requests:
        for message in payload['messages']:
            for call in message.get('tool_calls', []):
                assert isinstance(call['function']['arguments'], str)


def test_truncated_reply_is_nudged_not_blindly_repeated(tmp_path):
    """A length-truncated turn must be told it was cut off, then recover."""
    model = FakeModel(truncate_calls=(2,))
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    worker.run(rollout)
    assert rollout.phase == 'done', 'episode should recover after a truncated turn'
    followup = [m for m in worker.messages if m.get('role') == 'user'
                and isinstance(m.get('content'), str) and 'cut off' in m['content']]
    assert followup, 'the model must be told its reply was truncated'
    assert 'nothing was executed' in followup[0]['content']


def test_repeated_truncation_fails_fast(tmp_path):
    model = FakeModel(truncate_calls=(2, 3, 4, 5, 6))
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    with pytest.raises(RuntimeError, match='exhausted its output budget'):
        worker.run(rollout)


def test_simulator_idle_timeout_is_configurable():
    """A slow local model must not be cut off by the transport keep-alive."""
    import importlib
    from .robodojo_server import rpc
    assert rpc.IDLE_TIMEOUT_SECONDS == 900, 'the Codex default must be unchanged'
    import os
    os.environ['ROBODOJO_RPC_TIMEOUT_SECONDS'] = '7200'
    try:
        reloaded = importlib.reload(rpc)
        assert reloaded.IDLE_TIMEOUT_SECONDS == 7200
    finally:
        del os.environ['ROBODOJO_RPC_TIMEOUT_SECONDS']
        importlib.reload(rpc)


def test_rejections_do_not_age_the_context_window(tmp_path, monkeypatch):
    """A rejected call must not push the live proposal out of full detail.

    Otherwise the model loses sight of the request_id it is being rejected for
    getting wrong, and the rejection loop becomes self-sustaining.
    """
    monkeypatch.setattr(qwen_settings, 'FULL_DETAIL_CYCLES', 2)
    model = FakeModel(corrupt_call=2)
    worker = policy(tmp_path, model)
    rollout = RoboDojoTools(controller(tmp_path), 'arrange_largest_number', Student(),
                            rpc_factory=Sim, prompt_sha256=worker.prompt_sha256)
    before = worker.cycle
    worker.run(rollout)
    accepted = sum(1 for m in worker.messages if m.get('role') == 'tool'
                   and '"error_type"' not in (m.get('content') or ''))
    assert worker.cycle - before == accepted, 'cycle must count accepted calls only'
    assert rollout.phase == 'done'


def test_identifier_hint_only_after_repeated_stale_rejections():
    """The hint restates the id only when repetition shows the model mis-copied it."""
    from .qwen_backend.policy import _identifier_hint
    rid = 'a' * 28 + 'b807'
    stale = dict(error_type='recoverable_tool_input', request_id=rid,
                 error='Response is stale or belongs to another observation')
    assert _identifier_hint(stale, 1) is None, 'one failure is not yet a pattern'
    assert _identifier_hint(stale, 2) is None
    hint = _identifier_hint(stale, 3)
    assert hint and rid in hint and '32 characters' in hint
    assert 'in groups of four' in hint
    # Unrelated rejections must not trigger it.
    other = dict(error_type='recoverable_tool_input', request_id=rid,
                 error='Recovery EEF target exceeds 5 cm')
    assert _identifier_hint(other, 9) is None
    assert _identifier_hint(dict(step_id=3), 9) is None
