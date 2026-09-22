"""Drive one rollout with a locally served Qwen VLM through the same services.

The Codex backend hands the model three (hybrid) or two (direct) blocking
rollout services and lets it act until the simulator terminates natively. This
module keeps that contract byte-for-byte -- identical prompt sources, identical
tool schemas, identical host-side handlers -- and replaces only the agent
driver: instead of the Codex app-server it speaks OpenAI-compatible
chat-completions to a vLLM server holding the open-weights model.

Only the standard library is used for transport so the controller can keep
running inside the Isaac Sim interpreter without new dependencies.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from ..io import InputError, write_json
from ..method import evaluation_method
from ..prompt_context import CONTEXT_VERSION
from ..skill.image_preview import image_max_edge, prepare_image
from ..skill.run import CONTROLLER_VERSION, rejected_input, tool_specs
from ..skill.workspace import prepare_workspace
from . import settings

SKILL_ROOT = Path(__file__).resolve().parents[1]/'skill'

# The Codex agent also owns shell, file and image-viewing tools. A plain
# chat-completions server has none of them, so the difference is stated to the
# model instead of leaving it waiting for a capability that will never answer.
RUNTIME_DIFFERENCES = """

# Runtime differences for this deployment

You are served by a local open-weights model through a chat API. The rollout
service tools listed below are your ONLY tools: there is no shell, no file
browser, no separate image viewer and no code execution in this deployment.

- Do not announce or wait for shell commands, scripts, crops or file reads.
- Every observation's images are attached directly to the conversation right
  after the tool result that produced them. `images[].path` still names the
  full-resolution file on disk for the record, but you cannot open it; judge
  from the attached views.
- Keep working memory in your own replies rather than in NOTES.md.
- Reply with a tool call whenever the episode is unfinished. Use the exact
  `next_call` arguments supplied in the most recent tool result.
"""


def _openai_tools(specs):
    """Codex dynamicTools -> OpenAI function-tool wire format (same schemas)."""
    return [dict(type='function', function=dict(
        name=spec['name'], description=spec['description'],
        parameters=spec['inputSchema'])) for spec in specs]


def _is_context_overflow(body):
    text = (body or '').lower()
    return ('context' in text and 'length' in text) or 'maximum context' in text \
        or 'longer than the maximum' in text or 'reduce the length' in text


def _replayable_tool_calls(tool_calls):
    """Echo tool calls back in the shape the request schema accepts.

    `arguments` must be a JSON *string* here: that is what the OpenAI request
    schema validates, and the server parses it into the mapping its chat
    template needs. Sending the decoded object instead is rejected outright.
    Unparseable arguments are kept verbatim so the model sees what it sent.
    """
    replay = []
    for call in tool_calls:
        function = dict(call.get('function') or {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            function['arguments'] = json.dumps(arguments if arguments is not None else {})
        replay.append(dict(call, function=function))
    return replay


def _decode_object_arguments(arguments, spec):
    """Accept an object-valued parameter that arrived as a JSON string.

    Qwen's XML tool format carries each parameter as text, so a nested object
    such as `response` can come back as a JSON string rather than a mapping.
    Decoding it here is a wire-format concession only: the decoded value still
    faces the unchanged validators, and anything that is not a JSON object is
    left exactly as sent so the model sees its own mistake.
    """
    if not isinstance(arguments, dict):
        return arguments
    properties = (spec or {}).get('properties', {})
    decoded = dict(arguments)
    for name, schema in properties.items():
        value = decoded.get(name)
        if schema.get('type') != 'object' or not isinstance(value, str):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            decoded[name] = parsed
    return decoded


def _identifier_hint(packet, consecutive):
    """Spell out the expected request_id after repeated identical rejections.

    The validator requires the response to echo the observation's 32-character
    request_id exactly, and a model that mis-copies one character reproduces the
    same typo on every retry -- the rejection already contains the right value,
    so repeating it changes nothing. After a few failures, restate the id in
    chunks and make the copy instruction explicit. This is recovery feedback
    only: the validator, the action contract and scoring are untouched.
    """
    if consecutive < 3 or packet.get('error_type') != 'recoverable_tool_input':
        return None
    if 'stale or belongs to another observation' not in str(packet.get('error', '')):
        return None
    identifier = packet.get('request_id')
    if not isinstance(identifier, str) or not identifier:
        return None
    chunks = ' '.join(identifier[i:i + 4] for i in range(0, len(identifier), 4))
    return ('Your request_id does not match this observation, so nothing executed. '
            f'The required value is exactly {len(identifier)} characters:\n'
            f'  {identifier}\n'
            f'  (in groups of four: {chunks})\n'
            'Copy it character for character into response.request_id. Do not '
            'shorten, reformat or retype it from memory.')


def _elision_notice(count):
    return dict(role='user', content=(
        f'[{count} earlier decision cycles are no longer in context. The simulator '
        'state, step_id and counters in the most recent tool result remain '
        'authoritative; nothing was reset or replayed.]'))


def _truncate(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + f'\n...[{len(text) - limit} characters elided from an earlier cycle]'


class ModelTransportError(RuntimeError):
    """Server/socket failure. Never a validation reject and never a physics error."""


class QwenPolicy:
    def __init__(self, workspace, *, method='pi05_plus_gpt', timeout=None,
                 base_url=None, model=None, opener=urllib.request.urlopen):
        self.method = evaluation_method(method)
        direct = self.method == 'gpt_only'
        skill_root = SKILL_ROOT.parent/'robodojo-gpt-only-rollout' if direct else SKILL_ROOT
        self.model = model or settings.MODEL
        self.base_url = (base_url or settings.BASE_URL).rstrip('/')
        self.timeout = float(timeout if timeout is not None else settings.REQUEST_TIMEOUT)
        self.opener = opener
        self.image_max_edge = image_max_edge(os.environ.get(
            'ROLLOUT_QWEN_IMAGE_MAX_EDGE', str(settings.DEFAULT_IMAGE_MAX_EDGE)))
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            controller_version=CONTROLLER_VERSION, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, validation_unchanged=True,
            same_thread=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False))
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        skill = (skill_root/'SKILL.md').read_text()
        gate = '' if direct else (skill_root/'gate_prompt.md').read_text()
        context = (skill_root/'context/teacher_context.md').read_text()
        self.prompt = (skill + ('' if direct else '\n\n# Unchanged baseline gate prompt\n\n' + gate)
                       + '\n\n' + context + RUNTIME_DIFFERENCES)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        if not direct:
            (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
        write_json(self.workspace/'tools.json', specs)
        self.tools = _openai_tools(specs)
        self.schemas = {spec['name']: spec['inputSchema'] for spec in specs}
        self.messages = []
        self.cycle = 0
        self.usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, requests=0)
        write_json(self.workspace/'worker.json', dict(
            model=self.model, model_provider=settings.PROVIDER,
            controller_version=CONTROLLER_VERSION, evaluation_method=self.method,
            backend='qwen_vllm', base_url=self.base_url,
            reasoning_effort=settings.REASONING_EFFORT,
            context_version=CONTEXT_VERSION, prompt_sha256=self.prompt_sha256,
            no_rollback=True, tools=[s['name'] for s in specs],
            service_tools_are_additive=False,
            native_tools='None; chat-completions deployment has no shell or image tools',
            agent_workspace=str(self.agent_workspace),
            sampling=dict(temperature=settings.TEMPERATURE, top_p=settings.TOP_P,
                          top_k=settings.TOP_K, min_p=settings.MIN_P,
                          max_tokens=settings.MAX_OUTPUT_TOKENS),
            context_window=dict(full_detail_cycles=settings.FULL_DETAIL_CYCLES,
                                compacted_text_chars=settings.COMPACTED_TEXT_CHARS),
            qwen_image_max_edge=self.image_max_edge, policy_images_resized=False,
            baseline_full_conversation_available=False))

    # ----- transport -------------------------------------------------------
    def _post(self, payload):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + '/chat/completions', data=body,
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + settings.API_KEY})
        with self.opener(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def _complete(self, rollout):
        detail_cycles = settings.FULL_DETAIL_CYCLES
        retained = settings.MAX_RETAINED_CYCLES
        last = None
        for attempt, delay in enumerate((0,) + settings.RETRY_DELAYS):
            if delay:
                print(json.dumps(dict(event='qwen_transport_retry', attempt=attempt,
                    delay_seconds=delay, step_id=rollout.tick, error=str(last))), flush=True)
                time.sleep(delay)
            payload = dict(model=self.model, messages=self._wire_messages(detail_cycles, retained),
                           tools=self.tools, tool_choice='auto',
                           temperature=settings.TEMPERATURE, top_p=settings.TOP_P,
                           top_k=settings.TOP_K, min_p=settings.MIN_P,
                           max_tokens=settings.MAX_OUTPUT_TOKENS,
                           chat_template_kwargs=dict(reasoning_effort=settings.REASONING_EFFORT))
            try:
                result = self._post(payload)
            except urllib.error.HTTPError as error:
                body = ''
                try:
                    body = error.read().decode()[:2000]
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
                last = RuntimeError(f'HTTP {error.code}: {body}')
                # A prompt that no longer fits is fixed by dropping older detail,
                # not by sending the identical request again.
                if error.code == 400 and _is_context_overflow(body):
                    # Drop verbatim detail first (a stale proposal is ~18k tokens),
                    # then start releasing whole older cycles.
                    if detail_cycles > 1:
                        detail_cycles -= 1
                    elif retained > 4:
                        retained //= 2
                    else:
                        raise ModelTransportError(f'Prompt exceeds the server context: {body}')
                    print(json.dumps(dict(event='qwen_context_shrink', step_id=rollout.tick,
                        full_detail_cycles=detail_cycles, retained_cycles=retained)), flush=True)
                    continue
                if 400 <= error.code < 500 and error.code != 429:
                    raise ModelTransportError(f'Server rejected the request: {last}')
                continue
            except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as error:
                last = error
                continue
            usage = result.get('usage') or {}
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                self.usage[key] = usage.get(key, self.usage[key])
            self.usage['requests'] += 1
            write_json(self.workspace.parent/'token_usage.json', dict(
                model=self.model, provider=settings.PROVIDER, usage=dict(self.usage)))
            choices = result.get('choices') or []
            if not choices:
                last = RuntimeError('Model returned no choices')
                continue
            choice = choices[0]
            return (choice.get('message') or {}), choice.get('finish_reason')
        raise ModelTransportError(f'Qwen server unreachable after retries: {last}')

    # ----- conversation ----------------------------------------------------
    def _wire_messages(self, detail_cycles=None, retained=None):
        """Recent cycles verbatim; older ones keep decisions, lose bulk and images."""
        if detail_cycles is None:
            detail_cycles = settings.FULL_DETAIL_CYCLES
        if retained is None:
            retained = settings.MAX_RETAINED_CYCLES
        keep_from = self.cycle - detail_cycles
        drop_before = self.cycle - retained
        wire = []
        dropped = 0
        for message in self.messages:
            entry = {k: v for k, v in message.items() if not k.startswith('_')}
            if not message.get('_pin') and message.get('_cycle', 0) < drop_before:
                dropped += 1
                continue
            if dropped:
                wire.append(_elision_notice(dropped))
                dropped = 0
            if message.get('_cycle', 0) >= keep_from or message.get('_pin'):
                wire.append(entry)
                continue
            content = entry.get('content')
            if isinstance(content, list):
                texts = [p['text'] for p in content if p.get('type') == 'text']
                released = sum(1 for p in content if p.get('type') == 'image_url')
                note = ' '.join(texts)
                if released:
                    note += f' [{released} image attachment(s) released from context; superseded by later observations]'
                entry['content'] = _truncate(note, settings.COMPACTED_TEXT_CHARS)
            elif isinstance(content, str):
                entry['content'] = _truncate(content, settings.COMPACTED_TEXT_CHARS)
            wire.append(entry)
        if dropped:  # Everything after the pinned opening aged out.
            wire.append(_elision_notice(dropped))
        return wire

    def _record(self, event):
        with (self.workspace/'agent_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def _tool_content(self, packet, *, attach_images):
        """Tool text plus the attachment payloads, mirroring Codex content items."""
        visible, attachments = packet, []
        if attach_images:
            visible = dict(packet, images=[])
            for item in packet.get('images', []):
                descriptor, payload = prepare_image(item, self.image_max_edge)
                visible['images'].append(descriptor)
                attachments.append(base64.b64encode(payload).decode('ascii'))
        return json.dumps(visible, separators=(',', ':')), attachments

    # ----- main loop -------------------------------------------------------
    def run(self, rollout):
        handlers = (dict(robodojo_start=rollout.start, robodojo_act=rollout.act)
                    if self.method == 'gpt_only' else
                    dict(robodojo_start=rollout.start, pi05_infer=rollout.infer,
                         robodojo_execute=rollout.execute))
        opening = ('Act as the autonomous policy agent for this single simulation rollout. '
                   'You control the robot through the rollout service tools listed for you; '
                   'keep your working notes inside your replies. Use English for every public '
                   'explanation, assessment, progress update and final report. '
                   'First call: ' + json.dumps(rollout.next_call()))
        self.messages = [dict(role='system', content=self.prompt, _pin=True, _cycle=0),
                         dict(role='user', content=opening, _pin=True, _cycle=0)]
        call_index = 0
        errors = 0
        idle_turns = 0
        consecutive_rejections = 0
        truncated_turns = 0
        while True:
            message, finish_reason = self._complete(rollout)
            tool_calls = message.get('tool_calls') or []
            text = message.get('content') or ''
            assistant = dict(role='assistant', content=text, _cycle=self.cycle)
            if tool_calls:
                assistant['tool_calls'] = _replayable_tool_calls(tool_calls)
            self.messages.append(assistant)
            if text.strip():
                self._record(dict(event='agent_message', step_id=rollout.tick, text=text))
                print(json.dumps(dict(event='agent_activity', item_type='agentMessage',
                    step_id=rollout.tick, text=text[:4000]), ensure_ascii=False), flush=True)
            if not tool_calls:
                if rollout.phase == 'done':
                    return
                idle_turns += 1
                if idle_turns > 6:
                    raise RuntimeError('Qwen stopped calling rollout services before native termination')
                if finish_reason == 'length':
                    # The whole output budget went to reasoning, so nothing was
                    # emitted. Repeating the request verbatim just spends it
                    # again; ask for the decision directly instead.
                    truncated_turns += 1
                    print(json.dumps(dict(event='qwen_output_truncated', step_id=rollout.tick,
                        phase=rollout.phase, consecutive=truncated_turns,
                        max_tokens=settings.MAX_OUTPUT_TOKENS)), flush=True)
                    if truncated_turns > 3:
                        raise RuntimeError(
                            'Model repeatedly exhausted its output budget before emitting a tool '
                            'call; lower ROLLOUT_QWEN_REASONING_EFFORT or raise ROLLOUT_QWEN_MAX_TOKENS')
                    self.messages.append(dict(role='user', _cycle=self.cycle, content=(
                        'Your previous reply was cut off at the output limit before any tool call, '
                        'so nothing was executed and the simulator is unchanged. Keep your analysis '
                        'short and emit the tool call immediately, using these arguments: '
                        + json.dumps(rollout.next_call()))))
                    continue
                self.messages.append(dict(role='user', _cycle=self.cycle, content=(
                    'The episode is unfinished and only a tool call advances it. '
                    'Reply with exactly one tool call now, using these arguments: '
                    + json.dumps(rollout.next_call()))))
                continue
            idle_turns = truncated_turns = 0
            finished = False
            for call in tool_calls:
                if finished:
                    # Every tool call must be answered or the next request is
                    # malformed; the episode is over, so nothing is executed.
                    self.messages.append(dict(role='tool', tool_call_id=call.get('id'),
                        name=(call.get('function') or {}).get('name'), _cycle=self.cycle,
                        content=json.dumps(dict(error='Episode already terminated natively',
                                                no_execution=True, retryable=False))))
                    continue
                function = call.get('function') or {}
                name = function.get('name')
                raw = function.get('arguments')
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                    phase=rollout.phase, step_id=rollout.tick)), flush=True)
                write_json(self.workspace/f'call_{call_index:04d}_request.json',
                           dict(call_id=call.get('id'), tool=name, arguments=raw))
                arguments = raw
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments or '{}')
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    if name not in handlers:
                        raise InputError(f'Unknown tool {name!r}; available: {sorted(handlers)}')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    arguments = _decode_object_arguments(arguments, self.schemas.get(name))
                    packet = handlers[name](**arguments)
                except InputError as error:
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                except TypeError as error:
                    # Wrong/missing keyword names are an argument fault, not a crash.
                    errors += 1
                    packet = rejected_input(InputError(str(error)), rollout, arguments, errors)
                    success = False
                else:
                    success = True
                if success:
                    consecutive_rejections = 0
                else:
                    consecutive_rejections += 1
                    if consecutive_rejections >= settings.MAX_CONSECUTIVE_REJECTIONS:
                        write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                        raise RuntimeError(
                            f'{consecutive_rejections} consecutive rejected tool calls without '
                            'an accepted action; abandoning this episode rather than spinning')
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                hint = _identifier_hint(packet, consecutive_rejections)
                attach = success and name != 'pi05_infer'
                content, attachments = self._tool_content(packet, attach_images=attach)
                self.messages.append(dict(role='tool', tool_call_id=call.get('id'),
                                          name=name, content=content, _cycle=self.cycle))
                if hint:
                    self.messages.append(dict(role='user', _cycle=self.cycle, content=hint))
                if attachments:
                    parts = [dict(type='text', text=(
                        f'Observation images for step_id={rollout.tick} '
                        '(head, left wrist, right wrist).'))]
                    parts += [dict(type='image_url', image_url=dict(
                        url='data:image/png;base64,' + data)) for data in attachments]
                    self.messages.append(dict(role='user', content=parts, _cycle=self.cycle))
                call_index += 1
                # Only an accepted call advances the rollout, so only an accepted
                # call ages the context window. Counting rejections here pushed the
                # pi05 proposal -- which carries the request_id the next call must
                # echo -- out of full detail, so the model could no longer see the
                # value it was being rejected for getting wrong: each rejection
                # made the next one more likely.
                if success:
                    self.cycle += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                if rollout.phase == 'done':
                    finished = True
            if finished:
                self._final_report(rollout)
                return

    def _final_report(self, rollout):
        """One closing turn for the native outcome. Never required for scoring."""
        self.messages.append(dict(role='user', _cycle=self.cycle, content=(
            'The episode has terminated natively. Briefly report the native outcome '
            'and the saved paths in English. Do not call any tool.')))
        try:
            message, _ = self._complete(rollout)
        except ModelTransportError:
            return
        text = (message.get('content') or '').strip()
        if text:
            self._record(dict(event='final_report', step_id=rollout.tick, text=text))
            print(json.dumps(dict(event='agent_activity', item_type='agentMessage',
                step_id=rollout.tick, text=text[:4000]), ensure_ascii=False), flush=True)

    def close(self):
        if self.messages:
            transcript = self.workspace/'conversation.json'
            try:
                write_json(transcript, [
                    {k: (v if k != 'content' or not isinstance(v, list) else
                         [p if p.get('type') != 'image_url' else dict(type='image_url', image_url='<elided>')
                          for p in v])
                     for k, v in m.items() if not k.startswith('_')}
                    for m in self.messages])
            except (OSError, ValueError):
                pass
