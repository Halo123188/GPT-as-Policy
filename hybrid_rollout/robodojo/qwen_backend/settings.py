"""Locally served Qwen backend settings.

The endpoint is an OpenAI-compatible chat-completions server (vLLM) that the
operator starts separately; nothing here reaches a third-party provider. Unlike
the Codex backend there is no reasoning-effort dial, so the sampling parameters
are the reproducibility surface and are recorded in ``worker.json``.
"""
import os

MODEL = os.environ.get('ROLLOUT_QWEN_MODEL', 'Qwen/Qwen3.8-27B')
PROVIDER = 'vllm_openai_compatible'
BASE_URL = os.environ.get('ROLLOUT_QWEN_BASE_URL', 'http://127.0.0.1:8000/v1')
API_KEY = os.environ.get('ROLLOUT_QWEN_API_KEY', 'EMPTY')

# The model card's thinking-mode sampling. Greedy decoding is explicitly
# discouraged for this model ("endless repetitions"), and a decode loop inside a
# long agentic episode would burn a GPU for hours, so reproducibility gives way
# to the vendor's setting. The exact values are recorded in worker.json.
TEMPERATURE = float(os.environ.get('ROLLOUT_QWEN_TEMPERATURE', '1.0'))
TOP_P = float(os.environ.get('ROLLOUT_QWEN_TOP_P', '0.95'))
TOP_K = int(os.environ.get('ROLLOUT_QWEN_TOP_K', '20'))
MIN_P = float(os.environ.get('ROLLOUT_QWEN_MIN_P', '0.0'))

# Qwen3.8 exposes its own reasoning_effort through the chat template, with the
# same 'xhigh' level the report used for the Codex model. Sent explicitly rather
# than relied on as a template default.
REASONING_EFFORT = os.environ.get('ROLLOUT_QWEN_REASONING_EFFORT', 'xhigh')
# Measured on a real hybrid decision (32k-token prompt, xhigh effort): ~5.8k
# completion tokens, of which ~4.5k is reasoning. Reasoning length is
# high-variance, and an 8192 cap truncated a live turn before any tool call, so
# the budget is set to roughly three times the observed mean. A truncated reply
# carries no tool call and wastes the whole turn, which at ~25 tok/s is minutes.
MAX_OUTPUT_TOKENS = int(os.environ.get('ROLLOUT_QWEN_MAX_TOKENS', '16384'))
# Must cover queueing plus generation on a *batched* server, not the solo
# latency. One decision is ~5.8k tokens; at ~25 tok/s alone that is minutes, but
# under 8-way batching per-stream decode drops to ~8-15 tok/s and a queued
# request waits behind others. A 900 s bound killed episodes mid-run once
# several replicas were saturated, and each expiry re-sends the whole request,
# adding load. Kept below the simulator's own 7200 s transport timeout.
REQUEST_TIMEOUT = float(os.environ.get('ROLLOUT_QWEN_REQUEST_TIMEOUT', '3600'))

# Attachment downscaling matches the Codex run so both methods see equal pixels.
DEFAULT_IMAGE_MAX_EDGE = int(os.environ.get('ROLLOUT_QWEN_IMAGE_MAX_EDGE', '480'))

# Sliding window over decision cycles. A frozen RoboDojo episode can take well
# over a hundred decisions; keeping every observation and every 50x14 proposal
# would exceed any context. Recent cycles stay verbatim, older ones keep their
# decision record but lose attachments and bulk arrays.
# Two cycles is exactly what a decision needs: the newest pi05 proposal and the
# observation it follows. A single proposal packet is ~18k tokens (the 50x14 FK
# trajectory), so a wider window holds two of them and pushes a measured 63k
# prompt toward the server's limit for no decision-relevant gain.
FULL_DETAIL_CYCLES = int(os.environ.get('ROLLOUT_QWEN_FULL_DETAIL_CYCLES', '2'))
COMPACTED_TEXT_CHARS = int(os.environ.get('ROLLOUT_QWEN_COMPACTED_TEXT_CHARS', '1200'))
# Truncation alone still grows without bound: a 150-decision episode would hold
# ~450 compacted messages and overflow any window. Past this age a whole cycle
# is dropped -- always the complete assistant/tool/attachment block, so every
# remaining tool reply still answers a tool call the model can see.
MAX_RETAINED_CYCLES = int(os.environ.get('ROLLOUT_QWEN_MAX_RETAINED_CYCLES', '36'))

# Transport-only retries. Model-side refusals and validation rejects are not
# retried here: they travel back to the agent as recoverable tool feedback.
RETRY_DELAYS = (5, 10, 20, 40, 60)

# The Codex backend leaves rejected tool calls unbounded because the hosted
# agent eventually stops on its own. An unattended local server does not, and a
# model that keeps re-sending one invalid action would hold a GPU forever. This
# caps only *consecutive* rejections: any accepted call resets the counter, so a
# recovering agent is never cut off mid-episode.
MAX_CONSECUTIVE_REJECTIONS = int(os.environ.get('ROLLOUT_QWEN_MAX_CONSECUTIVE_REJECTIONS', '40'))
