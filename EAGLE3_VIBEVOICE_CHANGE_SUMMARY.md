# EAGLE3 + VibeVoice Change Summary

This document is a clean summary of what was changed in this vLLM fork to make
EAGLE3 work with `microsoft/VibeVoice-ASR`, and why each change was necessary.

## Goal

Enable stable speculative decoding with `method=eagle3` for audio-conditioned
VibeVoice-ASR requests.

## Core vLLM code changes

### 1) `vllm/model_executor/models/vibevoice_asr.py`

**What changed**
- Added `SupportsEagle3` to `VibeVoiceASRForConditionalGeneration`.
- Added:
  - `set_aux_hidden_state_layers(...)`
  - `get_eagle3_aux_hidden_state_layers(...)`
- Added missing import:
  - `VibeVoiceTokenizerStreamingCache`

**Why it was needed**
- EAGLE3 requires verifier models to expose auxiliary hidden-state layer hooks.
  VibeVoice-ASR did not expose these interfaces directly.
- Without these methods, the proposer cannot configure/read verifier aux layers
  for EAGLE3.
- The missing tokenizer cache import caused a runtime `NameError` during replay
  of real captured payloads.

---

### 2) `vllm/model_executor/models/llama_eagle3.py`

**What changed**
- Updated `embed_input_ids(...)` to merge multimodal embeddings into
  `inputs_embeds` using `_merge_multimodal_embeddings(...)` when provided.

**Why it was needed**
- The draft model path was ignoring multimodal embeddings.
- For VibeVoice-ASR audio requests, this made draft inputs inconsistent with
  verifier conditioning, causing speculative decode instability/incorrectness.

---

### 3) `vllm/v1/spec_decode/eagle.py`

**What changed**
- In draft model setup, replaced hard access to
  `target_model.config.image_token_index` with safe fallback logic:
  - try `image_token_index`
  - else try `image_token_id`
  - else continue without setting it (debug log)

**Why it was needed**
- VibeVoice-ASR is multimodal audio, not a vision model, and may not define
  vision-specific config fields.
- Previous logic could crash with attribute errors on valid audio models.

## Deployment/runtime glue changes

### 4) `serve.py` (Modal deployment script)

**What changed**
- Added deploy path that overlays local vLLM Python code onto a matching
  `vllm/vllm-openai:v0.15.0` base image.
- Disabled VibeVoice plugin auto-registration so it does not override this
  fork's patched model class.
- Added EAGLE3 config wiring via env vars and `--speculative-config`.
- Added startup diagnostics (`Using vLLM from: ...`) and increased startup wait
  timeout to handle EAGLE3 cold starts.
- Added `get_metrics_snapshot()` method to expose parsed vLLM `/metrics`
  counters (prefill/decode/e2e/ttft/queue + speculative counters) for
  per-request profiling via before/after metric deltas.
- Added `transcribe_chunk_two_pass_capture()` to run generation and hidden-state
  prefill capture sequentially on the same deployed vLLM server.

**Why it was needed**
- Needed a reliable way to run patched Python code while keeping compiled vLLM
  extensions from the base image.
- Prevented plugin override conflicts.
- Enabled controlled baseline vs EAGLE3 rollouts and debugging.
- Enabled direct profiling of prefill vs decode time on real captured payloads
  without custom engine instrumentation.
- Added an end-to-end hook for vLLM-only two-pass data regeneration at request
  granularity.

---

### 5) vLLM-native hidden-state capture prototype

**What changed**
- Added `vllm/v1/worker/hidden_state_capture.py` with
  `HiddenStateCaptureManager` to accumulate prefill hidden-state chunks and
  persist `.ckpt` files per request.
- Wired `vllm/v1/worker/gpu_model_runner.py` to:
  - capture prefill-only tokens per request from model output,
  - include aux layers when present (EAGLE3 path),
  - finalize and save capture files when requests finish.
- Extended `serve.py` to support capture-triggered requests:
  - `transcribe_chunk(..., capture_hidden_states=False, capture_id=None)`
  - sets `vllm_xargs.capture_hidden_states/capture_id` for the OpenAI request.
  - adds helper `list_hidden_state_captures(...)`.
- Added a dedicated hidden-state output volume:
  - fresh v2 volume name: `eagle3-hidden-states-v2` (env-overridable)
  - mount path: `/hidden-states-v2`
  - env passed to vLLM workers: `VLLM_HIDDEN_STATES_OUTPUT_DIR`.

**Why it was needed**
- OpenAI-compatible responses do not directly return hidden states.
- Offline EAGLE3 draft training requires verifier hidden states at scale.
- We need capture inside the same vLLM multimodal execution path as production
  (audio-conditioned), not a separate fallback stack.

## Result

With these changes:
- VibeVoice-ASR runs with EAGLE3 on real audio payloads.
- Real captured 300s payload replay succeeds end-to-end.
- No speculative-path shape/assert crashes were observed in the validated runs.
- A first vLLM-native hidden-state capture path now exists for incremental pilot
  runs on real requests, with output persisted to a dedicated v2 volume.
