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

**Why it was needed**
- Needed a reliable way to run patched Python code while keeping compiled vLLM
  extensions from the base image.
- Prevented plugin override conflicts.
- Enabled controlled baseline vs EAGLE3 rollouts and debugging.
- Enabled direct profiling of prefill vs decode time on real captured payloads
  without custom engine instrumentation.

## Result

With these changes:
- VibeVoice-ASR runs with EAGLE3 on real audio payloads.
- Real captured 300s payload replay succeeds end-to-end.
- No speculative-path shape/assert crashes were observed in the validated runs.
