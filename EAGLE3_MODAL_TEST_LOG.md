# EAGLE3 + VibeVoice Modal Test Plan and Experiment Log

## Goal

Validate that this fork's EAGLE3 path works end-to-end for audio-conditioned
VibeVoice-ASR on Modal, then benchmark latency impact against baseline.

## Scope

- Target model: `microsoft/VibeVoice-ASR`
- Runtime: Modal GPU container running local vLLM fork code
- Primary method: `eagle3`
- Baseline comparator: spec decode disabled

## Preconditions

1. Modal auth is configured locally.
2. Modal secret `huggingface-secret` is available in selected env.
3. Local code changes are present in this repo:
   - `vllm/model_executor/models/vibevoice_asr.py`
   - `vllm/model_executor/models/llama_eagle3.py`
   - `vllm/v1/spec_decode/eagle.py`
   - `serve.py`

## Incremental Test Strategy

### Phase 0: Environment + packaging sanity

- Confirm `modal` CLI works in env `shankha-dev`.
- Deploy app and confirm container startup.
- Confirm startup log prints local vLLM path from `/workspace/vllm/...`.

Success criteria:
- App deploys.
- Health endpoint returns 200.
- Logs confirm local fork path is loaded.

### Phase 1: Baseline ASR path (no speculation)

- Run with `VLLM_SPEC_MODE=none`.
- Test short dummy audio (silence + tone).
- Verify requests complete and return structured response.

Success criteria:
- No startup/runtime crash.
- No fatal API errors.
- `server_timing_s` returned.

### Phase 2: EAGLE3 smoke

- Run with `VLLM_SPEC_MODE=eagle3`.
- Start with `VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=1`.
- Use same dummy audio inputs.

Success criteria:
- Server boots with EAGLE3 config.
- No shape/assert errors in speculative path.
- Request completes.

### Phase 3: EAGLE3 incremental ramp

- Increase speculation depth: 2 -> 4.
- Run short speech clips and then longer clips.
- Check output quality/retry behavior.

Success criteria:
- Stable serving at each depth.
- No regressions like repeated crashes or token-shape errors.

### Phase 4: Metrics + latency comparison

- Collect N repeated runs for baseline and EAGLE3.
- Read vLLM metrics endpoint and capture spec decode counters.
- Compare mean and p95 timing.

Success criteria:
- Spec counters populated.
- Quantified latency delta vs baseline.

## Runbook Commands (to execute incrementally)

1. Deploy:
   - `modal deploy --env shankha-dev serve.py`
2. Smoke:
   - `modal run --env shankha-dev serve.py`
3. Logs:
   - `modal app logs <app-id>`
4. Optional env override examples:
   - `VLLM_SPEC_MODE=none`
   - `VLLM_SPEC_MODE=eagle3`
   - `VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=1`

## Experiment Log

### 2026-03-04

- [x] Step 0.1: Validate Modal CLI and environment access.
- [x] Step 0.2: Deploy app in `shankha-dev`.
- [x] Step 0.3: Check startup logs for local vLLM path — **PASSED** (v0.15.0 overlay, no import errors).
- [x] Step 1.1: Baseline run with spec disabled — **PASSED**.
- [x] Step 2.1: EAGLE3 run with 1 speculative token — **PASSED**.
- [ ] Step 3.1: EAGLE3 run with 2 speculative tokens — skipped (jumped to 4).
- [x] Step 3.2: EAGLE3 run with 4 speculative tokens — **PASSED**.
- [x] Step 4.1: Gather metrics and timing summary — see below.

## Notes / Findings

### Early attempts (file overlay strategy) — superseded

- 2026-03-04 13:30 EST: `modal --version` succeeded (`1.3.0.post1`).
- 2026-03-04 13:31 EST: `modal environment list` confirms `shankha-dev` is active.
- 2026-03-04 13:51–14:46 EST: Tried several strategies to apply local vLLM fork
  changes onto the `vllm/vllm-openai:latest` (v0.11.0) Docker image:
  1. `pip install -e /workspace/vllm` — image build timed out (C extension compile).
  2. `PYTHONPATH=/workspace/vllm` — lost compiled `vllm._C`.
  3. Individual file overlays (`cp` specific `.py` files) — worked partially but
     hit cascading `ModuleNotFoundError`s due to version drift between our fork
     (based on v0.15.0) and the base image (v0.11.0). Required extensive
     `try/except ImportError` hacks in `llama_eagle3.py` and `eagle.py`.
  4. Disabled `eagle.py` overlay to reduce surface area — still fragile.

### Current strategy (pip overlay, matching base image) — working

- 2026-03-04 15:15 EST: Root cause identified — our fork is based on vLLM
  **v0.15.0** (confirmed via commit `d6416fdde` referencing "vLLM v0.15.0"),
  but `vllm/vllm-openai:latest` ships **v0.11.0**. Four minor versions of API
  drift caused all the import failures.
- Fix applied (same pattern as SGLang deploys):
  1. Base image pinned to `vllm/vllm-openai:v0.15.0`.
  2. `VLLM_TARGET_DEVICE=empty pip install --target /tmp/vllm-overlay` builds a
     pure-Python wheel (no C extension compilation, ~5MB, takes ~10s).
  3. `cp -rf /tmp/vllm-overlay/vllm/* .../dist-packages/vllm/` overlays Python
     files on top of the base image without uninstalling it. All compiled `.so`
     files and subpackages (e.g. `vllm_flash_attn`) remain intact.
- All compatibility hacks in `llama_eagle3.py` and `eagle.py` reverted to clean
  upstream imports. Only genuine fixes remain:
  - `llama_eagle3.py`: multimodal embedding merge in `embed_input_ids`.
  - `eagle.py`: graceful `image_token_index` fallback for non-vision models.
- VibeVoice plugin (`vllm_plugin`) override disabled:
  - `__init__.py` emptied so `register_vibevoice()` is never defined.
  - `entry_points.txt` truncated so vLLM doesn't call the plugin.
  - Tokenizer tool invoked via direct script path instead of `python -m`.
- Image builds in ~60s, deploys in ~90s total.

### Phase 0 results

- 2026-03-04 15:30 EST: Deploy succeeded with clean image build (62s).
- 2026-03-04 15:31 EST: Smoke test (`modal run`) passed:
  - `Using vLLM from: /usr/local/lib/python3.12/dist-packages/vllm/__init__.py`
  - No plugin override warning (no "will be overwritten" log).
  - Architecture resolved: `VibeVoiceForASRTraining` (our fork's model class).
  - Model loaded: 18.22 GiB, 37.8s. CUDA graphs captured. KV cache: 896K tokens.
  - `/health` returned 200 OK. `vLLM server ready after 193s`.
  - `Health: {'status': 'ok', 'vllm_status': 200}` — exit code 0.

### Phase 1 results — Baseline (no speculation)

- 2026-03-04 20:40 EST: Deployed with `VLLM_SPEC_MODE=none`. Health 200.
- Test: 1s 440Hz sine WAV, `recovery_mode=pure`.
- Result:
  - `server_timing_s`: **13.222s**
  - `retries`: 0, `error`: None
  - Segments: `[{"start_time": 0.0, "end_time": 1.0, "speaker_id": "0", "text": "[Music]"}]`
- Correct output (tone → `[Music]`). Baseline latency: ~13.2s.

### Phase 2 results — EAGLE3, 1 speculative token

- 2026-03-04 20:50 EST: Deployed with `VLLM_SPEC_MODE=eagle3`,
  `VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=1`,
  draft model `Rayzl/qwen2.5-vl-7b-eagle3-sgl`.
- Startup log confirmed: `EAGLE3 enabled with draft=Rayzl/qwen2.5-vl-7b-eagle3-sgl num_speculative_tokens=1`
- `speculative_config` visible in vLLM non-default args.
- Cold start: ~262s (draft model download included).
- Test: same 1s sine WAV.
- Result:
  - `server_timing_s`: **15.619s**
  - `retries`: 0, `error`: None
  - Segments: `[{"start_time": 0.0, "end_time": 1.0, "speaker_id": "0", "text": "[Music]"}]`
- EAGLE3 path runs end-to-end with audio inputs. No shape/assert errors.

### Phase 3 results — EAGLE3, 4 speculative tokens

- 2026-03-04 21:00 EST: Deployed with `VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=4`.
- First deploy timed out (300s startup limit hit during CUDA graph capture for
  spec decode). Increased `startup_timeout` from 300 → 600s and redeployed.
- Cold start: ~240s with 600s timeout — success.
- Test: 3× sequential requests, same 1s sine WAV.
- Results:

| Request | Wall (client) | Server timing | Error | Output       |
|---------|---------------|---------------|-------|--------------|
| 1 (cold)| 11.550s       | 11.228s       | None  | `[Music]`    |
| 2 (warm)| 0.416s        | **0.209s**    | None  | `[Music]`    |
| 3 (warm)| 0.395s        | **0.209s**    | None  | `[Music]`    |

- Warm-path server latency reached **0.209s** in this run.
- Important caveat: baseline and EAGLE3 measurements were not fully apples-to-apples
  (cold/warm mismatch, tiny-output synthetic workload), so treat `63x` as
  exploratory only until controlled A/B benchmarking is completed.
- All outputs correct. No shape errors, no retries, stable serving.

### Timing comparison summary (preliminary)

| Config                    | Server latency observed | Interpretation |
|---------------------------|-------------------------|----------------|
| Baseline (no spec)        | 13.222s                 | single short synthetic clip |
| EAGLE3, 1 spec token      | 15.619s                 | single cold-path run |
| EAGLE3, 4 spec tokens     | 0.209s (warm)           | promising warm-path signal |

Do not use this table as a production claim yet. Run controlled warm-vs-warm
paired A/B across a real speech set before concluding net speedup.

### Wave 2 follow-up validation (same deployment, EAGLE3=4)

- Anti-cache mini benchmark (10 unique synthetic clips, 0.8-2.0s):
  - `server_timing_s` median: **0.231s**
  - `server_timing_s` p95: **0.239s**
  - errors/retries: **0/10**
- Repeat-clip sensitivity (same clip x10):
  - `server_timing_s` median: **0.222s**
  - `server_timing_s` p95: **0.236s**
  - repeat vs unique delta: ~4% (small), suggesting speedup is not only cache artifact.
- Light concurrency (4 concurrent requests, different clips):
  - median `server_timing_s`: **0.281s**, p95 **0.294s**
  - errors/retries: **0/4**
  - response schema valid for all requests.

### Wave 3 longer-duration stability checks

- Sequential unique synthetic clips (2 each): 5s, 15s, 30s, 60s.
- Results:
  - 5s: median **0.190s**, p95 **0.190s**
  - 15s: median **0.209s**, p95 **0.209s**
  - 30s: median **0.227s**, p95 **0.228s**
  - 60s: median **0.298s**, p95 **0.300s**
- Stability: **8/8 successful**, no retries, no malformed segment schema.

### Critical caveats discovered during deeper audit

- Current benchmark conclusions are still vulnerable to confounders:
  1. cold-vs-warm mixing in headline comparisons,
  2. synthetic tone clips produce very short outputs (`[Music]`), not real ASR load,
  3. Modal lifecycle/caching effects are not fully controlled in log schema.
- `serve.py` startup timeout had to be increased from 300s to 600s for EAGLE3 cold starts.
- Some code-level validations are missing/partial and should be covered by tests:
  - aux hidden layer id count/bounds checks,
  - aux hidden tensor count/shape checks before combine,
  - stricter `d2t` mapping validation,
  - explicit mismatch checks for multimodal placeholder vs embedding count.

### Trainability readiness status

- Current state: **inference-ready smoke for EAGLE3+audio** is demonstrated.
- Not yet proven: **end-to-end draft training readiness** for voice workloads.
- Key gaps before claiming trainability:
  - no in-repo EAGLE3 draft trainer (requires external Speculators/SpecForge stack),
  - no automated `EAGLE3 + VibeVoice` CI tests yet,
  - no controlled quality regression study (WER/CER) under spec decode.

### Next test matrix (must-run before production claim)

1. **Controlled A/B latency benchmark** (warm-vs-warm, paired clips, real speech set).
2. **Quality regression battery** across clean/noisy/music/multi-speaker slices.
3. **Spec-depth sweep** (`none`, `1`, `2`, `4`, `6`) with acceptance + latency metrics.
4. **Startup/reliability soak** (cold start success rate, timeout/crash incidence).
5. **Cross-engine parity spot-check** (vLLM vs SGLang on identical prompts/configs).

### Real captured payload replay (`f5b6dcd...jsonl`)

- Source payload: `/Users/shankha/modal-projects/flash-projects/f5b6dcd86fbc41cdbb9312055b6c8c68.jsonl`
  (single-line JSON object with `messages[1].content[0].audio_url.url` as large
  `data:audio/wav;base64,...`).
- Initial replay attempt caused engine crash. Root cause from logs:
  - `NameError`: `VibeVoiceTokenizerStreamingCache` not defined.
  - Then `AssertionError`: expected 1 multimodal embedding, got 0.
- Fix applied in `vllm/model_executor/models/vibevoice_asr.py`:
  - added missing import for `VibeVoiceTokenizerStreamingCache` from
    `vllm.transformers_utils.processors.vibevoice_asr`.
- After redeploy, replay of the same payload succeeded:
  - Health cold-start time: **208.592s**
  - Request wall time: **25.826s**
  - `server_timing_s`: **23.719s**
  - `retries`: 0, `error`: None
  - Parsed segments: **36**
  - First segment starts at 0.0s and last reaches 300.0s.

### Hidden-state regeneration planning (vLLM-native)

Goal: generate offline training data for EAGLE3 draft training from real traffic
while preserving audio-conditioned verifier behavior.

Observed constraints:
- Current OpenAI-compatible `/v1/chat/completions` API does not directly return
  hidden states.
- This fork has EAGLE/EAGLE3 methods in `vllm/v1/spec_decode`, but does **not**
  currently include upstream `extract_hidden_states` speculative method files.
- We need hidden capture for **audio-conditioned multimodal** requests, so the
  capture path must execute inside the same vLLM model stack as production.

Recommended pflow for vLLM:
1. **Pass 1: Generation**
   - Replay request JSONL to vLLM endpoint and save regenerated assistant text.
2. **Pass 2: Prefill-only hidden capture**
   - Build full conversation (user + regenerated assistant) and run prefill-only
     forward to capture:
       - `hidden_state` (last layer)
       - `aux_hidden_state` (EAGLE3 selected layers)
   - Save to `.ckpt` with sharded prefix directories.
3. **Training data packaging**
   - Emit `input_ids`, `loss_mask`, `hidden_state`, `aux_hidden_state`.
   - Build `token_freq.pt` and `d2t/t2d` mappings for draft vocab training.

Implementation decision (locked):
- **vLLM-only**. Do not use SGLang fallback for hidden capture, because the
  current voice multimodal path is on vLLM.

Implementation track (vLLM-native):
1. Add a vLLM worker-extension capture path (Speculators-style) to collect:
   - selected EAGLE3 aux layers
   - last-layer hidden states
   - per-request metadata for chunked prefill attribution
2. Build a prefill-only capture runner using vLLM internals (scheduler/executor)
   that runs full multimodal preprocessing and aborts decode after prefill.
3. Integrate with replay pipeline:
   - pass 1 regenerate via deployed vLLM endpoint
   - pass 2 capture hidden states with local vLLM prefill runner
4. Persist `.ckpt` records with schema + reproducibility metadata.

Missing elements before large-scale regen:
1. **Multimodal prefill capture path in vLLM** for audio requests (not just text
   token IDs).
2. **Per-request token attribution** across chunked prefill iterations.
3. **Loss-mask alignment** for assistant spans under the exact chat template.
4. **Storage format + size controls** (bf16/fp16, chunking, compression).
5. **Resume/idempotency** for shard retries and partial failures.
6. **Data quality gates** (shape checks, token-length checks, layer count checks).
7. **Version pinning metadata** (`vllm` SHA, model revision, layer IDs, tokenizer
   revision) for reproducibility.

Current recommendation:
- Implement a small pilot (100-500 requests) first, validate hidden tensor
  integrity and training-readiness schema, then scale to full traffic replay.

### Hidden capture prototype (vLLM-only, incremental)

Implemented first-pass vLLM-native hidden capture wiring:
- Added `HiddenStateCaptureManager` in
  `vllm/v1/worker/hidden_state_capture.py`.
- Wired `GPUModelRunner` to:
  - capture **prefill-only** chunks per request (ignores decode tokens),
  - optionally include EAGLE3 aux layers when available,
  - finalize and persist `.ckpt` when request finishes.
- Capture is request-gated via OpenAI request `vllm_xargs`:
  - `capture_hidden_states: true`
  - `capture_id: "<stable-id>"`
- `serve.py` updates:
  - `transcribe_chunk(..., capture_hidden_states=False, capture_id=None)`
  - `transcribe_chunk_two_pass_capture(..., capture_id=None)`:
    - pass 1: normal generation/transcription
    - pass 2: prefill-only capture request (`max_tokens=1`) on full conversation
      using the same deployed vLLM server
  - hidden capture metadata returned in response (`capture_id`, relpath/path)
  - `list_hidden_state_captures(limit=...)` helper method.

Storage decision (requested):
- Use a **fresh v2 hidden-state volume**:
  - volume name: `eagle3-hidden-states-v2` (overridable via env var)
  - mount path: `/hidden-states-v2`
  - worker env: `VLLM_HIDDEN_STATES_OUTPUT_DIR=/hidden-states-v2`

Small local validation completed:
- Syntax compile checks passed for modified files.
- `HiddenStateCaptureManager` smoke test passed with dummy tensors:
  - output file written
  - `hidden_state` and `aux_hidden_state` shapes match expectations.

Live smoke validation (tiny request):
- Deployed updated `serve.py` to `shankha-dev`.
- Added/fixed same-deployment two-pass method call path
  (`transcribe_chunk_two_pass_capture` -> `transcribe_chunk.local(...)`).
- Ran a tiny 0.25s silent WAV request with
  `capture_id=pilot-smoke-1772751972`.
- Result:
  - `error=None`
  - `capture_error=None`
  - `hidden_state_relpath=pi/pilot-smoke-1772751972.ckpt`
  - `list_hidden_state_captures(limit=5)` returned that file
  - `two_pass_timing_s=9.314` (method timing)
  - end-to-end client wall (including cold start): ~207s

Scaling note (high-level):
- Not arbitrary/unbounded. Throughput is constrained by:
  - GPU memory + scheduler limits (`max_num_seqs`, batched tokens),
  - host RAM pressure from temporary hidden-state chunk buffers,
  - v2 volume write/commit throughput and file-count behavior,
  - per-request multimodal preprocessing cost (audio path).
- For scale-out, keep capture opt-in and run dedicated regen workers with lower
  concurrency than latency-benchmark settings.

### Prefill vs decode profiling (new)

- Added `get_metrics_snapshot()` method in `serve.py` that fetches
  `http://localhost:8000/metrics` and returns parsed vLLM counters:
  - `vllm:request_prefill_time_seconds_{sum,count}`
  - `vllm:request_decode_time_seconds_{sum,count}`
  - `vllm:e2e_request_latency_seconds_{sum,count}`
  - `vllm:time_to_first_token_seconds_{sum,count}`
  - `vllm:request_queue_time_seconds_{sum,count}`
  - speculative counters (`draft/accepted tokens`, etc.)
- Profiling method:
  1. Snapshot metrics before request.
  2. Run one request.
  3. Snapshot metrics after request.
  4. Compute deltas (`after - before`) for prefill/decode/e2e/ttft.
  (Single in-flight request used to keep deltas attributable.)

### Batch replay profiling results (4 real captured requests)

- Source directories sampled:
  - `fathom-quasar/quasar/2026/03/05/11`
  - `fathom-quasar/quasar/2026/03/05/12`
- Files replayed:
  - `001447dec73a4d2d9d911820d5b89dc5.jsonl`
  - `015dea3626b24d5a9fec53dc8f8214fc.jsonl`
  - `0020cc3775bf40819b796951352dd56e.jsonl`
  - `00b4c7bbd3724cb8a649d9c91b2573a0.jsonl`
- All requests succeeded (`4/4`), no retries, no errors.

| File | server_timing_s | prefill_delta_s | decode_delta_s | ttft_delta_s | segment_count |
|------|------------------|-----------------|----------------|--------------|---------------|
| `001447...` | 24.864 | 2.650 | 14.075 | 2.852 | 44 |
| `015dea...` | 14.200 | 1.009 | 12.789 | 1.194 | 38 |
| `0020cc...` | 14.437 | 1.008 | 13.020 | 1.200 | 37 |
| `00b4c7...` | 13.868 | 1.011 | 12.454 | 1.198 | 36 |

Aggregate (n=4):
- `server_timing_s`: p50 **14.437s**, p95 **24.864s**
- `prefill_delta_s`: p50 **1.011s**, p95 **2.650s**
- `decode_delta_s`: p50 **13.020s**, p95 **14.075s**
- `ttft_delta_s`: p50 **1.200s**, p95 **2.852s**
- `queue_delta_s`: ~0 (single in-flight execution)

Interpretation:
- For these long (~300s) captured payloads, **decode dominates total latency**.
- Prefill is non-trivial but much smaller than decode on warm runs.
- Detailed raw profiling output saved to:
  `/Users/shankha/OSS/vllm/replay_profile_results.json`.

