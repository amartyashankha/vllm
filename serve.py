"""Minimal VibeVoice-ASR vLLM server for latency benchmarking.

Deploy:
    modal deploy --env ai-dev projects/vibevoice-latency-bench/serve.py

Smoke test:
    modal run --env ai-dev projects/vibevoice-latency-bench/serve.py

Usage from Python:
    import modal
    cls = modal.Cls.from_name("vibevoice-latency-bench", "VibeVoiceServer", environment_name="ai-dev")
    server = cls()
    result = server.transcribe_chunk.remote(audio_b64="...", duration_secs=300.0)
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("vibevoice-latency-bench")
LOCAL_VLLM_DIR = Path(__file__).resolve().parent

gpu_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.15.0")
    .dockerfile_commands(
        "ENTRYPOINT []",
        "CMD []",
        "RUN ln -sf /usr/bin/python3 /usr/local/bin/python3.12",
        "RUN ln -sf /usr/bin/python3 /usr/local/bin/python",
        "RUN python3 -m ensurepip --default-pip || true",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .add_local_dir(str(LOCAL_VLLM_DIR), "/workspace/vllm", copy=True)
    .run_commands(
        "pip install google-cloud-storage requests huggingface_hub aiohttp",
        "git clone https://github.com/microsoft/VibeVoice.git /vibevoice",
        "cd /vibevoice && pip install -e . --no-deps",
        # Disable the VibeVoice vLLM plugin so it doesn't override our fork's
        # model class (which has EAGLE3 hooks). We gut __init__.py and remove
        # the entry-point metadata so vLLM won't try to call register_vibevoice.
        "rm -f /vibevoice/vllm_plugin/__init__.py && touch /vibevoice/vllm_plugin/__init__.py",
        "find /usr/local/lib/python3.12/dist-packages -path '*/vibevoice*.dist-info/entry_points.txt' -exec truncate -s0 {} +",
        "pip install librosa soundfile numba llvmlite ml-collections absl-py pydub av accelerate diffusers gradio aiortc",
        # Install our local vLLM fork's Python code over the base image.
        # Build a pure-Python wheel then overlay it WITHOUT uninstalling the
        # base package (which would delete compiled .so / flash-attn wrappers).
        "VLLM_TARGET_DEVICE=empty pip install /workspace/vllm --no-deps --no-build-isolation --target /tmp/vllm-overlay",
        "cp -rf /tmp/vllm-overlay/vllm/* /usr/local/lib/python3.12/dist-packages/vllm/ && rm -rf /tmp/vllm-overlay",
    )
)

model_volume = modal.Volume.from_name("vibevoice-model-cache", create_if_missing=True)
HIDDEN_STATE_VOLUME_NAME = os.getenv(
    "VLLM_HIDDEN_STATE_VOLUME_NAME", "eagle3-hidden-states-v2"
)
hidden_state_volume = modal.Volume.from_name(
    HIDDEN_STATE_VOLUME_NAME, create_if_missing=True
)

MODEL_ID = "microsoft/VibeVoice-ASR"
MODEL_CACHE_DIR = "/model-cache"
HIDDEN_STATE_OUTPUT_DIR = "/hidden-states-v2"
VLLM_PORT = 8000
MAX_RECOVERY_RETRIES = 3
DEFAULT_EAGLE3_DRAFT_MODEL = "Rayzl/qwen2.5-vl-7b-eagle3-sgl"
EAGLE3_DRAFT_MODEL = os.getenv(
    "VLLM_EAGLE3_DRAFT_MODEL", DEFAULT_EAGLE3_DRAFT_MODEL
)
EAGLE3_NUM_SPECULATIVE_TOKENS = int(
    os.getenv("VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS", "4")
)
VLLM_SPEC_MODE = os.getenv("VLLM_SPEC_MODE", "eagle3").strip().lower()
MODAL_NUM_CONTAINERS = int(os.getenv("MODAL_NUM_CONTAINERS", "1"))


def _sanitize_capture_id(capture_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", capture_id).strip("._-")
    return cleaned or "capture"


def _hidden_state_relpath(capture_id: str) -> str:
    safe_id = _sanitize_capture_id(capture_id)
    prefix = safe_id[:2] if len(safe_id) >= 2 else "00"
    return f"{prefix}/{safe_id}.ckpt"


def _build_eagle3_speculative_config() -> str | None:
    if VLLM_SPEC_MODE in ("none", "off", "disabled"):
        return None
    if VLLM_SPEC_MODE != "eagle3":
        raise ValueError(
            f"Unsupported VLLM_SPEC_MODE={VLLM_SPEC_MODE!r}; "
            "expected one of: eagle3, none"
        )
    return json.dumps(
        {
            "model": EAGLE3_DRAFT_MODEL,
            "method": "eagle3",
            "num_speculative_tokens": EAGLE3_NUM_SPECULATIVE_TOKENS,
            "draft_tensor_parallel_size": 1,
        }
    )


def _iter_prometheus_samples(metrics_text: str):
    """Yield parsed Prometheus samples as (base_name, full_name, value)."""
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        full_name, value_str = parts
        try:
            value = float(value_str)
        except ValueError:
            continue
        base_name = full_name.split("{", 1)[0]
        yield base_name, full_name, value


def _sum_prom_metric(metrics_text: str, metric_name: str) -> float | None:
    total = 0.0
    found = False
    for base_name, _full_name, value in _iter_prometheus_samples(metrics_text):
        if base_name == metric_name:
            total += value
            found = True
    return total if found else None


def _safe_mean(sum_value: float | None, count_value: float | None) -> float | None:
    if sum_value is None or count_value is None or count_value <= 0:
        return None
    return sum_value / count_value


def _extract_vllm_latency_stats(metrics_text: str) -> dict[str, float | None]:
    """Extract core latency + speculative counters from /metrics text."""
    stats: dict[str, float | None] = {}

    metric_names = [
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:mm_cache_queries_total",
        "vllm:mm_cache_hits_total",
    ]
    for name in metric_names:
        stats[name] = _sum_prom_metric(metrics_text, name)

    stats["prefill_mean_s"] = _safe_mean(
        stats["vllm:request_prefill_time_seconds_sum"],
        stats["vllm:request_prefill_time_seconds_count"],
    )
    stats["decode_mean_s"] = _safe_mean(
        stats["vllm:request_decode_time_seconds_sum"],
        stats["vllm:request_decode_time_seconds_count"],
    )
    stats["e2e_mean_s"] = _safe_mean(
        stats["vllm:e2e_request_latency_seconds_sum"],
        stats["vllm:e2e_request_latency_seconds_count"],
    )
    stats["ttft_mean_s"] = _safe_mean(
        stats["vllm:time_to_first_token_seconds_sum"],
        stats["vllm:time_to_first_token_seconds_count"],
    )
    stats["queue_mean_s"] = _safe_mean(
        stats["vllm:request_queue_time_seconds_sum"],
        stats["vllm:request_queue_time_seconds_count"],
    )

    draft_tokens = stats["vllm:spec_decode_num_draft_tokens_total"]
    accepted_tokens = stats["vllm:spec_decode_num_accepted_tokens_total"]
    if (
        draft_tokens is not None
        and draft_tokens > 0
        and accepted_tokens is not None
    ):
        stats["spec_accept_rate"] = accepted_tokens / draft_tokens
    else:
        stats["spec_accept_rate"] = None

    mm_queries = stats["vllm:mm_cache_queries_total"]
    mm_hits = stats["vllm:mm_cache_hits_total"]
    if mm_queries is not None and mm_queries > 0 and mm_hits is not None:
        stats["mm_cache_hit_rate"] = mm_hits / mm_queries
    else:
        stats["mm_cache_hit_rate"] = None

    return stats


# ---------------------------------------------------------------------------
# Streaming repetition detector (from VibeVoice upstream)
# ---------------------------------------------------------------------------
class RepetitionDetector:
    """Detect repetition patterns in streaming text."""

    def __init__(self, min_pattern_len=10, min_repeats=10, window_size=400):
        self.min_pattern_len = min_pattern_len
        self.min_repeats = min_repeats
        self.window_size = window_size
        self.text = ""

    def add_text(self, new_text):
        self.text += new_text
        return self._check_repetition()

    def _check_repetition(self):
        if len(self.text) < self.min_pattern_len * self.min_repeats:
            return False, len(self.text)

        window = self.text[-self.window_size:] if len(self.text) > self.window_size else self.text

        for pattern_len in range(self.min_pattern_len, len(window) // self.min_repeats + 1):
            pattern = window[-pattern_len:]
            count = 0
            pos = len(window)
            while pos >= pattern_len:
                if window[pos - pattern_len:pos] == pattern:
                    count += 1
                    pos -= pattern_len
                else:
                    break
            if count >= self.min_repeats:
                return True, len(self.text) - (count * pattern_len)

        words = window.split()
        if len(words) >= self.min_repeats * 2:
            for phrase_len in range(2, 6):
                if len(words) < phrase_len * self.min_repeats:
                    continue
                phrase = " ".join(words[-phrase_len:])
                count = 0
                idx = len(words)
                while idx >= phrase_len:
                    candidate = " ".join(words[idx - phrase_len:idx])
                    if candidate == phrase:
                        count += 1
                        idx -= phrase_len
                    else:
                        break
                if count >= self.min_repeats:
                    repeated_text = (phrase + " ") * count
                    good_end = len(self.text) - len(repeated_text.rstrip()) + len(phrase)
                    return True, max(0, good_end)

        return False, len(self.text)


def _find_last_segment_boundary(text):
    """Find position after the last complete JSON segment boundary (},)."""
    pos = text.rfind("},")
    return pos + 2 if pos != -1 else -1


def _parse_segments(text: str) -> list[dict]:
    """Parse VibeVoice JSON output into normalized segments."""
    if not text or not text.strip():
        return []

    text = text.strip()
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        boundary = _find_last_segment_boundary(text)
        if boundary > 0:
            text = text[:boundary].rstrip(",") + "]"
        else:
            text = text + "]"

    text = text.rstrip()
    if text.endswith(",]"):
        text = text[:-2] + "]"

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        text = text.replace(",]", "]").replace(",}", "}")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, list):
        return []

    KEY_MAP = {
        "Start time": "start_time", "Start": "start_time", "start": "start_time",
        "End time": "end_time", "End": "end_time", "end": "end_time",
        "Speaker ID": "speaker_id", "Speaker": "speaker_id", "speaker": "speaker_id",
        "Content": "text", "content": "text", "Text": "text",
    }

    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = {}
        for k, v in item.items():
            norm_key = KEY_MAP.get(k, k)
            normalized[norm_key] = v
        if "start_time" in normalized and "text" in normalized:
            try:
                normalized["start_time"] = float(normalized["start_time"])
                normalized["end_time"] = float(normalized.get("end_time", normalized["start_time"]))
                normalized["speaker_id"] = str(normalized.get("speaker_id", "0"))
                normalized["text"] = str(normalized["text"])
                segments.append(normalized)
            except (ValueError, TypeError):
                continue

    return segments


# ---------------------------------------------------------------------------
# GPU server — minimal, single container for benchmarking
# ---------------------------------------------------------------------------
@app.cls(
    image=gpu_image,
    gpu="H100",
    min_containers=MODAL_NUM_CONTAINERS,
    max_containers=MODAL_NUM_CONTAINERS,
    timeout=86400,
    scaledown_window=600,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        HIDDEN_STATE_OUTPUT_DIR: hidden_state_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
@modal.concurrent(max_inputs=4)
class VibeVoiceServer:
    @modal.enter()
    def start_server(self):
        import warnings

        from huggingface_hub import snapshot_download

        print(
            "Server config: gpu=H100 mode=bf16 "
            f"containers={MODAL_NUM_CONTAINERS} max_inputs=4"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import vllm; print(f'Using vLLM from: {vllm.__file__}')",
            ],
            check=True,
        )

        # Download model if not cached
        marker = Path(MODEL_CACHE_DIR) / f".downloaded_{MODEL_ID.replace('/', '__')}"
        if not marker.exists():
            print(f"Downloading model {MODEL_ID} ...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snapshot_download(
                    MODEL_ID,
                    cache_dir=MODEL_CACHE_DIR,
                    local_dir=f"{MODEL_CACHE_DIR}/model",
                )
            marker.touch()
            model_volume.commit()
        else:
            print("Model already cached.")

        # Always attempt tokenizer-file generation in case an existing cached
        # model directory was created before tokenizer compatibility files existed.
        plugin_script = "/vibevoice/vllm_plugin/tools/generate_tokenizer_files.py"
        plugin_ret = subprocess.run(
            [
                sys.executable,
                plugin_script,
                "--output",
                f"{MODEL_CACHE_DIR}/model",
            ],
            check=False,
        ).returncode
        if plugin_ret == 0:
            print("Tokenizer generation completed.")
            model_volume.commit()
        else:
            print(
                f"Tokenizer generation skipped (exit code={plugin_ret})."
            )

        # Start vLLM
        model_path = f"{MODEL_CACHE_DIR}/model"
        os.environ.setdefault("VLLM_HIDDEN_STATES_OUTPUT_DIR", HIDDEN_STATE_OUTPUT_DIR)
        speculative_config = _build_eagle3_speculative_config()
        cmd = [
            "vllm", "serve", model_path,
            "--served-model-name", "vibevoice",
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--max-num-seqs", "64",
            "--max-model-len", "65536",
            "--max-num-batched-tokens", "32768",
            "--gpu-memory-utilization", "0.9",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--chat-template-content-format", "openai",
            "--tensor-parallel-size", "1",
            "--port", str(VLLM_PORT),
        ]
        if speculative_config is not None:
            cmd.extend(["--speculative-config", speculative_config])
            print(
                f"EAGLE3 enabled with draft={EAGLE3_DRAFT_MODEL} "
                f"num_speculative_tokens={EAGLE3_NUM_SPECULATIVE_TOKENS}"
            )
        else:
            print("Spec decode disabled (VLLM_SPEC_MODE=none)")
        print(
            "Hidden-state capture path enabled at "
            f"{HIDDEN_STATE_OUTPUT_DIR} (volume={HIDDEN_STATE_VOLUME_NAME})"
        )
        print(f"Starting vLLM: {' '.join(cmd)}")
        self.vllm_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        import requests
        startup_timeout = 600
        for i in range(startup_timeout):
            ret = self.vllm_proc.poll()
            if ret is not None:
                raise RuntimeError(f"vLLM process exited with code {ret}")
            try:
                r = requests.get(f"http://localhost:{VLLM_PORT}/health")
                if r.status_code == 200:
                    print(f"vLLM server ready after {i}s")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            self.vllm_proc.terminate()
            raise RuntimeError(
                f"vLLM server did not start within {startup_timeout}s"
            )

    @modal.method()
    def transcribe_chunk(
        self,
        audio_b64: str,
        duration_secs: float,
        audio_mime: str = "audio/wav",
        recovery_mode: str = "recover",
        capture_hidden_states: bool = False,
        capture_id: str | None = None,
    ) -> dict:
        """Transcribe a single audio chunk.

        Args:
            audio_b64: base64-encoded audio (WAV or MP3)
            duration_secs: audio duration in seconds (used in prompt)
            audio_mime: MIME type ("audio/wav" or "audio/mpeg")
            recovery_mode: "recover" = detect loops + retry (production behavior)
                           "pure" = detect loops + immediate error (clean latency)
            capture_hidden_states: whether to store prefill hidden states to disk
            capture_id: optional stable ID used for output file naming

        Returns:
            dict with: segments, retries, error, server_timing_s, loop_detected_at_s
        """
        import aiohttp

        async def _transcribe():
            t_start = time.monotonic()
            data_url = f"data:{audio_mime};base64,{audio_b64}"
            prompt_text = (
                f"This is a {duration_secs:.2f} seconds audio, please transcribe it "
                "with these keys: Start time, End time, Speaker ID, Content"
            )

            base_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            accumulated_text = ""
            retry_count = 0
            is_recovery = False
            loop_detected_at_s = None
            requested_capture_id = (
                _sanitize_capture_id(capture_id)
                if capture_id
                else f"capture-{time.time_ns()}"
            ) if capture_hidden_states else None

            def _attach_capture_meta(result: dict) -> dict:
                if not capture_hidden_states or requested_capture_id is None:
                    return result
                relpath = _hidden_state_relpath(requested_capture_id)
                result["capture_hidden_states"] = True
                result["capture_id"] = requested_capture_id
                result["hidden_state_relpath"] = relpath
                result["hidden_state_path"] = f"{HIDDEN_STATE_OUTPUT_DIR}/{relpath}"
                return result

            while retry_count <= MAX_RECOVERY_RETRIES:
                messages = list(base_messages)
                if accumulated_text:
                    messages.append({"role": "assistant", "content": accumulated_text})

                if is_recovery:
                    recovery_temp = 0.1 + 0.1 * retry_count
                    payload = {
                        "model": "vibevoice",
                        "messages": messages,
                        "max_tokens": 32768,
                        "temperature": recovery_temp,
                        "top_p": 0.95,
                        "stream": True,
                    }
                else:
                    payload = {
                        "model": "vibevoice",
                        "messages": messages,
                        "max_tokens": 32768,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "stream": True,
                    }
                if capture_hidden_states and requested_capture_id is not None:
                    payload["vllm_xargs"] = {
                        "capture_hidden_states": True,
                        "capture_id": requested_capture_id,
                    }

                detector = RepetitionDetector(min_pattern_len=10, min_repeats=10, window_size=400)
                detector.text = accumulated_text
                new_text = ""
                loop_detected = False

                try:
                    timeout = aiohttp.ClientTimeout(total=1200)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                            json=payload,
                        ) as resp:
                            resp.raise_for_status()
                            async for raw_line in resp.content:
                                line = raw_line.decode("utf-8").strip()
                                if not line.startswith("data: "):
                                    continue
                                json_str = line[6:]
                                if json_str.strip() == "[DONE]":
                                    break
                                try:
                                    data = json.loads(json_str)
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        new_text += content
                                        full_text = accumulated_text + new_text
                                        detector.text = full_text
                                        is_looping, good_end = detector._check_repetition()
                                        if is_looping:
                                            loop_detected_at_s = round(time.monotonic() - t_start, 3)

                                            if recovery_mode == "pure":
                                                # Pure mode: detect + error immediately
                                                segments = _parse_segments(full_text[:good_end])
                                                return _attach_capture_meta({
                                                    "segments": segments,
                                                    "transcript_text": full_text[:good_end],
                                                    "retries": 0,
                                                    "error": "loop_detected",
                                                    "server_timing_s": loop_detected_at_s,
                                                    "loop_detected_at_s": loop_detected_at_s,
                                                })

                                            # Recover mode: truncate + retry
                                            boundary = _find_last_segment_boundary(full_text[:good_end])
                                            if boundary > 0:
                                                accumulated_text = full_text[:boundary]
                                            else:
                                                accumulated_text = ""
                                            is_recovery = True
                                            retry_count += 1
                                            loop_detected = True
                                            print(f"  Loop detected at {good_end} chars, retry {retry_count} from {len(accumulated_text)} chars")
                                            break
                                except json.JSONDecodeError:
                                    continue

                    if not loop_detected:
                        final_text = accumulated_text + new_text
                        if accumulated_text and new_text:
                            stripped = new_text.lstrip()
                            if stripped.startswith("[{"):
                                new_text = stripped[1:]
                            elif stripped.startswith("["):
                                new_text = stripped[1:]
                            elif stripped.startswith("},"):
                                new_text = stripped[2:]
                            final_text = accumulated_text + new_text

                        segments = _parse_segments(final_text)
                        elapsed = time.monotonic() - t_start
                        return _attach_capture_meta({
                            "segments": segments,
                            "transcript_text": final_text,
                            "retries": retry_count,
                            "error": None,
                            "server_timing_s": round(elapsed, 3),
                            "loop_detected_at_s": loop_detected_at_s,
                        })

                except Exception as e:
                    print(f"  Error (retry {retry_count}): {type(e).__name__}: {e}")
                    if retry_count >= MAX_RECOVERY_RETRIES:
                        segments = _parse_segments(accumulated_text) if accumulated_text else []
                        elapsed = time.monotonic() - t_start
                        return _attach_capture_meta({
                            "segments": segments,
                            "transcript_text": accumulated_text,
                            "retries": retry_count,
                            "error": str(e),
                            "server_timing_s": round(elapsed, 3),
                            "loop_detected_at_s": loop_detected_at_s,
                        })
                    retry_count += 1
                    is_recovery = True

            print(f"  All {MAX_RECOVERY_RETRIES} retries exhausted")
            segments = _parse_segments(accumulated_text) if accumulated_text else []
            elapsed = time.monotonic() - t_start
            return _attach_capture_meta({
                "segments": segments,
                "transcript_text": accumulated_text,
                "retries": retry_count,
                "error": "max retries exhausted",
                "server_timing_s": round(elapsed, 3),
                "loop_detected_at_s": loop_detected_at_s,
            })

        result = asyncio.run(_transcribe())
        if capture_hidden_states:
            try:
                hidden_state_volume.commit()
            except Exception as e:
                result["hidden_state_commit_error"] = (
                    f"{type(e).__name__}: {e}"
                )
        return result

    @modal.method()
    def transcribe_chunk_two_pass_capture(
        self,
        audio_b64: str,
        duration_secs: float,
        audio_mime: str = "audio/wav",
        recovery_mode: str = "recover",
        capture_id: str | None = None,
    ) -> dict:
        """Run generation pass, then prefill capture pass on same deployment."""
        import requests

        t_start = time.monotonic()
        result = self.transcribe_chunk.local(
            audio_b64=audio_b64,
            duration_secs=duration_secs,
            audio_mime=audio_mime,
            recovery_mode=recovery_mode,
            capture_hidden_states=False,
        )
        if result.get("error"):
            result["capture_hidden_states"] = False
            result["capture_error"] = "skip_capture_due_to_transcription_error"
            result["two_pass_timing_s"] = round(time.monotonic() - t_start, 3)
            return result

        transcript_text = result.get("transcript_text")
        if not transcript_text:
            transcript_text = json.dumps(result.get("segments", []), ensure_ascii=False)

        stable_capture_id = _sanitize_capture_id(
            capture_id or f"capture-{time.time_ns()}"
        )
        prompt_text = (
            f"This is a {duration_secs:.2f} seconds audio, please transcribe it "
            "with these keys: Start time, End time, Speaker ID, Content"
        )
        data_url = f"data:{audio_mime};base64,{audio_b64}"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that transcribes audio input "
                    "into text output in JSON format."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {"role": "assistant", "content": str(transcript_text)},
        ]

        payload = {
            "model": "vibevoice",
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "vllm_xargs": {
                "capture_hidden_states": True,
                "capture_id": stable_capture_id,
            },
        }

        try:
            r = requests.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                json=payload,
                timeout=1200,
            )
            r.raise_for_status()
            _ = r.json()
        except Exception as e:
            result["capture_hidden_states"] = True
            result["capture_id"] = stable_capture_id
            result["capture_error"] = f"{type(e).__name__}: {e}"
            result["two_pass_timing_s"] = round(time.monotonic() - t_start, 3)
            return result

        relpath = _hidden_state_relpath(stable_capture_id)
        result["capture_hidden_states"] = True
        result["capture_id"] = stable_capture_id
        result["hidden_state_relpath"] = relpath
        result["hidden_state_path"] = f"{HIDDEN_STATE_OUTPUT_DIR}/{relpath}"
        result["two_pass_timing_s"] = round(time.monotonic() - t_start, 3)
        try:
            hidden_state_volume.commit()
        except Exception as e:
            result["hidden_state_commit_error"] = f"{type(e).__name__}: {e}"
        return result

    @modal.method()
    def get_metrics_snapshot(self) -> dict:
        """Return parsed vLLM /metrics counters for latency profiling."""
        import requests

        r = requests.get(f"http://localhost:{VLLM_PORT}/metrics", timeout=30)
        r.raise_for_status()
        stats = _extract_vllm_latency_stats(r.text)
        stats["collected_at_unix_s"] = time.time()
        return stats

    @modal.method()
    def list_hidden_state_captures(self, limit: int = 20) -> list[str]:
        """List recently written hidden-state capture files."""
        hidden_state_volume.reload()
        root = Path(HIDDEN_STATE_OUTPUT_DIR)
        if not root.exists():
            return []
        files = sorted(
            root.rglob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        max_results = max(limit, 0)
        return [str(p.relative_to(root)) for p in files[:max_results]]

    @modal.method()
    def health(self) -> dict:
        import requests
        r = requests.get(f"http://localhost:{VLLM_PORT}/health")
        return {"status": "ok", "vllm_status": r.status_code}

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "vllm_proc"):
            self.vllm_proc.terminate()
            try:
                self.vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_proc.kill()
                self.vllm_proc.wait(timeout=5)


@app.local_entrypoint()
def main():
    """Quick smoke test."""
    server = VibeVoiceServer()
    print("Checking health...")
    h = server.health.remote()
    print(f"Health: {h}")
    print("Server is ready for transcription requests.")