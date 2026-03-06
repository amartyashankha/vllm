#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

logger = init_logger(__name__)

HIDDEN_STATES_OUTPUT_DIR_ENV = "VLLM_HIDDEN_STATES_OUTPUT_DIR"


@dataclass
class _CaptureBuffers:
    capture_id: str
    request_id: str
    created_at_unix_s: float
    input_ids: list[int] | None
    hidden_chunks: list[torch.Tensor] = field(default_factory=list)
    aux_chunks_by_layer: list[list[torch.Tensor]] = field(default_factory=list)


class HiddenStateCaptureManager:
    """Incrementally capture request prefill hidden states and persist to disk."""

    def __init__(self, output_dir: str | None):
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.enabled = self.output_dir is not None
        self._buffers_by_req_id: dict[str, _CaptureBuffers] = {}

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Hidden-state capture enabled. Output directory: %s",
                self.output_dir,
            )

    @classmethod
    def from_env(cls) -> "HiddenStateCaptureManager":
        return cls(os.getenv(HIDDEN_STATES_OUTPUT_DIR_ENV))

    @staticmethod
    def _sanitize_capture_id(capture_id: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", capture_id).strip("._-")
        return cleaned or "capture"

    @classmethod
    def _get_capture_id(cls, req_state: "CachedRequestState") -> str | None:
        sampling_params = req_state.sampling_params
        if sampling_params is None:
            return None
        extra_args = sampling_params.extra_args or {}
        if not extra_args.get("capture_hidden_states", False):
            return None
        raw_capture_id = str(
            extra_args.get("capture_id")
            or extra_args.get("record_id")
            or req_state.req_id
        )
        return cls._sanitize_capture_id(raw_capture_id)

    def should_capture(self, req_state: "CachedRequestState") -> bool:
        return self.enabled and self._get_capture_id(req_state) is not None

    def _get_or_create_buffers(self, req_state: "CachedRequestState") -> _CaptureBuffers:
        capture_id = self._get_capture_id(req_state)
        if capture_id is None:
            raise ValueError("Capture was requested without a capture id.")

        buffer = self._buffers_by_req_id.get(req_state.req_id)
        if buffer is not None:
            return buffer

        buffer = _CaptureBuffers(
            capture_id=capture_id,
            request_id=req_state.req_id,
            created_at_unix_s=time.time(),
            input_ids=req_state.prompt_token_ids,
        )
        self._buffers_by_req_id[req_state.req_id] = buffer
        return buffer

    def append_prefill_chunk(
        self,
        req_state: "CachedRequestState",
        hidden_chunk: torch.Tensor,
        aux_hidden_chunks: list[torch.Tensor] | None,
    ) -> None:
        if not self.should_capture(req_state):
            return
        if hidden_chunk.numel() == 0:
            return

        buffer = self._get_or_create_buffers(req_state)
        buffer.hidden_chunks.append(hidden_chunk.detach().cpu().contiguous())

        if aux_hidden_chunks is None:
            return

        if not buffer.aux_chunks_by_layer:
            buffer.aux_chunks_by_layer = [[] for _ in aux_hidden_chunks]

        if len(buffer.aux_chunks_by_layer) != len(aux_hidden_chunks):
            logger.warning(
                "Aux hidden-state layer count changed for request %s: %d -> %d",
                req_state.req_id,
                len(buffer.aux_chunks_by_layer),
                len(aux_hidden_chunks),
            )
            return

        for layer_idx, aux_chunk in enumerate(aux_hidden_chunks):
            if aux_chunk.numel() == 0:
                continue
            buffer.aux_chunks_by_layer[layer_idx].append(
                aux_chunk.detach().cpu().contiguous()
            )

    def _output_path(self, capture_id: str) -> Path:
        assert self.output_dir is not None
        prefix = capture_id[:2] if len(capture_id) >= 2 else "00"
        return self.output_dir / prefix / f"{capture_id}.ckpt"

    def finalize_request(self, req_state: "CachedRequestState") -> Path | None:
        if not self.enabled:
            return None

        buffer = self._buffers_by_req_id.pop(req_state.req_id, None)
        if buffer is None or not buffer.hidden_chunks:
            return None

        hidden_state = torch.cat(buffer.hidden_chunks, dim=0).contiguous()

        aux_hidden_state = None
        if buffer.aux_chunks_by_layer:
            aux_layers = []
            for layer_chunks in buffer.aux_chunks_by_layer:
                if not layer_chunks:
                    continue
                aux_layers.append(torch.cat(layer_chunks, dim=0).contiguous())
            if aux_layers:
                aux_hidden_state = torch.cat(aux_layers, dim=-1).contiguous()

        seq_len = hidden_state.shape[0]
        prompt_ids = buffer.input_ids or []
        if prompt_ids:
            input_ids = torch.tensor(prompt_ids[:seq_len], dtype=torch.long)
        else:
            input_ids = torch.empty((seq_len,), dtype=torch.long)
        loss_mask = torch.zeros((seq_len,), dtype=torch.bool)

        payload: dict[str, object] = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden_state": hidden_state,
            "aux_hidden_state": aux_hidden_state,
            "metadata": {
                "capture_id": buffer.capture_id,
                "request_id": buffer.request_id,
                "created_at_unix_s": buffer.created_at_unix_s,
                "saved_at_unix_s": time.time(),
                "seq_len": seq_len,
                "aux_layer_count": len(buffer.aux_chunks_by_layer),
            },
        }

        output_path = self._output_path(buffer.capture_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(output_path))
        logger.info(
            "Saved hidden-state capture for request %s to %s",
            req_state.req_id,
            output_path,
        )
        return output_path

    def discard_request(self, req_id: str) -> None:
        self._buffers_by_req_id.pop(req_id, None)
