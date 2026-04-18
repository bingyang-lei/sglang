"""DFlash Student speculative worker.

Supports two decode modes:
  - `no-verify`: always-accept block generation.
  - `verify`: standard target-model verification after each drafted block.
"""

import logging
from copy import deepcopy
from typing import Any, List, Optional, Union, cast

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_student_info import (
    DFlashStudentDraftInput,
    DFlashStudentVerifyInput,
)
from sglang.srt.speculative.dflash_utils import (
    parse_dflash_draft_config,
    scale_kv_cell_size_per_token_for_dflash,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DFlashStudentWorker:
    """DFlash Student speculative decoding worker (tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        assert server_args.page_size is not None
        self.page_size = int(server_args.page_size)
        self.device = target_worker.device

        # Draft runner shares the target's KV pool allocator.
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "fa3"
        elif draft_backend not in supported_draft_backends:
            logger.warning(
                "DFLASH_STUDENT draft worker only supports attention_backend in %s, "
                "but got %r. Falling back to 'flashinfer'.",
                supported_draft_backends,
                draft_backend,
            )
            draft_backend = "fa3"
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=target_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        setattr(self.draft_worker, "draft_runner", self.draft_model_runner)
        self.draft_model = cast(Any, self.draft_model_runner.model)

        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            resolved_block_size = draft_config.resolve_block_size(default=16)
            assert resolved_block_size is not None
            self.block_size = int(resolved_block_size)
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = int(self.block_size)
        self.dflash_student_mode = str(server_args.dflash_student_mode)

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH_STUDENT draft runner. attention_backend=%s, model=%s, "
                "block_size=%s, mask_token_id=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self._mask_token_id,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None
        self._draft_block_positions_buf: Optional[torch.Tensor] = None
        self._draft_block_tokens_buf: Optional[torch.Tensor] = None
        self._draft_block_end_buf: Optional[torch.Tensor] = None
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        pass

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return
        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    f"DFLASH_STUDENT mask_token_id={resolved_id} >= vocab_size={vocab_size}."
                )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH_STUDENT requires tokenizer when mask_token_id is not set in config."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)
            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH_STUDENT mask token to tokenizer. token=%s, id=%s",
                    mask_token,
                    resolved_id,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                f"DFLASH_STUDENT could not resolve mask_token={mask_token!r}."
            )
        if resolved_id >= vocab_size:
            raise ValueError(
                f"DFLASH_STUDENT mask_token_id={resolved_id} >= vocab_size={vocab_size}."
            )
        return int(resolved_id)

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashStudentDraftInput,
    ) -> None:
        """Materialize target hidden-state features into the draft KV cache."""
        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError("DFLASH_STUDENT missing target_hidden context features.")

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)
            ctx_positions = pos2d.reshape(-1)
        else:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH_STUDENT invalid max_ctx={max_ctx}.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]
            pos2d = ctx_start[:, None] + r
            mask = r < ctx_lens[:, None]

            safe_pos2d = pos2d.masked_fill(~mask, 0)
            ctx_cache_loc = target_req_to_token[
                req_pool_indices[:, None], safe_pos2d
            ][mask].to(torch.int64)
            ctx_positions = pos2d[mask]

        with torch.inference_mode():
            draft_model = cast(Any, self.draft_model)
            ctx_hidden = draft_model.project_target_hidden(
                draft_input.target_hidden
            )

            for layer in cast(Any, draft_model.layers):
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    ctx_cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

        draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way."""
        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,), torch.finfo(weight_dtype).min,
                    dtype=weight_dtype, device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            if num_added > 0:
                added_logits = torch.matmul(
                    hs, weight[num_org_padded : num_org_padded + num_added].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            gathered_max = torch.empty(
                (tp_size * chunk_len,), dtype=local_max.dtype, device=hs.device
            )
            gathered_ids = torch.empty(
                (tp_size * chunk_len,), dtype=global_ids.dtype, device=hs.device
            )
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)
            best_rank = torch.argmax(gathered_max, dim=0)
            rank_index = best_rank.unsqueeze(0)
            selected_ids = torch.gather(gathered_ids, 0, rank_index)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids

    def _add_no_verify_output_logprobs(
        self,
        *,
        batch: ScheduleBatch,
        draft_tokens: torch.Tensor,
        commit_lens_cpu: List[int],
        logits_output: LogitsProcessorOutput,
    ) -> None:
        """Attach per-token logprobs for the committed no-verify block.

        In the no-verify path, committed tokens come from `draft_tokens`, and the
        target extend pass provides their scores as `input_token_logprobs`.
        """
        input_token_logprobs = logits_output.input_token_logprobs
        if input_token_logprobs is None:
            return

        flat_logprobs = input_token_logprobs.tolist()
        top_vals = logits_output.input_top_logprobs_val
        top_idxs = logits_output.input_top_logprobs_idx
        token_ids_vals = logits_output.input_token_ids_logprobs_val
        token_ids_idxs = logits_output.input_token_ids_logprobs_idx
        draft_tokens_cpu = draft_tokens.cpu().tolist()

        pt = 0
        for i, (req, commit_len) in enumerate(zip(batch.reqs, commit_lens_cpu, strict=True)):
            for j in range(int(self.block_size)):
                if req.return_logprob and j < commit_len:
                    assert req.output_token_logprobs_val is not None
                    assert req.output_token_logprobs_idx is not None
                    req.output_token_logprobs_val.append(flat_logprobs[pt])
                    req.output_token_logprobs_idx.append(int(draft_tokens_cpu[i][j]))
                    if req.top_logprobs_num > 0 and top_vals is not None and top_idxs is not None:
                        assert req.output_top_logprobs_val is not None
                        assert req.output_top_logprobs_idx is not None
                        req.output_top_logprobs_val.append(top_vals[i][j])
                        req.output_top_logprobs_idx.append(top_idxs[i][j])
                    if req.token_ids_logprob is not None and token_ids_vals is not None and token_ids_idxs is not None:
                        assert req.output_token_ids_logprobs_val is not None
                        assert req.output_token_ids_logprobs_idx is not None
                        cast(Any, req.output_token_ids_logprobs_val).append(
                            cast(Any, token_ids_vals[i][j])
                        )
                        cast(Any, req.output_token_ids_logprobs_idx).append(
                            cast(Any, token_ids_idxs[i][j])
                        )
                pt += 1

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        # ---- Extend (prefill) ----
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            assert isinstance(logits_output, LogitsProcessorOutput)
            assert isinstance(next_token_ids, torch.Tensor)
            hidden_states = logits_output.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "DFLASH_STUDENT requires target aux hidden capture for prefill, but got None."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH_STUDENT expected extend_seq_lens / extend_prefix_lens in extend mode."
                )

            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DFlashStudentDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=_to_int32_device_tensor(
                    model_worker_batch.extend_prefix_lens
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # ---- Decode: always-accept block generation ----
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashStudentDraftInput):
            raise RuntimeError(
                "DFLASH_STUDENT decode requires DFlashStudentDraftInput on the running batch."
            )

        bs = batch.batch_size()

        # Step 1: Append committed target hidden into draft KV cache
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = cast(Any, self.target_worker.model_runner.model)
        embed_module = cast(Any, target_model.get_input_embeddings())
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH_STUDENT requires the target model to expose a vocab-parallel lm_head."
            )

        # Step 2: Build draft block input
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        target_prefix_lens = batch.seq_lens
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )
        positions = positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        assert allocator is not None
        assert self.draft_model_runner.req_to_token_pool is not None
        token_to_kv_pool_state_backup = allocator.backup_state()

        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH_STUDENT draft OOM: {bs * self.block_size} tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            seq_lens = draft_prefix_lens
            total_seq_lens = seq_lens + int(self.block_size)
            total_seq_lens_cpu = seq_lens_cpu + int(self.block_size)
            seq_lens_sum = int(draft_prefix_lens.sum().item())
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=total_seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=int(total_seq_lens.sum().item()),
                seq_lens_cpu=total_seq_lens_cpu,
                positions=positions,
                extend_num_tokens=bs * int(self.block_size),
                extend_seq_lens=torch.full(
                    (bs,),
                    int(self.block_size),
                    dtype=torch.int32,
                    device=self.device,
                ),
                extend_prefix_lens=seq_lens,
                extend_seq_lens_cpu=[int(self.block_size)] * bs,
                extend_prefix_lens_cpu=seq_lens_cpu.tolist(),
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH_STUDENT,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            allocator.restore_state(token_to_kv_pool_state_backup)

        assert isinstance(draft_logits_output, LogitsProcessorOutput)
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH_STUDENT draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)

        # Step 3: Sample block_size tokens from draft hidden via target lm_head
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, self.block_size - 1)
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        if self.dflash_student_mode == "verify":
            verify_input = DFlashStudentVerifyInput(
                draft_token=draft_tokens.reshape(-1),
                positions=positions,
                draft_token_num=int(self.block_size),
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            verify_input.prepare_for_verify(batch, self.page_size)
            batch.forward_mode = (
                ForwardMode.TARGET_VERIFY
                if not batch.forward_mode.is_idle()
                else ForwardMode.IDLE
            )
            batch.spec_info = verify_input
            batch.return_hidden_states = False

            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True, **kwargs
            )
            target_logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            assert isinstance(target_logits_output, LogitsProcessorOutput)

            (
                new_verified_id,
                commit_lens,
                next_target_hidden,
                accept_length_per_req_cpu,
            ) = verify_input.verify(
                batch=batch,
                logits_output=target_logits_output,
                page_size=self.page_size,
            )

            draft_input.verified_id = new_verified_id
            draft_input.target_hidden = next_target_hidden
            draft_input.ctx_lens = commit_lens
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            batch.forward_mode = ForwardMode.DECODE

            num_accepted_tokens = sum(accept_length_per_req_cpu)
            return GenerationBatchResult(
                logits_output=target_logits_output,
                next_token_ids=new_verified_id,
                num_accepted_tokens=num_accepted_tokens,
                accept_length_per_req_cpu=accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

        # Step 4: Run target model on draft tokens to get new hidden states
        # Use extend mode on the target for the drafted tokens
        target_input_ids = draft_tokens.reshape(-1)
        target_positions = positions_2d.reshape(-1)

        # Allocate target KV cache for the block_size tokens
        if self.page_size == 1:
            target_cache_loc = alloc_token_slots(
                batch.tree_cache, bs * self.block_size
            )
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.block_size
            end_offset_cpu = prefix_lens_cpu + self.block_size
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            target_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                bs * self.block_size,
            )
        assert isinstance(target_cache_loc, torch.Tensor)

        end_offset = batch.seq_lens + self.block_size
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            target_cache_loc,
            bs,
        )

        target_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=target_input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens + int(self.block_size),
            out_cache_loc=target_cache_loc,
            seq_lens_sum=int((batch.seq_lens + int(self.block_size)).sum().item()),
            seq_lens_cpu=batch.seq_lens_cpu + int(self.block_size),
            positions=target_positions,
            extend_num_tokens=bs * int(self.block_size),
            extend_seq_lens=torch.full(
                (bs,),
                int(self.block_size),
                dtype=torch.int32,
                device=self.device,
            ),
            extend_prefix_lens=batch.seq_lens,
            extend_seq_lens_cpu=[int(self.block_size)] * bs,
            extend_prefix_lens_cpu=batch.seq_lens_cpu.tolist(),
            extend_logprob_start_lens_cpu=[0] * bs,
            extend_input_logprob_token_ids_gpu=target_input_ids,
            req_to_token_pool=batch.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            sampling_info=batch.sampling_info,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            spec_algorithm=SpeculativeAlgorithm.DFLASH_STUDENT,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        with torch.inference_mode():
            target_result = self.model_runner.forward(target_forward_batch)
        target_logits_output = target_result.logits_output
        assert isinstance(target_logits_output, LogitsProcessorOutput)
        with torch.inference_mode():
            new_verified_id = self.model_runner.sample(
                target_logits_output, target_forward_batch
            ).to(torch.int64)

        # Step 5: Accept ALL tokens unconditionally
        accept_length_per_req_cpu: List[int] = []
        for i, req in enumerate(batch.reqs):
            tokens_cpu = draft_tokens[i].cpu().tolist()
            appended = 0
            for token_id in tokens_cpu:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                appended += 1
                req.check_finished()
                if req.finished():
                    break
                if req.grammar is not None:
                    req.grammar.accept_token(token_id)

            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]

        # Compute commit lengths (how many tokens actually accepted per request)
        commit_lens_cpu = [max(0, a + 1) for a in accept_length_per_req_cpu]
        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=self.device)

        if batch.return_logprob:
            self._add_no_verify_output_logprobs(
                batch=batch,
                draft_tokens=draft_tokens,
                commit_lens_cpu=commit_lens_cpu,
                logits_output=target_logits_output,
            )

        # Free uncommitted KV cache slots
        out_cache_loc = target_cache_loc.view(bs, self.block_size)
        if self.page_size == 1:
            keep_mask = (
                torch.arange(self.block_size, device=self.device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            row_offsets = torch.arange(self.block_size, device=self.device)[None, :]
            free_mask = row_offsets >= commit_lens[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])
            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # Update req-level KV accounting
        for req, cl in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += cl
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool + seq lens
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from committed target hidden
        hidden = target_logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH_STUDENT requires target hidden states from target prefill."
            )
        hidden = hidden.view(bs, self.block_size, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # Update draft state for next iteration.
        # `new_verified_id` is the target posterior token after the committed block;
        # it anchors the next draft block but is not committed into target KV yet.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)

        target_logits_output.hidden_states = None

        return GenerationBatchResult(
            logits_output=target_logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=False,
        )

    def update_weights_from_distributed(self, recv_req):
        """Rewrite DFlash Student param names and update the draft worker only."""
        rewritten_names = []
        for name in recv_req.names:
            if name.startswith("model.dflash.layers."):
                rewritten_names.append("layers." + name[len("model.dflash.layers.") :])
            elif name.startswith("model.dflash.fc."):
                rewritten_names.append("fc." + name[len("model.dflash.fc.") :])
            elif name.startswith("model.dflash.norm."):
                rewritten_names.append("norm." + name[len("model.dflash.norm.") :])
            elif name.startswith("model.dflash.hidden_norm."):
                rewritten_names.append(
                    "hidden_norm." + name[len("model.dflash.hidden_norm.") :]
                )
            elif name.startswith("dflash.layers."):
                rewritten_names.append("layers." + name[len("dflash.layers.") :])
            elif name.startswith("dflash.fc."):
                rewritten_names.append("fc." + name[len("dflash.fc.") :])
            elif name.startswith("dflash.norm."):
                rewritten_names.append("norm." + name[len("dflash.norm.") :])
            elif name.startswith("dflash.hidden_norm."):
                rewritten_names.append(
                    "hidden_norm." + name[len("dflash.hidden_norm.") :]
                )
            else:
                rewritten_names.append(name)

        recv_req.names = rewritten_names
        return self.draft_worker.update_weights_from_distributed(recv_req)

    def init_weights_update_group(self, recv_req):
        """Initialize online weight updates for the draft worker."""
        return self.draft_worker.init_weights_update_group(recv_req)

    def destroy_weights_update_group(self, recv_req):
        """Destroy online weight updates for the draft worker."""
        return self.draft_worker.destroy_weights_update_group(recv_req)

    def update_weights_from_tensor(self, recv_req):
        """Forward tensor-based updates to the draft worker."""
        return self.draft_worker.update_weights_from_tensor(recv_req)
