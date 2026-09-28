"""K1 of the flashback double backward: forward-mode JVP of attention, derived from FA4's SM90 forward.

Q-block outer loop over KV blocks with the forward LSE known (no online softmax). Per KV block:
  S = Q K^T,  Sdot' = Q gk^T + gq K^T,  P = exp2(S * scale_log2 - lse_log2),  W = c * P * Sdot'
  z1 += rowsum(W),  acc += W V + P gv
Epilogue: dot_dout = acc - z1 * O (written in the input dtype), z1 written to the fp32 padded
stats layout (tile_m_bwd padding, as the backward preprocess / K2 expect).
"""

import math
import operator
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90


class FlashbackJvpFwdSm90(FlashAttentionForwardSm90):
    def __init__(
        self,
        dtype,
        head_dim: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        tile_m_bwd: int,
        tile_m: int = 128,
        tile_n: int = 64,
        num_stages: int = 2,
    ):
        super().__init__(
            dtype,
            head_dim,
            head_dim,
            qhead_per_kvhead,
            is_causal,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=num_stages,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=False,
            mma_pv_is_rs=True,
        )
        # Padding granularity of the fp32 stats buffers (lse_log2, z1) shared with the backward.
        self.tile_m_bwd = tile_m_bwd

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageJvp:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sdQdot: sQ_struct
            sdKdot: sK_struct
            sdVdot: sV_struct

        return SharedStorageJvp

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d) or (total_k, h_k, d)
        mV: cute.Tensor,
        mdQdot: cute.Tensor,  # cotangent of dQ, like mQ
        mdKdot: cute.Tensor,  # cotangent of dK, like mK
        mdVdot: cute.Tensor,  # cotangent of dV, like mV
        mO: cute.Tensor,  # forward output, like mQ
        mLSElog2: cute.Tensor,  # (b, h, s_q_rounded) or (h, total_q_padded), fp32
        mdOdot: cute.Tensor,  # output, like mQ
        mZ1: cute.Tensor,  # output, like mLSElog2
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdOdot, mZ1, mCuSeqlensQ, mCuSeqlensK, None, None)
            )
        )
        assert mdQdot.element_type == mdKdot.element_type == mdVdot.element_type == self.dtype
        assert mO.element_type == self.dtype
        assert mLSElog2.element_type == Float32

        self.varlen_q = mCuSeqlensQ is not None

        mQ, mK, mV, mdQdot, mdKdot, mdVdot, mO, mdOdot = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdQdot, mdKdot, mdVdot, mO, mdOdot)
        ]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdQdot, mO, mdOdot = [
            layout_utils.select(t, QO_layout_transpose) for t in (mQ, mdQdot, mO, mdOdot)
        ]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV, mdKdot, mdVdot = [
            layout_utils.select(t, KV_layout_transpose) for t in (mK, mV, mdKdot, mdVdot)
        ]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSElog2, mZ1 = [layout_utils.select(t, LSE_layout_transpose) for t in (mLSElog2, mZ1)]

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma == 2
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = 240, 24
        self.use_block_sparsity = False
        self.use_scheduler_barrier = self.num_wg_mma == 2
        self.use_tma_Q = True
        self.use_tma_O = True
        self.rescale_O_before_gemm = False
        self._setup_attributes()
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mdOdot, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        SharedStorage = self._get_shared_storage_cls()

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mQ, self.sQ_layout, (self.tile_m, self.tile_hdim)
        )
        tma_atom_dQdot, tma_tensor_dQdot = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mdQdot, self.sQ_layout, (self.tile_m, self.tile_hdim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_dKdot, tma_tensor_dKdot = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mdKdot,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        tma_atom_dVdot, tma_tensor_dVdot = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mdVdot,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        mdOdot_tma = mdOdot
        if const_expr(self.varlen_q):
            mdOdot_tma = copy_utils.create_ragged_tensor_for_tma(
                mdOdot, ragged_dim=0, ptr_shift=True
            )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O, mdOdot_tma, self.sO_layout, (self.tile_m, self.tile_hdimv)
        )

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileLPTScheduler if const_expr(self.is_causal) else SingleTileScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dQdot,
            tma_tensor_dKdot,
            tma_tensor_dVdot,
            mO,
            mLSElog2,
            tma_tensor_O,
            mZ1,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dQdot,
            tma_atom_dKdot,
            tma_atom_dVdot,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQdot: cute.Tensor,
        mdKdot: cute.Tensor,
        mdVdot: cute.Tensor,
        mO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdOdot: cute.Tensor,
        mZ1: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dQdot: cute.CopyAtom,
        tma_atom_dKdot: cute.CopyAtom,
        tma_atom_dVdot: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dQdot,
                tma_atom_dKdot,
                tma_atom_dVdot,
                tma_atom_O,
            ):
                cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        # Each barrier covers a pair of TMA loads: (Q, dQdot), (K, dKdot), (V, dVdot).
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=2 * self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=2 * self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=2 * self.tma_copy_bytes["V"],
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdQdot = storage.sdQdot.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdKdot = storage.sdKdot.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdVdot = storage.sdVdot.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sdVdott = layout_utils.transpose_view(sdVdot)
        # reuse sQ's data iterator
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            False,  # is_local
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            # padded_offset_q must match the stats layout written by the bwd preprocess.
            tile_m=self.tile_m_bwd,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=None,
            window_size_right=None,
            qhead_per_kvhead_packgqa=1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                mdQdot,
                mdKdot,
                mdVdot,
                sQ,
                sK,
                sV,
                sdQdot,
                sdKdot,
                sdVdot,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dQdot,
                tma_atom_dKdot,
                tma_atom_dVdot,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSElog2,
                mdOdot,
                mZ1,
                sQ,
                sK,
                sVt,
                sdQdot,
                sdKdot,
                sdVdott,
                sO,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQdot: cute.Tensor,
        mdKdot: cute.Tensor,
        mdVdot: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdQdot: cute.Tensor,
        sdKdot: cute.Tensor,
        sdVdot: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dQdot: cute.CopyAtom,
        tma_atom_dKdot: cute.CopyAtom,
        tma_atom_dVdot: cute.CopyAtom,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = head_idx // self.qhead_per_kvhead

                gQ, gdQdot = [
                    cute.local_tile(
                        seqlen.offset_batch_Q(mX, batch_idx, dim=3)[None, None, head_idx],
                        (self.tile_m, self.tile_hdim),
                        (m_block, 0),
                    )
                    for mX in (mQ, mdQdot)
                ]
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                load_dQdot, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dQdot, 0, cute.make_layout(1), gdQdot, sdQdot, single_stage=True
                )
                gK, gdKdot = [
                    cute.local_tile(
                        seqlen.offset_batch_K(mX, batch_idx, dim=3)[None, None, head_idx_kv],
                        (self.tile_n, self.tile_hdim),
                        (None, 0),
                    )
                    for mX in (mK, mdKdot)
                ]
                gV, gdVdot = [
                    cute.local_tile(
                        seqlen.offset_batch_K(mX, batch_idx, dim=3)[None, None, head_idx_kv],
                        (self.tile_n, self.tile_hdimv),
                        (None, 0),
                    )
                    for mX in (mV, mdVdot)
                ]
                load_K, load_dKdot, load_V, load_dVdot = [
                    copy_utils.tma_producer_copy_fn(
                        copy_utils.tma_get_copy_fn(atom, 0, cute.make_layout(1), gX, sX)[0],
                        pipeline_x,
                    )
                    for atom, gX, sX, pipeline_x in (
                        (tma_atom_K, gK, sK, pipeline_k),
                        (tma_atom_dKdot, gdKdot, sdKdot, pipeline_k),
                        (tma_atom_V, gV, sV, pipeline_v),
                        (tma_atom_dVdot, gdVdot, sdVdot, pipeline_v),
                    )
                ]

                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                # TMA handles n_block = -1 (fully masked Q tile) by filling zeros.
                n_block = n_block_max - 1
                # First iteration: K (+ dKdot), then Q (+ dQdot), then V (+ dVdot)
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                load_dKdot(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_k.producer_commit(kv_producer_state)
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                load_dQdot(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                q_producer_phase ^= 1
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                load_dVdot(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_commit(kv_producer_state)
                kv_producer_state.advance()
                for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - i - 1
                    pipeline_k.producer_acquire(kv_producer_state)
                    load_K(src_idx=n_block, producer_state=kv_producer_state)
                    load_dKdot(src_idx=n_block, producer_state=kv_producer_state)
                    pipeline_k.producer_commit(kv_producer_state)
                    pipeline_v.producer_acquire(kv_producer_state)
                    load_V(src_idx=n_block, producer_state=kv_producer_state)
                    load_dVdot(src_idx=n_block, producer_state=kv_producer_state)
                    pipeline_v.producer_commit(kv_producer_state)
                    kv_producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdOdot: cute.Tensor,
        mZ1: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sdQdot: cute.Tensor,
        sdKdot: cute.Tensor,
        sdVdott: cute.Tensor,
        sO: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        tma_atom_O: cute.CopyAtom,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(wg_mma_qk, shape_mnk_S, sQ, sK)
        _, _, tSrdKdot = sm90_utils.partition_fragment_ABC(wg_mma_qk, shape_mnk_S, sQ, sdKdot)
        _, tSrdQdot, _ = sm90_utils.partition_fragment_ABC(wg_mma_qk, shape_mnk_S, sdQdot, sK)
        # S = Q K^T
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        # Sdot' = Q gk^T + gq K^T
        mma_qdk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrdKdot
        )
        mma_dqk_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_qk, tCrA=tSrdQdot, tCrB=tSrK)
        shape_mnk_O = (self.tile_m, self.tile_hdimv, self.tile_n)
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(wg_mma_pv, shape_mnk_O, None, sVt)
        _, _, tOrdVdott = sm90_utils.partition_fragment_ABC(wg_mma_pv, shape_mnk_O, None, sdVdott)
        tOrW = cute.make_rmem_tensor_like(tOrP)
        # acc_O += W V + P gv
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tCrB=tOrVt)
        mma_pdv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tCrB=tOrdVdott)

        num_rows = acc_O.shape[0][0] * acc_O.shape[1]
        z1 = cute.make_rmem_tensor(num_rows, Float32)
        tLSErLSE = cute.make_rmem_tensor(num_rows, Float32)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        assert cute.size(tScS_mn, mode=[0]) == num_rows

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        mma_one_n_block = partial(
            self.mma_one_n_block_jvp,
            mma_qk_fn=mma_qk_fn,
            mma_qdk_fn=mma_qdk_fn,
            mma_dqk_fn=mma_dqk_fn,
            mma_pv_fn=mma_pv_fn,
            mma_pdv_fn=mma_pdv_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            tOrP=tOrP,
            tOrW=tOrW,
            tLSErLSE=tLSErLSE,
            z1=z1,
            softmax_scale_log2=softmax_scale_log2,
            softmax_scale=softmax_scale,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # Row stats: lse_log2 for valid rows, +inf (P = 0) past seqlen_q. The 128-row tile can
            # run past the end of the 64-row-padded stats buffer, so the predicate is required.
            mLSE_cur = seqlen.offset_batch_Q(mLSElog2, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSE_expanded = cute.make_tensor(
                gLSE.iterator,
                cute.append(gLSE.layout, cute.make_layout((self.tile_n,), stride=(0,))),
            )
            tLSEgLSE = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(gLSE_expanded))
            for r in cutlass.range(num_rows, unroll_full=True):
                tLSErLSE[r] = Float32.inf
                if tScS_mn[r, 0][0] < seqlen.seqlen_q - m_block * self.tile_m:
                    tLSErLSE[r] = tLSEgLSE[r, 0]
            z1.fill(0.0)
            acc_O.fill(0.0)

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=False,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            # First iteration with seqlen masking
            self.warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block_max - 1,
                mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
            )
            n_block_max -= 1
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                    )
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            # The remaining iterations have no masking
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=n_block_max - 1 - n_tile,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
            # Release Q pipeline so the producer can load the next tile's Q
            pipeline_q.consumer_release_w_index(0)
            self.warp_scheduler_barrier_arrive()
            q_consumer_phase ^= 1

            self.epilogue_jvp(
                acc_O,
                z1,
                mO,
                mdOdot,
                mZ1,
                sO,
                seqlen,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_n_block_jvp(
        self,
        smem_pipe_read: pipeline.PipelineState,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_qdk_fn: Callable,
        mma_dqk_fn: Callable,
        mma_pv_fn: Callable,
        mma_pdv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        tOrP: cute.Tensor,
        tOrW: cute.Tensor,
        tLSErLSE: cute.Tensor,
        z1: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mask_fn: Optional[Callable] = None,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        # S = Q K^T ; Sdot' = Q gk^T + gq K^T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        acc_Sdot = mma_qdk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        mma_dqk_fn(acc=acc_Sdot, zero_init=False, B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_Sdot_mn = layout_utils.reshape_acc_to_mn(acc_Sdot)
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            lse = tLSErLSE[r]
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                p = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse, fastmath=True)
                acc_S_mn[r, c] = p
                acc_Sdot_mn[r, c] = p * acc_Sdot_mn[r, c] * softmax_scale  # W = P * Sdot
            z1[r] = utils.fadd_reduce(acc_Sdot_mn[r, None].load(), init_val=z1[r], arch=90)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_Sdot), tOrW)

        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # acc_O += W V + P gv
        mma_pv_fn(tCrA=tOrW, zero_init=False, B_idx=smem_pipe_read.index, wg_wait=-1)
        mma_pdv_fn(tCrA=tOrP, zero_init=False, B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def epilogue_jvp(
        self,
        acc_O: cute.Tensor,
        z1: cute.Tensor,
        mO: cute.Tensor,
        mdOdot: cute.Tensor,
        mZ1: cute.Tensor,
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # quad reduction for z1 (each row's columns are spread over 4 threads)
        z1.store(utils.warp_reduce(z1.load(), operator.add, width=4))

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
        row_limit = seqlen.seqlen_q - m_block * self.tile_m

        # dot_dout = acc_O - z1 * O  (rows of acc_S and acc_O coincide per thread)
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        tOgO = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(gO))
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(acc_O_mn, mode=[0]) == cute.size(z1)
        for r in cutlass.range(cute.size(acc_O_mn, mode=[0]), unroll_full=True):
            if tOcO[r, 0][0] < row_limit:
                o_row = tOgO[r, None].load().to(Float32)
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() - z1[r] * o_row)

        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Make sure all threads have finished reading sQ (aliased by sO) and V
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        # Write z1 (natural units, padded stats layout); only the column-0 thread of each quad.
        mZ1_cur = seqlen.offset_batch_Q(mZ1, batch_idx, dim=2, padded=True)[None, head_idx]
        gZ1 = cute.local_tile(mZ1_cur, (self.tile_m,), (m_block,))
        gZ1_expanded = cute.make_tensor(
            gZ1.iterator,
            cute.append(gZ1.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))),
        )
        taccOgZ1 = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(gZ1_expanded))
        if tOcO[0][1] == 0:
            for r in cutlass.range(cute.size(taccOgZ1, mode=[0]), unroll_full=True):
                if tOcO[r, 0][0] < row_limit:
                    taccOgZ1[r, 0] = z1[r]

        ragged = seqlen.has_cu_seqlens_q
        mdOdot_cur = seqlen.offset_batch_Q(mdOdot, batch_idx, dim=3, ragged=ragged)[
            None, None, head_idx
        ]
        # ensure smem writes are visible to TMA
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )
        gdOdot = cute.local_tile(mdOdot_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        store_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_O, 0, cute.make_layout(1), sO, gdOdot, single_stage=True
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
