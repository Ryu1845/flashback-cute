"""K2 of the flashback double backward, derived from FA4's SM90 backward (KV-block outer).

Per (n_block, m_block) tile, with c = softmax_scale:
  S = Q K^T,  Sdot' = Q gk^T + gq K^T,  dP = dO V^T,  E = dO gv^T
  P = exp2(S * scale_log2 - lse_log2),  Pdot = P (c Sdot' - z1)
  dS = P (dP - D),  dSdot = Pdot (dP - D) + P (E - Ddot)
  dot_v += Pdot^T dO,  dot_k += dSdot^T Q + dS^T gq,  dot_q (fp32 accum) += dSdot K + dS gk
dot_k and dot_q are scaled by c in the epilogue / postprocess exactly like FA4's dK / dQ.

Two MMA layouts:
  SS dK/dV (hdim 128): one warpgroup computes S and Pdot, the other dP and dS.
    P is parked in sdS until both warpgroups synchronize; Pdot occupies sP. The
    dot_v / dot_k GEMMs read shared A operands. P and Pdot are rounded to bf16
    before the second warpgroup computes dS and dSdot.
  RS dK/dV (hdim 64): transposed S-like accumulators feed Pdot^T / dS^T /
    dSdot^T directly to dot_v / dot_k. Row statistics are loaded from shared
    memory in pairs rather than shuffled between lanes. Each warpgroup owns half
    the key rows and accumulates a full-width dot_q partial over them, so the
    warpgroups share only the Q / dO pipeline and ping-pong on the tensor cores.
Only the causal diagonal and the last, partial n_block run apply_mask (mask_split).
"""

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisorV2
from cutlass import Float32, Int32, Boolean, const_expr

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
from quack.sm90_utils import gemm_zero_init, gemm_w_idx
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
)
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90

# Per-warpgroup dS / dSdot hand-off barriers (ids after FA4's NamedBarrierBwd, which ends at 11).
PDS_WG0_BARRIER_ID = int(NamedBarrierBwd.dQEmptyWG2) + 1


@cute.jit
def _gemm_k_range(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    k_start: Int32,
    num_k: cutlass.Constexpr[int],
    zero_init: cutlass.Constexpr[bool],
    wg_wait: cutlass.Constexpr[int],
):
    """quack's sm90 gemm restricted to K blocks [k_start, k_start + num_k)."""
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(num_k):
        cute.gemm(mma_atom, acc, tCrA[None, None, k_start + k], tCrB[None, None, k_start + k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


def gemm_k_range_w_idx(
    tiled_mma, acc, tCrA, tCrB, zero_init, k_start, num_k, A_idx=None, B_idx=None, wg_wait=-1
):
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    _gemm_k_range(tiled_mma, acc, rA, rB, k_start, num_k, zero_init, wg_wait)


def gemm_k_range_zero_init(
    tiled_mma, shape, tCrA, tCrB, k_start, num_k, A_idx=None, B_idx=None, wg_wait=-1
):
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
    gemm_k_range_w_idx(tiled_mma, acc, tCrA, tCrB, True, k_start, num_k, A_idx, B_idx, wg_wait)
    return acc


class FlashbackHvpBwdSm90(FlashAttentionBackwardSm90):
    def __init__(
        self,
        dtype,
        head_dim: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        tile_m: int = 64,
        tile_n: int = 64,
        Q_stage: int = 2,
        dO_stage: Optional[int] = None,
        PdS_stage: int = 1,
        SdP_swapAB: bool = False,
        AtomLayoutNdKV: int = 1,
        mask_split: str = "loops",
    ):
        super().__init__(
            dtype,
            head_dim,
            head_dim,
            qhead_per_kvhead,
            is_causal,
            is_local=False,
            deterministic=False,
            tile_m=tile_m,
            tile_n=tile_n,
            Q_stage=Q_stage,
            dO_stage=Q_stage if dO_stage is None else dO_stage,
            PdS_stage=PdS_stage,
            SdP_swapAB=SdP_swapAB,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=AtomLayoutNdKV,
            AtomLayoutMdQ=1,
            num_threads=384,
        )
        assert self.num_wg_dQ == self.num_wg_mma == 2
        assert self.mma_dkv_is_rs or (
            not self.SdP_swapAB and self.PdS_stage == self.Q_stage == 2
        )
        # RS path: each warpgroup computes a full-width dot_q partial over its own key rows, so
        # the warpgroups share only the Q / dO pipeline and take turns on the tensor cores.
        # mask_split: "loops" runs the unmasked m_blocks in a mask-free copy of the block (FA3's
        # SeparateMaskingIterations); "branch" keeps one copy with apply_mask behind a branch.
        assert mask_split in ("loops", "branch")
        self.mask_split = mask_split
        self.spt = False
        self.use_block_sparsity = False

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]
        sPdS_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024
        ]
        # RS dK/dV feeds Pdot^T from registers: no sP.
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]

        # FA4 fields first, in FA4's order: the GQA epilogue stages fp32 dK/dV across the
        # contiguous sV -> sK region.
        @cute.struct
        class SharedStorageHvp:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: sP_struct
            sdS: sPdS_struct
            sdQaccum: sdQaccum_struct
            sZ1: sLSE_struct
            sDdot: sdPsum_struct
            sdQdot: sQ_struct
            sdKdot: sK_struct
            sdVdot: sV_struct
            sdSdot: sPdS_struct

        return SharedStorageHvp

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdQdot: cute.Tensor,
        mdKdot: cute.Tensor,
        mdVdot: cute.Tensor,
        mLSE: cute.Tensor,  # lse_log2, padded stats layout
        mdPsum: cute.Tensor,  # D
        mZ1: cute.Tensor,
        mDdot: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,  # dot_k (MHA) or fp32 accum (GQA)
        mdV: cute.Tensor,  # dot_v (MHA) or fp32 accum (GQA)
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.varlen_k = mCuSeqlensK is not None
        self._check_type(
            *(t.element_type for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV))
        )
        assert mdQdot.element_type == mdKdot.element_type == mdVdot.element_type == self.dtype
        assert mZ1.element_type == mDdot.element_type == Float32
        self.is_varlen_q = mCuSeqlensQ is not None

        mQ, mK, mV, mdO, mdQdot, mdKdot, mdVdot = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdQdot, mdKdot, mdVdot)
        ]
        mLSE, mdPsum, mZ1, mDdot, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mLSE, mdPsum, mZ1, mDdot, mdQaccum, mdK, mdV)
        ]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO, mdQdot, mdKdot, mdVdot = [
            _qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdQdot, mdKdot, mdVdot)
        ]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        stats_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mZ1, mDdot, mdQaccum = [
            layout_utils.select(t, stats_transpose) for t in (mLSE, mdPsum, mZ1, mDdot, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        if const_expr(self.mma_dkv_is_rs):
            # Per-warpgroup dot_q partials: one warpgroup's (tile_m, hdim) tile each.
            tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype, self.dtype, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
                Float32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.tile_hdim),
            )
        tiled_mma_S1 = None
        if const_expr(not self.mma_dkv_is_rs):
            tiled_mma_S1 = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype, self.dtype, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
                Float32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.tile_n),
            )
        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads
        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32
        self.num_mma_regs_wg0 = 240
        self.num_mma_regs_wg1 = 240
        self.num_mma_regs = self.num_mma_regs_wg0
        self.num_producer_regs = 24

        self._setup_attributes()
        if const_expr(self.mma_dkv_is_rs):
            # One full (tile_m, hdim) dot_q partial per warpgroup, both added to the same gmem tile.
            self.sdQaccum_layout = cute.make_layout(
                (self.tile_m * self.tile_hdim, self.num_wg_mma)
            )
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
                ("dQdot", mdQdot, self.sQ_layout),
                ("dKdot", mdKdot, self.sK_layout),
                ("dVdot", mdVdot, self.sV_layout),
            ]
        }
        for name in ("LSE", "dPsum", "Z1", "Ddot"):
            self.tma_copy_bytes[name] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = cute.size_in_bytes(
            Float32, cute.select(self.sdQaccum_layout, mode=[0])
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        make_load_atom = partial(cpasync.make_tiled_tma_atom, cpasync.CopyBulkTensorTileG2SOp())
        Q_like = (cute.select(self.sQ_layout, mode=[0, 1]), (self.tile_m, self.tile_hdim))
        K_like = (cute.select(self.sK_layout, mode=[0, 1]), (self.tile_n, self.tile_hdim))
        V_like = (cute.select(self.sV_layout, mode=[0, 1]), (self.tile_n, self.tile_hdimv))
        dO_like = (cute.select(self.sdO_layout, mode=[0, 1]), (self.tile_m, self.tile_hdimv))
        tma_atom_Q, tma_tensor_Q = make_load_atom(mQ, *Q_like)
        tma_atom_K, tma_tensor_K = make_load_atom(mK, *K_like)
        tma_atom_V, tma_tensor_V = make_load_atom(mV, *V_like)
        tma_atom_dO, tma_tensor_dO = make_load_atom(mdO, *dO_like)
        tma_atom_dQdot, tma_tensor_dQdot = make_load_atom(mdQdot, *Q_like)
        tma_atom_dKdot, tma_tensor_dKdot = make_load_atom(mdKdot, *K_like)
        tma_atom_dVdot, tma_tensor_dVdot = make_load_atom(mdVdot, *V_like)
        if const_expr(self.qhead_per_kvhead == 1):
            mdK_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdK
            )
            mdV_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdV
            )
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mdK_tma, *K_like
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mdV_tma, *V_like
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(mCuSeqlensK is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batch
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=False,
            head_swizzle=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2 = softmax_scale * math.log2(math.e)
        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dQdot,
            tma_tensor_dKdot,
            tma_tensor_dVdot,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dQdot,
            tma_atom_dKdot,
            tma_atom_dVdot,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mZ1,
            mDdot,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            tiled_mma_S1,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdQdot: cute.Tensor,
        mdKdot: cute.Tensor,
        mdVdot: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQdot: cute.CopyAtom,
        tma_atom_dKdot: cute.CopyAtom,
        tma_atom_dVdot: cute.CopyAtom,
        tma_atom_dK: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mZ1: cute.Tensor,
        mDdot: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_S1: Optional[cute.TiledMma],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dQdot,
                tma_atom_dKdot,
                tma_atom_dVdot,
                tma_atom_dK,
                tma_atom_dV,
            ]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"]
            + self.tma_copy_bytes["dQdot"]
            + self.tma_copy_bytes["LSE"]
            + self.tma_copy_bytes["Z1"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"]
            + self.tma_copy_bytes["dPsum"]
            + self.tma_copy_bytes["Ddot"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdQdot = storage.sdQdot.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sdKdot = storage.sdKdot.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdVdot = storage.sdVdot.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdSdot = storage.sdSdot.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        stats_layout_Q = cute.make_layout(
            (self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        stats_layout_dO = cute.make_layout(
            (self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        sLSE = storage.sLSE.get_tensor(stats_layout_Q)
        sZ1 = storage.sZ1.get_tensor(stats_layout_Q)
        sdPsum = storage.sdPsum.get_tensor(stats_layout_dO)
        sDdot = storage.sDdot.get_tensor(stats_layout_dO)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

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
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=None,
            window_size_right=None,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mdQdot,
                    mdKdot,
                    mdVdot,
                    mLSE,
                    mdPsum,
                    mZ1,
                    mDdot,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sdQdot,
                    sdKdot,
                    sdVdot,
                    sLSE,
                    sdPsum,
                    sZ1,
                    sDdot,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    tma_atom_dQdot,
                    tma_atom_dKdot,
                    tma_atom_dVdot,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    qhead_per_kvhead_divmod,
                )
            if warp_idx == 1:
                self.dQaccum_store(mdQaccum, sdQaccum, block_info, TileSchedulerCls, SeqlenInfoCls)
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
            self.mma(
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                tiled_mma_S1,
                mdK,
                mdV,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                sdQdot,
                sdKdot,
                sdVdot,
                sdSdot,
                sZ1,
                sDdot,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdQdot: cute.Tensor,
        mdKdot: cute.Tensor,
        mdVdot: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mZ1: cute.Tensor,
        mDdot: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdQdot: cute.Tensor,
        sdKdot: cute.Tensor,
        sdVdot: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sZ1: cute.Tensor,
        sDdot: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQdot: cute.CopyAtom,
        tma_atom_dKdot: cute.CopyAtom,
        tma_atom_dVdot: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )
                gK, gdKdot = [
                    cute.local_tile(
                        seqlen.offset_batch_K(mX, batch_idx, dim=3)[None, None, head_idx_kv],
                        (self.tile_n, self.tile_hdim),
                        (n_block, 0),
                    )
                    for mX in (mK, mdKdot)
                ]
                gV, gdVdot = [
                    cute.local_tile(
                        seqlen.offset_batch_K(mX, batch_idx, dim=3)[None, None, head_idx_kv],
                        (self.tile_n, self.tile_hdimv),
                        (n_block, 0),
                    )
                    for mX in (mV, mdVdot)
                ]
                gQ, gdQdot = [
                    cute.local_tile(
                        seqlen.offset_batch_Q(mX, batch_idx, dim=3)[None, None, head_idx],
                        (self.tile_m, self.tile_hdim),
                        (None, 0),
                    )
                    for mX in (mQ, mdQdot)
                ]
                gdO = cute.local_tile(
                    seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx],
                    (self.tile_m, self.tile_hdimv),
                    (None, 0),
                )
                gLSE, gdPsum, gZ1, gDdot = [
                    cute.local_tile(
                        seqlen.offset_batch_Q(mX, batch_idx, dim=2, padded=True)[None, head_idx],
                        (self.tile_m,),
                        (None,),
                    )
                    for mX in (mLSE, mdPsum, mZ1, mDdot)
                ]

                load_K, load_dKdot, load_V, load_dVdot = [
                    copy_utils.tma_get_copy_fn(atom, 0, cute.make_layout(1), gX, sX, single_stage=True)[0]
                    for atom, gX, sX in (
                        (tma_atom_K, gK, sK),
                        (tma_atom_dKdot, gdKdot, sdKdot),
                        (tma_atom_V, gV, sV),
                        (tma_atom_dVdot, gdVdot, sdVdot),
                    )
                ]
                load_Q, load_dQdot, load_dO = [
                    copy_utils.tma_producer_copy_fn(
                        copy_utils.tma_get_copy_fn(atom, 0, cute.make_layout(1), gX, sX)[0],
                        pipeline_x,
                    )
                    for atom, gX, sX, pipeline_x in (
                        (tma_atom_Q, gQ, sQ, pipeline_Q),
                        (tma_atom_dQdot, gdQdot, sdQdot, pipeline_Q),
                        (tma_atom_dO, gdO, sdO, pipeline_dO),
                    )
                ]
                load_LSE, load_Z1, load_dPsum, load_Ddot = [
                    copy_utils.tma_producer_copy_fn(
                        copy_utils.cpasync_bulk_get_copy_fn(gX, sX), pipeline_x
                    )
                    for gX, sX, pipeline_x in (
                        (gLSE, sLSE, pipeline_Q),
                        (gZ1, sZ1, pipeline_Q),
                        (gdPsum, sdPsum, pipeline_dO),
                        (gDdot, sDdot, pipeline_dO),
                    )
                ]

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
                process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

                if process_tile:
                    first_m_block = m_block_min
                    pipeline_Q.producer_acquire(
                        producer_state_Q,
                        extra_tx_count=self.tma_copy_bytes["K"] + self.tma_copy_bytes["dKdot"],
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                    load_dKdot(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                    load_Q(first_m_block, producer_state=producer_state_Q)
                    load_dQdot(first_m_block, producer_state=producer_state_Q)
                    # Wait for K1 / bwd preprocess to finish writing LSE, D, z1, Ddot
                    cute.arch.griddepcontrol_wait()
                    load_LSE(first_m_block, producer_state=producer_state_Q)
                    load_Z1(first_m_block, producer_state=producer_state_Q)
                    producer_state_dO_cur = (
                        producer_state_dO
                        if const_expr(self.Q_stage != self.dO_stage)
                        else producer_state_Q
                    )
                    pipeline_dO.producer_acquire(
                        producer_state_dO_cur,
                        extra_tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["dVdot"],
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                    load_dVdot(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                    load_dO(first_m_block, producer_state=producer_state_dO_cur)
                    load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                    load_Ddot(first_m_block, producer_state=producer_state_dO_cur)
                    producer_state_Q.advance()
                    producer_state_dO.advance()

                    for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                        pipeline_Q.producer_acquire(producer_state_Q)
                        load_Q(m_block, producer_state=producer_state_Q)
                        load_dQdot(m_block, producer_state=producer_state_Q)
                        load_LSE(m_block, producer_state=producer_state_Q)
                        load_Z1(m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(producer_state_dO_cur)
                        load_dO(m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(m_block, producer_state=producer_state_dO_cur)
                        load_Ddot(m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_mma_S1: Optional[cute.TiledMma],
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        sdQdot: cute.Tensor,
        sdKdot: cute.Tensor,
        sdVdot: cute.Tensor,
        sdSdot: cute.Tensor,
        sZ1: cute.Tensor,
        sDdot: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        if const_expr(self.mma_dkv_is_rs):
            wg_mma_dQ = tiled_mma_dQ.get_slice(0)
        else:
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx))
        if const_expr(not self.mma_dkv_is_rs):
            tidx_wg = tidx % self.num_threads_per_warp_group
            thr_mma_S1 = tiled_mma_S1.get_slice(tidx_wg)
            wg_mma_S1 = tiled_mma_S1.get_slice(0)

        # Fragments are named by the math operand; with SdP_swapAB the S-like GEMMs run
        # transposed (S^T = K Q^T etc.) and A_idx / B_idx keep referring to the math operands.
        swap = self.SdP_swapAB
        # S = Q K^T ; Sdot' = Q gk^T + gq K^T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        if const_expr(self.mma_dkv_is_rs):
            partition_SdP = partial(sm90_utils.partition_fragment_ABC, wg_mma_SdP, swap_AB=swap)
            _, tSrQ, tSrK = partition_SdP(shape_mnk_S, sQ, sK)
            _, _, tSrdKdot = partition_SdP(shape_mnk_S, sQ, sdKdot)
            _, tSrdQdot, _ = partition_SdP(shape_mnk_S, sdQdot, sK)
            mma_qk_fn = partial(gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=swap)
            mma_qdk_fn = partial(
                gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrdKdot, swap_AB=swap
            )
            mma_dqk_fn = partial(gemm_w_idx, tiled_mma_SdP, tCrA=tSrdQdot, tCrB=tSrK, swap_AB=swap)
            shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
            _, tdPrdO, tdPrV = partition_SdP(shape_mnk_dP, sdO, sV)
            _, _, tErdVdot = partition_SdP(shape_mnk_dP, sdO, sdVdot)
            mma_dov_fn = partial(
                gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=swap
            )
            mma_dodv_fn = partial(
                gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tErdVdot, swap_AB=swap
            )
        else:
            shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        if const_expr(not self.mma_dkv_is_rs):
            _, tSrQ1, tSrK1 = sm90_utils.partition_fragment_ABC(wg_mma_S1, shape_mnk_S, sQ, sK)
            _, _, tSrdKdot1 = sm90_utils.partition_fragment_ABC(wg_mma_S1, shape_mnk_S, sQ, sdKdot)
            _, tSrdQdot1, _ = sm90_utils.partition_fragment_ABC(wg_mma_S1, shape_mnk_S, sdQdot, sK)
            _, tdPrdO1, tdPrV1 = sm90_utils.partition_fragment_ABC(wg_mma_S1, shape_mnk_dP, sdO, sV)
            _, _, tErdVdot1 = sm90_utils.partition_fragment_ABC(wg_mma_S1, shape_mnk_dP, sdO, sdVdot)
            mma_qk1_fn = partial(gemm_zero_init, tiled_mma_S1, shape_mnk_S[:2], tSrQ1, tSrK1)
            mma_qdk1_fn = partial(gemm_zero_init, tiled_mma_S1, shape_mnk_S[:2], tSrQ1, tSrdKdot1)
            mma_dqk1_fn = partial(gemm_w_idx, tiled_mma_S1, tCrA=tSrdQdot1, tCrB=tSrK1)
            mma_dov1_fn = partial(gemm_zero_init, tiled_mma_S1, shape_mnk_dP[:2], tdPrdO1, tdPrV1)
            mma_dodv1_fn = partial(gemm_zero_init, tiled_mma_S1, shape_mnk_dP[:2], tdPrdO1, tErdVdot1)
        # dot_v += Pdot^T dO (shared A for SS, register A for RS)
        sPt = layout_utils.transpose_view(sP) if const_expr(sP is not None) else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt
        )
        # dot_k += dSdot^T Q + dS^T gq
        sQt = layout_utils.transpose_view(sQ)
        sdQdott = layout_utils.transpose_view(sdQdot)
        sdSt = layout_utils.transpose_view(sdS)
        sdSdott = layout_utils.transpose_view(sdSdot)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSdott, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSdott, sQt
        )
        _, tdKrdSt, tdKrdQdott = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sdQdott
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt)
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSdott, tdKrQt)
            mma_dsdq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrdQdott)
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
            mma_dsdq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrdQdott)
        # dot_q = dSdot K + dS gk
        sKt = layout_utils.transpose_view(sK)
        sdKdott = layout_utils.transpose_view(sdKdot)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        _, tdQrdSdot, tdQrKt = sm90_utils.partition_fragment_ABC(
            wg_mma_dQ, shape_mnk_dQ, sdSdot, sKt
        )
        _, tdQrdS, tdQrdKdott = sm90_utils.partition_fragment_ABC(
            wg_mma_dQ, shape_mnk_dQ, sdS, sdKdott
        )
        if const_expr(self.mma_dkv_is_rs):
            # Warpgroup w owns key rows [w, w + 1) * tile_n / 2: its K blocks of the dot_q GEMMs.
            num_k_dQ = cute.size(tdQrdS.shape[2]) // self.num_wg_mma
            k_start_dQ = warp_group_idx * num_k_dQ
            mma_dsk_fn = partial(
                gemm_k_range_zero_init, tiled_mma_dQ, shape_mnk_dQ[:2], tdQrdSdot, tdQrKt,
                k_start_dQ, num_k_dQ,
            )
            mma_dsdk_fn = partial(
                gemm_k_range_w_idx, tiled_mma_dQ, tCrA=tdQrdS, tCrB=tdQrdKdott,
                k_start=k_start_dQ, num_k=num_k_dQ,
            )
        else:
            mma_dsk_fn = partial(gemm_zero_init, tiled_mma_dQ, shape_mnk_dQ[:2], tdQrdSdot, tdQrKt)
            mma_dsdk_fn = partial(gemm_w_idx, tiled_mma_dQ, tCrA=tdQrdS, tCrB=tdQrdKdott)

        # RS R2S dS / dSdot ((n, m) accumulators store transposed).
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        copy_dS_r2s, copy_dSdot_r2s = [
            copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sX if const_expr(not swap) else layout_utils.transpose_view(sX),
                tidx,
                transpose=swap,
                position_independent=True,
                major_mode_size=mms_PdS,
            )[0]
            for sX in (sdS, sdSdot)
        ]
        # Row statistics, read from shared memory in pairs inside the block.
        tLSEsLSE, tLSEsdPsum, tLSEsZ1, tLSEsDdot = [
            layout_utils.mma_partition_C_vec(
                sX, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not swap
            )
            for sX in (sLSE, sdPsum, sZ1, sDdot)
        ]

        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        if const_expr(not self.mma_dkv_is_rs):
            PdS_barrier = cutlass.pipeline.NamedBarrier(
                barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
            )
            copy_P1_r2s, copy_dS1_r2s, copy_dSdot1_r2s = [
                copy_utils.get_smem_store_C(
                    tiled_mma_S1, sX, tidx_wg, transpose=False,
                    position_independent=True, major_mode_size=mms_PdS,
                )[0]
                for sX in (sP, sdS, sdSdot)
            ]
            tiled_copy_ld1 = cute.make_tiled_copy_C(
                copy_utils.get_smem_load_atom(self.dtype, major_mode_size=mms_PdS), tiled_mma_S1
            )
            thr_ld1 = tiled_copy_ld1.get_slice(tidx_wg)
            tSR_sP1 = copy_utils.partition_S_position_independent(thr_ld1, sP)
            tSR_sdS1 = copy_utils.partition_S_position_independent(thr_ld1, sdS)
            def load_P1(src_idx, dst):
                cute.copy(tiled_copy_ld1, tSR_sP1[None, None, None, src_idx], tiled_copy_ld1.retile(dst))
            def load_dS1(src_idx, dst):
                cute.copy(tiled_copy_ld1, tSR_sdS1[None, None, None, src_idx], tiled_copy_ld1.retile(dst))
            tLSEsLSE1, tLSEsZ1_1, tLSEsdPsum1, tLSEsDdot1 = [
                layout_utils.mma_partition_C_vec(
                    sX, thr_mma_S1, expand_shape=self.tile_n, is_colvec=True
                )
                for sX in (sLSE, sZ1, sdPsum, sDdot)
            ]

        if const_expr(not self.mma_dkv_is_rs):
            mma_one_m_block_all = partial(
                self.mma_one_m_block_split,
                warp_group_idx=warp_group_idx,
                mma_qk1_fn=mma_qk1_fn, mma_qdk1_fn=mma_qdk1_fn, mma_dqk1_fn=mma_dqk1_fn,
                mma_dov1_fn=mma_dov1_fn, mma_dodv1_fn=mma_dodv1_fn,
                mma_pdo_fn=mma_pdo_fn, mma_dsq_fn=mma_dsq_fn, mma_dsdq_fn=mma_dsdq_fn,
                mma_dsk_fn=mma_dsk_fn, mma_dsdk_fn=mma_dsdk_fn,
                copy_P1_r2s=copy_P1_r2s, copy_dS1_r2s=copy_dS1_r2s,
                copy_dSdot1_r2s=copy_dSdot1_r2s, load_P1=load_P1, load_dS1=load_dS1,
                pipeline_Q=pipeline_Q, pipeline_dO=pipeline_dO,
                tLSEsLSE1=tLSEsLSE1, tLSEsZ1_1=tLSEsZ1_1,
                tLSEsdPsum1=tLSEsdPsum1, tLSEsDdot1=tLSEsDdot1,
                tdQsdQaccum=tdQsdQaccum, softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale, PdS_barrier=PdS_barrier,
            )
        else:
            mma_one_m_block_all = partial(
                self.mma_one_m_block_wg,
                warp_group_idx=warp_group_idx,
                mma_qk_fn=mma_qk_fn, mma_qdk_fn=mma_qdk_fn, mma_dqk_fn=mma_dqk_fn,
                mma_dov_fn=mma_dov_fn, mma_dodv_fn=mma_dodv_fn,
                mma_pdo_fn=mma_pdo_fn, mma_dsq_fn=mma_dsq_fn, mma_dsdq_fn=mma_dsdq_fn,
                mma_dsk_fn=mma_dsk_fn, mma_dsdk_fn=mma_dsdk_fn,
                copy_dS_r2s=copy_dS_r2s,
                copy_dSdot_r2s=copy_dSdot_r2s, pipeline_Q=pipeline_Q,
                pipeline_dO=pipeline_dO, tLSEsLSE=tLSEsLSE, tLSEsdPsum=tLSEsdPsum,
                tLSEsZ1=tLSEsZ1, tLSEsDdot=tLSEsDdot,
                tdQsdQaccum=tdQsdQaccum, softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
            )
        if const_expr(self.mma_dkv_is_rs):
            # Warpgroup 0 takes the first turn on the tensor cores.
            if warp_group_idx == 0:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

            if process_tile:
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    n_block=n_block,
                    thr_mma=thr_mma_S1 if const_expr(not self.mma_dkv_is_rs) else thr_mma_SdP,
                    mask_seqlen=True,
                    mask_causal=self.is_causal,
                    mask_local=False,
                )
                dKV_accumulate = False
                # Ping-pong turns per tile (RS): {A(first)}, {B(i), A(i+1)}..., {B(last)}.
                self.pingpong_sync(warp_group_idx)
                # Only the diagonal m_blocks (causal) and the last, partial n_block need the mask.
                m_block_mask_end = self.m_block_mask_end(seqlen, n_block, m_block_min, m_block_max)
                if const_expr(self.mask_split == "loops"):
                    for m_block in cutlass.range(m_block_min, m_block_mask_end, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
                    for m_block in cutlass.range(m_block_mask_end, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
                else:
                    mask_fn = partial(
                        self.mask_if, mask_fn=mask_fn, m_block_mask_end=m_block_mask_end
                    )
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
                self.pingpong_arrive(warp_group_idx)

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                )
            else:
                # KV tile with zero Q blocks produces no dot_k / dot_v; write zeros.
                if const_expr(self.is_varlen_q):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def store_dQ_r2s(self, acc_dQ, tdQsdQaccum, warp_group_idx):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
        cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )

    @cute.jit
    def pingpong_sync(self, warp_group_idx: Int32):
        """Wait for this warpgroup's turn to issue a GEMM batch (FA3's warp-scheduler barrier)."""
        if const_expr(self.mma_dkv_is_rs):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1) + warp_group_idx,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    @cute.jit
    def pingpong_arrive(self, warp_group_idx: Int32):
        """Hand the tensor cores to the other warpgroup once this batch is issued."""
        if const_expr(self.mma_dkv_is_rs):
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1) + 1 - warp_group_idx,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    @cute.jit
    def mma_one_m_block_wg(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_qdk_fn: Callable,
        mma_dqk_fn: Callable,
        mma_dov_fn: Callable,
        mma_dodv_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsdq_fn: Callable,
        mma_dsk_fn: Callable,
        mma_dsdk_fn: Callable,
        copy_dS_r2s: Callable,
        copy_dSdot_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tLSEsZ1: cute.Tensor,
        tLSEsDdot: cute.Tensor,
        tdQsdQaccum: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mask_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
    ):
        """RS block with per-warpgroup dot_q partials: the warpgroups share only the Q / dO
        pipeline. Each warpgroup's turn on the tensor cores is this block's B batch (g6-g10)
        plus the next block's A batch (g1-g5), so the other warpgroup's pointwise work runs
        under it (the caller takes the tile's first and releases its last turn)."""
        consumer_state_dO_cur = (
            consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        # (1) S = Q K^T [g1], Sdot' = Q gk^T [g2] + gq K^T [g3], dP = dO V^T [g4], E = dO gv^T [g5]
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        acc_Sdot = mma_qdk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        mma_dqk_fn(acc=acc_Sdot, zero_init=False, A_idx=smem_idx_Q, wg_wait=-1)
        acc_dP = mma_dov_fn(A_idx=smem_idx_dO, wg_wait=-1)
        acc_E = mma_dodv_fn(A_idx=smem_idx_dO, wg_wait=-1)
        self.pingpong_arrive(warp_group_idx)
        warpgroup.wait_group(2)

        # (2) P = exp2(S * scale_log2 - lse_log2), Pdot = P (c Sdot' - z1)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        swap = self.SdP_swapAB
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=swap)
        acc_Sdot_mn = layout_utils.reshape_acc_to_mn(acc_Sdot, transpose=swap)
        lse_pairs = cute.logical_divide(tLSEsLSE[None, smem_idx_Q], cute.make_layout(2))
        z1_pairs = cute.logical_divide(tLSEsZ1[None, smem_idx_Q], cute.make_layout(2))
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse2 = copy_utils.load_s2r(lse_pairs[None, r // 2])
            z12 = copy_utils.load_s2r(z1_pairs[None, r // 2])
            lse_val = lse2[r % 2]
            z1_val = z12[r % 2]
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                p = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
                acc_S_mn[r, c] = p
                acc_Sdot_mn[r, c] = p * (acc_Sdot_mn[r, c] * softmax_scale - z1_val)
        tdVrPdot = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_Sdot), self.dtype)

        # (3) dS = P (dP - D), dSdot = Pdot (dP - D) + P (E - Ddot)
        dpsum_pairs = cute.logical_divide(tLSEsdPsum[None, smem_idx_dO], cute.make_layout(2))
        ddot_pairs = cute.logical_divide(tLSEsDdot[None, smem_idx_dO], cute.make_layout(2))
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=swap)
        acc_E_mn = layout_utils.reshape_acc_to_mn(acc_E, transpose=swap)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum2 = copy_utils.load_s2r(dpsum_pairs[None, r // 2])
            ddot2 = copy_utils.load_s2r(ddot_pairs[None, r // 2])
            dpsum_val = dpsum2[r % 2]
            ddot_val = ddot2[r % 2]
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                dp_minus_d = acc_dP_mn[r, c] - dpsum_val
                acc_E_mn[r, c] = acc_Sdot_mn[r, c] * dp_minus_d + acc_S_mn[r, c] * (
                    acc_E_mn[r, c] - ddot_val
                )
                acc_dP_mn[r, c] = acc_S_mn[r, c] * dp_minus_d

        # (4) R2S this warpgroup's key rows of dS / dSdot: only its own dot_q GEMMs read them,
        # and its previous block's reads finished at that block's wait_group(0).
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        tdKrdSdot = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_E), self.dtype)
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
        copy_dSdot_r2s(tdKrdSdot, dst_idx=smem_idx_PdS)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=PDS_WG0_BARRIER_ID + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group,
        )

        # (5) dot_v += Pdot^T dO [g6], dot_q partial = dSdot K + dS gk over this warpgroup's
        # key rows [g7-g8], dot_k += dSdot^T Q + dS^T gq [g9-g10]
        self.pingpong_sync(warp_group_idx)
        mma_pdo_fn(tCrA=tdVrPdot, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=-1)
        mma_dsdk_fn(acc=acc_dQ, zero_init=False, A_idx=smem_idx_PdS, wg_wait=-1)
        mma_dsq_fn(tCrA=tdKrdSdot, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=-1)
        mma_dsdq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=False, wg_wait=-1)
        warpgroup.wait_group(4)
        pipeline_dO.consumer_release(consumer_state_dO_cur)  # dO only read by g4-g6
        warpgroup.wait_group(2)
        self.store_dQ_r2s(acc_dQ, tdQsdQaccum, warp_group_idx)
        warpgroup.wait_group(0)
        pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def mask_if(self, acc_S: cute.Tensor, m_block: Int32, mask_fn, m_block_mask_end: Int32):
        if m_block < m_block_mask_end:
            mask_fn(acc_S, m_block=m_block)

    @cute.jit
    def m_block_mask_end(
        self, seqlen: SeqlenInfoQK, n_block: Int32, m_block_min: Int32, m_block_max: Int32
    ) -> Int32:
        """End of this n_block's m_blocks that need apply_mask. Causal masking reaches only the
        rows up to the n_block's last key (FA3's SeparateMaskingIterations); keys past seqlen_k
        exist only in the last, partial n_block, which masks all of its m_blocks."""
        mask_end = Int32(m_block_min)
        if const_expr(self.is_causal):
            m_idx_max = (n_block + 1) * self.tile_n - 1 + seqlen.seqlen_q - seqlen.seqlen_k
            mask_end = cutlass.max(
                m_block_min, cutlass.min(m_block_max, m_idx_max // self.tile_m + 1)
            )
        if (n_block + 1) * self.tile_n > seqlen.seqlen_k:
            mask_end = m_block_max
        return mask_end

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
    ):
        """Warp 1: TMA reduce-add each MMA warpgroup's smem dot_q chunk into the fp32 accumulator.
        SS: warpgroup w owns chunk w of the tile. RS: every warpgroup adds a full per-warpgroup
        partial tile to the same gmem slot."""
        num_gmem_chunks = 1 if const_expr(self.mma_dkv_is_rs) else self.num_wg_mma
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx]
                )
            # ((tile_m * hdim / num_gmem_chunks, num_gmem_chunks), num_m_blocks)
            gdQaccum = cute.local_tile(
                mdQaccum_cur,
                (
                    cute.make_layout(
                        (self.tile_m * self.tile_hdim // num_gmem_chunks, num_gmem_chunks)
                    ),
                ),
                (None,),
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                for wg in cutlass.range_constexpr(self.num_wg_mma):
                    cute.arch.cp_async_bulk_wait_group(self.num_wg_mma - 1 - wg, read=True)
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + wg,
                        number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                    )
                for wg in cutlass.range_constexpr(self.num_wg_mma):
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.dQFullWG0) + wg,
                        number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                    )
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdQaccum[None, wg].iterator,
                            gdQaccum[(None, wg % num_gmem_chunks), m_block].iterator,
                            self.tma_copy_bytes["dQ"],
                        )
                    cute.arch.cp_async_bulk_commit_group()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def mma_one_m_block_split(
        self, m_block, consumer_state_Q, consumer_state_dO, warp_group_idx,
        mma_qk1_fn, mma_qdk1_fn, mma_dqk1_fn, mma_dov1_fn, mma_dodv1_fn,
        mma_pdo_fn, mma_dsq_fn, mma_dsdq_fn, mma_dsk_fn, mma_dsdk_fn,
        copy_P1_r2s, copy_dS1_r2s, copy_dSdot1_r2s, load_P1, load_dS1,
        pipeline_Q, pipeline_dO, tLSEsLSE1, tLSEsZ1_1, tLSEsdPsum1, tLSEsDdot1,
        tdQsdQaccum, softmax_scale_log2, softmax_scale, PdS_barrier,
        mask_fn=None, dKV_accumulate=True,
    ):
        consumer_state_dO_cur = (
            consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        if warp_group_idx == 0:
            acc_S = mma_qk1_fn(A_idx=smem_idx_Q, wg_wait=-1)
            acc_Sdot = mma_qdk1_fn(A_idx=smem_idx_Q, wg_wait=-1)
            mma_dqk1_fn(acc=acc_Sdot, zero_init=False, A_idx=smem_idx_Q, wg_wait=-1)
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                number_of_threads=self.num_mma_threads,
            )
            tLSErLSE = copy_utils.load_s2r(tLSEsLSE1[None, smem_idx_Q])
            tLSErZ1 = copy_utils.load_s2r(tLSEsZ1_1[None, smem_idx_Q])
            warpgroup.wait_group(2)
            if const_expr(mask_fn is not None):
                mask_fn(acc_S, m_block=m_block)
            acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
            acc_Sdot_mn = layout_utils.reshape_acc_to_mn(acc_Sdot)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                for c in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
                    acc_S_mn[r, c] = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r], fastmath=True
                    )
            warpgroup.wait_group(0)
            for r in cutlass.range_constexpr(cute.size(acc_Sdot_mn, mode=[0])):
                for c in cutlass.range_constexpr(cute.size(acc_Sdot_mn, mode=[1])):
                    acc_Sdot_mn[r, c] = acc_S_mn[r, c] * (
                        acc_Sdot_mn[r, c] * softmax_scale - tLSErZ1[r]
                    )
            copy_dS1_r2s(utils.cvt_f16(acc_S, self.dtype), dst_idx=smem_idx_PdS)
            copy_P1_r2s(utils.cvt_f16(acc_Sdot, self.dtype), dst_idx=smem_idx_PdS)
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()
            pipeline_dO.consumer_wait(
                consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
            )
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO,
                zero_init=not dKV_accumulate, wg_wait=-1,
            )
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                number_of_threads=self.num_mma_threads,
            )
            pipeline_dO.consumer_wait(
                consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
            )
            acc_dP = mma_dov1_fn(A_idx=smem_idx_dO, wg_wait=-1)
            acc_E = mma_dodv1_fn(A_idx=smem_idx_dO, wg_wait=-1)
            tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum1[None, smem_idx_dO])
            tLSErDdot = copy_utils.load_s2r(tLSEsDdot1[None, smem_idx_dO])
            warpgroup.wait_group(0)
            acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
            acc_E_mn = layout_utils.reshape_acc_to_mn(acc_E)
            for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                for c in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[1])):
                    acc_dP_mn[r, c] -= tLSErdPsum[r]
                    acc_E_mn[r, c] -= tLSErDdot[r]
            PdS_barrier.arrive_and_wait()
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO,
                zero_init=not dKV_accumulate, wg_wait=-1,
            )
            rP = cute.make_rmem_tensor(acc_dP.shape, self.dtype)
            rPdot = cute.make_rmem_tensor(acc_dP.shape, self.dtype)
            load_dS1(smem_idx_PdS, rP)
            load_P1(smem_idx_PdS, rPdot)
            P = rP.load().to(Float32)
            Pdot = rPdot.load().to(Float32)
            dpmd = acc_dP.load()
            acc_E.store(Pdot * dpmd + P * acc_E.load())
            acc_dP.store(P * dpmd)
            copy_dS1_r2s(utils.cvt_f16(acc_dP, self.dtype), dst_idx=smem_idx_PdS)
            copy_dSdot1_r2s(utils.cvt_f16(acc_E, self.dtype), dst_idx=smem_idx_PdS)
            cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=-1)
        mma_dsdk_fn(acc=acc_dQ, zero_init=False, A_idx=smem_idx_PdS, wg_wait=2)
        pipeline_dO.consumer_release(consumer_state_dO_cur)
        mma_dsq_fn(
            A_idx=smem_idx_PdS, B_idx=smem_idx_Q,
            zero_init=not dKV_accumulate, wg_wait=-1,
        )
        mma_dsdq_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=False, wg_wait=2)
        self.store_dQ_r2s(acc_dQ, tdQsdQaccum, warp_group_idx)
        warpgroup.wait_group(0)
        pipeline_Q.consumer_release(consumer_state_Q)
        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO
