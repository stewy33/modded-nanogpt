
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 1342177280}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 786432
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 6) % 1024)
    tmp28 = tl.load(in_out_ptr0 + (r0_3 + 128*x4), None).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp0 = r0_3
    tmp1 = tl.full([1, 1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r0_3 + 128*x4), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + ((-64) + r0_3 + 64*x1), tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.load(in_ptr0 + ((-64) + r0_3 + 128*x4), tmp2, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + ((-64) + r0_3 + 64*x1), tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (64 + r0_3 + 128*x4), tmp14, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_3 + 64*x1), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = -tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tl.load(in_ptr0 + (r0_3 + 128*x4), tmp14, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (r0_3 + 64*x1), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp14, tmp22, tmp23)
    tmp25 = tl.where(tmp14, tmp24, tmp12)
    tmp26 = tmp13 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp33 = tl.sum(tmp31, 1)[:, None]
    tmp35 = tmp27 * tmp34
    tmp36 = -0.5
    tmp37 = tmp33 * tmp36
    tmp38 = tmp34 * tmp34
    tmp39 = tmp38 * tmp34
    tmp40 = tmp37 * tmp39
    tmp41 = 0.0078125
    tmp42 = tmp40 * tmp41
    tmp43 = 2.0
    tmp44 = tmp29 * tmp43
    tmp45 = tmp42 * tmp44
    tmp46 = tmp35 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_3 + 128*x4), tmp47, None)
