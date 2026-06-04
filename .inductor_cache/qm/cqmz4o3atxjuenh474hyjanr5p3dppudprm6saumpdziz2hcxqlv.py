
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_cat_mean_pow_rsqrt_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 2, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12582912, 'r0_': 2550136832}}
)
@triton.jit
def triton_per_fused__to_copy_add_cat_mean_pow_rsqrt_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_1 = r0_index
    x0 = xindex
    x3 = ((xindex // 6) % 1024)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 128.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1.1920928955078125e-07
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = (tmp16 / tmp6)
    tmp18 = tmp17 + tmp8
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = r0_1
    tmp21 = tl.full([1, 1], 0, tl.int64)
    tmp22 = tmp20 >= tmp21
    tmp23 = tl.full([1, 1], 64, tl.int64)
    tmp24 = tmp20 < tmp23
    tmp25 = tl.load(in_ptr0 + (128*x0 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp10
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (64*x3 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp10
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.load(in_ptr3 + (64*x3 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp30 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp24, tmp37, tmp38)
    tmp40 = tmp20 >= tmp23
    tmp41 = tl.full([1, 1], 128, tl.int64)
    tmp42 = tmp20 < tmp41
    tmp43 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp10
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.load(in_ptr3 + (64*x3 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = -tmp47
    tmp49 = tmp46 * tmp48
    tmp50 = tl.load(in_ptr0 + (64 + 128*x0 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp51 * tmp10
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tl.load(in_ptr2 + (64*x3 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp55 = tmp53 * tmp54
    tmp56 = tmp49 + tmp55
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp40, tmp56, tmp57)
    tmp59 = tl.where(tmp24, tmp39, tmp58)
    tmp60 = tl.load(in_ptr1 + (128*x0 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61 * tmp19
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp63 * tmp29
    tmp65 = tl.load(in_ptr1 + (64 + 128*x0 + (r0_1)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66 * tmp19
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp68 * tmp35
    tmp70 = tmp64 + tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp24, tmp70, tmp71)
    tmp73 = tl.load(in_ptr1 + (128*x0 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74 * tmp19
    tmp76 = tmp75.to(tl.float32)
    tmp77 = tmp76 * tmp48
    tmp78 = tl.load(in_ptr1 + (64 + 128*x0 + ((-64) + r0_1)), tmp40, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp79 * tmp19
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp81 * tmp54
    tmp83 = tmp77 + tmp82
    tmp84 = tl.full(tmp83.shape, 0.0, tmp83.dtype)
    tmp85 = tl.where(tmp40, tmp83, tmp84)
    tmp86 = tl.where(tmp24, tmp72, tmp85)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp19, None)
    tl.store(out_ptr0 + (r0_1 + 128*x0), tmp59, None)
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp86, None)
