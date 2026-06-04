# AOT ID: ['1_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/modded-nanogpt/.inductor_cache/46/c46e4pqardxyl654i4dsraotdl3nvvbpaxoc4duz4zxzlxro2t6a.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type, permute
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_0 = async_compile.triton('triton_poi_fused__to_copy_t_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/6a/c6aw4guqnysvgjunfzkzuu5ok2lbzhkvgjvlav7n2c5ozqzrzuhn.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


cpp_fused_mul_2 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(1024L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                        auto tmp1 = x0;
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp3 * tmp0;
                        tmp4.store(out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                    }
                }
            }
        }
    }
}
''')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/og/coglniryhf5xqf7fmz2rbvz3iqbidba767edpnkq7ccg53nosaxr.py
# Topologically Sorted Source Nodes: [cos, bfloat16, sin, bfloat16_1], Original ATen: [aten.cos, aten._to_copy, aten.sin]
# Source node to ATen node mapping:
#   bfloat16 => convert_element_type_14
#   bfloat16_1 => convert_element_type_15
#   cos => cos
#   sin => sin
# Graph fragment:
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%device_put_1,), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cos, torch.bfloat16), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%device_put_1,), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sin, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cos_sin_3 = async_compile.triton('triton_poi_fused__to_copy_cos_sin_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cos_sin_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cos_sin_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl_math.cos(tmp0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl_math.sin(tmp0)
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/qm/cqmz4o3atxjuenh474hyjanr5p3dppudprm6saumpdziz2hcxqlv.py
# Topologically Sorted Source Nodes: [rms_norm_1, rms_norm_2, cat, cat_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.cat]
# Source node to ATen node mapping:
#   cat => cat
#   cat_1 => cat_1
#   rms_norm_1 => add, convert_element_type_16, mean, pow_1, rsqrt
#   rms_norm_2 => add_1, convert_element_type_18, mean_1, pow_2, rsqrt_1
# Graph fragment:
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_16, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [3], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_18, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [3], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3], 3), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_4, %add_5], 3), kwargs = {})
triton_per_fused__to_copy_add_cat_mean_pow_rsqrt_4 = async_compile.triton('triton_per_fused__to_copy_add_cat_mean_pow_rsqrt_4', '''
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
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/sb/csbtyrsuxkswxotcl7bm224hcocdvkdyncqcacmltzgwgba5ywjk.py
# Topologically Sorted Source Nodes: [add_4, rms_norm_3, linear_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   add_4 => add_6
#   linear_4 => convert_element_type_24
#   rms_norm_3 => add_7, mean_2, mul_11, pow_3, rsqrt_2
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_7, %view_12), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [2], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_2, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt_2), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1048576, 'r0_': 1006632960}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 768.0
    tmp10 = (tmp8 / tmp9)
    tmp11 = 1.1920928955078125e-07
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp15, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/si/csinw6rzo7icv2o3bi3wjmtrxyd3zk4zszd7omftqqh2fdpf5rqb.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_4 => convert_element_type_23, permute_8
# Graph fragment:
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
#   %permute_8 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_23, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_6 = async_compile.triton('triton_poi_fused__to_copy_t_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/i5/ci52jc2bxjuffvrdzbqjgiusdkg3sb7gzl5rfoaxywjs6sxlk43i.py
# Topologically Sorted Source Nodes: [relu, square, linear_5], Original ATen: [aten.relu, aten._to_copy, aten.pow]
# Source node to ATen node mapping:
#   linear_5 => convert_element_type_29
#   relu => relu
#   square => convert_element_type_27, pow_4
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_14,), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%relu, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_27, 2), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_4, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_pow_relu_7 = async_compile.triton('triton_poi_fused__to_copy_pow_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_pow_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2415919104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_pow_relu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 402653184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/26/c26tpkxbfs5yuyuqzqiw2kiqpbsxpuzfwpn4ktxb5dzelzvkuzjz.py
# Topologically Sorted Source Nodes: [add_4, add_5, rms_norm_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_5 => add_8
#   rms_norm_4 => add_9, mean_3, mul_12, pow_5, rsqrt_3
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_7, %view_12), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_16), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_3, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_3), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1048576, 'r0_': 2415919104}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 131072
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = 768.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1.1920928955078125e-07
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp6 * tmp16
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp6, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp17, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(primals_2, (768, 768), (768, 1))
    assert_size_stride(primals_3, (768, 768), (768, 1))
    assert_size_stride(primals_4, (768, 768), (768, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(primals_8, (3072, 768), (768, 1))
    assert_size_stride(primals_9, (768, 3072), (3072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_0.run(primals_2, buf0, 589824, stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_1, buf1, 100663296, stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (131072, 768), (768, 1), 0), buf0, out=buf2)
        buf3 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_0.run(primals_3, buf3, 589824, stream=stream0)
        del primals_3
        buf4 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (131072, 768), (768, 1), 0), buf3, out=buf4)
        buf5 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_0.run(primals_4, buf5, 589824, stream=stream0)
        del primals_4
        buf6 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (131072, 768), (768, 1), 0), buf5, out=buf6)
    buf7 = empty_strided_cpu((1024, 64), (64, 1), torch.float32)
    cpp_fused_mul_2(primals_5, buf7)
    del primals_5
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1024, 64), (64, 1), torch.float32)
        buf8.copy_(buf7, False)
        del buf7
        buf9 = empty_strided_cuda((1024, 64), (64, 1), torch.bfloat16)
        buf10 = empty_strided_cuda((1024, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cos, bfloat16, sin, bfloat16_1], Original ATen: [aten.cos, aten._to_copy, aten.sin]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cos_sin_3.run(buf8, buf9, buf10, 65536, stream=stream0)
        del buf8
        buf11 = empty_strided_cuda((128, 1024, 6, 1), (6144, 6, 1, 786432), torch.float32)
        buf12 = reinterpret_tensor(buf11, (128, 1024, 6, 1), (6144, 6, 1, 1), 0); del buf11  # reuse
        buf13 = empty_strided_cuda((128, 1024, 6, 1), (6144, 6, 1, 786432), torch.float32)
        buf14 = reinterpret_tensor(buf13, (128, 1024, 6, 1), (6144, 6, 1, 1), 0); del buf13  # reuse
        buf15 = empty_strided_cuda((128, 1024, 6, 128), (786432, 768, 128, 1), torch.bfloat16)
        buf16 = empty_strided_cuda((128, 1024, 6, 128), (786432, 768, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rms_norm_1, rms_norm_2, cat, cat_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.cat]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_cat_mean_pow_rsqrt_4.run(buf12, buf14, buf2, buf4, buf9, buf10, buf15, buf16, 786432, 128, stream=stream0)
        # Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf17 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf15, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), reinterpret_tensor(buf16, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), reinterpret_tensor(buf6, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), 0.0, True, scale=0.08838834764831843)
        buf18 = buf17[0]
        assert_size_stride(buf18, (128, 6, 1024, 128), (786432, 128, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        buf19 = buf17[1]
        assert_size_stride(buf19, (128, 6, 1024), (6144, 1024, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf19, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        buf20 = buf17[6]
        assert_size_stride(buf20, (2, ), (1, ), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf20, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        buf21 = buf17[7]
        assert_size_stride(buf21, (), (), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf21, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf17
        buf23 = empty_strided_cuda((768, 768), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_0.run(primals_6, buf23, 589824, stream=stream0)
        del primals_6
        buf24 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (131072, 768), (768, 1), 0), buf23, out=buf24)
        buf25 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
        buf26 = reinterpret_tensor(buf25, (128, 1024, 1), (1024, 1, 1), 0); del buf25  # reuse
        buf28 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [add_4, rms_norm_3, linear_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf26, primals_7, buf24, buf28, 131072, 768, stream=stream0)
        buf27 = empty_strided_cuda((768, 3072), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_8, buf27, 2359296, stream=stream0)
        del primals_8
        buf29 = empty_strided_cuda((131072, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (131072, 768), (768, 1), 0), buf27, out=buf29)
        buf30 = empty_strided_cuda((3072, 768), (1, 3072), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_6.run(primals_9, buf30, 2359296, stream=stream0)
        del primals_9
        buf31 = empty_strided_cuda((128, 1024, 3072), (3145728, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [relu, square, linear_5], Original ATen: [aten.relu, aten._to_copy, aten.pow]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_pow_relu_7.run(buf29, buf31, 402653184, stream=stream0)
        buf32 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (131072, 3072), (3072, 1), 0), buf30, out=buf32)
        buf33 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.float32)
        buf34 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
        buf35 = reinterpret_tensor(buf34, (128, 1024, 1), (1024, 1, 1), 0); del buf34  # reuse
        buf36 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, add_5, rms_norm_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf35, primals_7, buf24, buf32, buf33, buf36, 131072, 768, stream=stream0)
        del buf32
    return (buf36, buf33, buf10, buf9, primals_7, reinterpret_tensor(buf1, (131072, 768), (768, 1), 0), buf2, buf4, buf9, buf10, buf12, buf14, reinterpret_tensor(buf15, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), reinterpret_tensor(buf16, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), reinterpret_tensor(buf6, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), buf18, buf19, buf20, buf21, buf24, buf26, reinterpret_tensor(buf28, (131072, 768), (768, 1), 0), buf29, reinterpret_tensor(buf31, (131072, 3072), (3072, 1), 0), buf33, buf35, reinterpret_tensor(buf30, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf27, (3072, 768), (768, 1), 0), reinterpret_tensor(buf23, (768, 768), (768, 1), 0), reinterpret_tensor(buf5, (768, 768), (768, 1), 0), reinterpret_tensor(buf3, (768, 768), (768, 1), 0), reinterpret_tensor(buf0, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
