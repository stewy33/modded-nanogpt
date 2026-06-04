# AOT ID: ['29_backward']
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


# kernel path: /workspace/modded-nanogpt/.inductor_cache/ru/cruaouveecwocunuguk3l2vlevazef2wx2i42ynyylp76dlpx5xp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.div, aten.pow, aten.add, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %add_8), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %rsqrt_3), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [2], True), kwargs = {dtype: torch.float32})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 768), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 1.0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_7, 2.0), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_16), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_17), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_0 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_pow_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1572864, 'r0_': 1207959552}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_pow_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp0 * tmp7
    tmp9 = -0.5
    tmp10 = tmp6 * tmp9
    tmp11 = tmp7 * tmp7
    tmp12 = tmp11 * tmp7
    tmp13 = tmp10 * tmp12
    tmp14 = 0.0013020833333333333
    tmp15 = tmp13 * tmp14
    tmp16 = 2.0
    tmp17 = tmp1 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp8 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp20, r0_mask)
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/iw/ciwtznrmuwfsa42hryp36wgqmcviqmncovx6jubbhcdtghxwempl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/h7/ch7hm55q2yfkel3lwaxajxyl7fior7fgb6mrsmdd5fblavpxqbvq.py
# Topologically Sorted Source Nodes: [relu, square], Original ATen: [aten._to_copy, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
#   square => convert_element_type_23
# Graph fragment:
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_17, torch.float32), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_13,), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%relu, torch.float32), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_23, 1.0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_8, 2.0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_33, %mul_18), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %convert_element_type_35), kwargs = {})
triton_poi_fused__to_copy_mul_pow_relu_threshold_backward_2 = async_compile.triton('triton_poi_fused__to_copy_mul_pow_relu_threshold_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_pow_relu_threshold_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3221225472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_pow_relu_threshold_backward_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 402653184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp2.to(tl.float32)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.where(tmp4, tmp3, tmp11)
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/kg/ckgim73uc23n6rwidi5u2mmw5oaru3rtzmhc24jet73bevq4xsrr.py
# Topologically Sorted Source Nodes: [add_4], Original ATen: [aten.mul, aten.div, aten.pow, aten.add, aten._to_copy, aten.sum, aten.view]
# Source node to ATen node mapping:
#   add_4 => add_6
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %rsqrt_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 768), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 1.0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_7, 2.0), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_16), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_17), kwargs = {})
#   %convert_element_type_40 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_8, %view_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_40, %add_6), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_40, %rsqrt_2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [2], True), kwargs = {dtype: torch.float32})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_21), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 768), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 1.0), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_10, 2.0), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_24), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_25), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   %view_20 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_42, [131072, 768]), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_view_3 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_pow_sum_view_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1572864, 'r0_': 2818572288}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_pow_sum_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 * tmp12
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 * tmp12
    tmp18 = tmp17 * tmp12
    tmp19 = tmp16 * tmp18
    tmp20 = 0.0013020833333333333
    tmp21 = tmp19 * tmp20
    tmp23 = 2.0
    tmp24 = tmp22 * tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = tmp13 + tmp25
    tmp28 = tmp1 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp10 * tmp15
    tmp31 = tmp27 * tmp27
    tmp32 = tmp31 * tmp27
    tmp33 = tmp30 * tmp32
    tmp34 = tmp33 * tmp20
    tmp35 = tmp5 * tmp23
    tmp36 = tmp34 * tmp35
    tmp37 = tmp29 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp37, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp38, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/73/c73prbvgvjdrrful73mmiq22ru7mtgbziuhbd75bvcf5a3kls5rf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_10, torch.float32), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/uj/cuj2bzb6ntmh3trl2ngjmtcnmrl2cu2ylcawxemq2jlzpnhgohuw.py
# Topologically Sorted Source Nodes: [neg, rms_norm_45], Original ATen: [aten.mul, aten.neg, aten.add, aten.slice_backward, aten._to_copy, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   neg => neg
#   rms_norm_45 => convert_element_type_14
# Graph fragment:
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %unsqueeze_1), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%unsqueeze_3,), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_10, %neg), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_9, %unsqueeze_3), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_9, %unsqueeze_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %mul_29), kwargs = {})
#   %full_default_1 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 1024, 6, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %add_13, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %add_14, 3, 0, 64), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %convert_element_type_48 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.float32), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float32), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %convert_element_type_14), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %rsqrt_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_34, [3], True), kwargs = {dtype: torch.float32})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_2, 128), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 1.0), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_12, 2.0), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_38), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %mul_39), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5', '''
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
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/xp/cxpcqcbk4yqtgzrbdefuj3hzf5xbi3wvtfea6vsnzm3xvr47wx5u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_25, torch.float32), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_56, %convert_element_type_62), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_31, torch.float32), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %convert_element_type_68), kwargs = {})
triton_poi_fused__to_copy_add_6 = async_compile.triton('triton_poi_fused__to_copy_add_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1409286144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tl.store(out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_5, primals_6, primals_8, view, mm, mm_1, rsqrt, rsqrt_1, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, mm_3, rsqrt_2, view_12, mm_4, view_14, add_8, rsqrt_3, permute_12, permute_16, permute_20, permute_28, permute_32, permute_36, tangents_1 = args
    args.clear()
    assert_size_stride(primals_5, (1024, 64), (64, 1))
    assert_size_stride(primals_6, (1024, 64), (64, 1))
    assert_size_stride(primals_8, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(view, (131072, 768), (768, 1))
    assert_size_stride(mm, (131072, 768), (768, 1))
    assert_size_stride(mm_1, (131072, 768), (768, 1))
    assert_size_stride(rsqrt, (128, 1024, 6, 1), (6144, 6, 1, 1))
    assert_size_stride(rsqrt_1, (128, 1024, 6, 1), (6144, 6, 1, 1))
    assert_size_stride(permute_3, (128, 6, 1024, 128), (786432, 128, 768, 1))
    assert_size_stride(permute_4, (128, 6, 1024, 128), (786432, 128, 768, 1))
    assert_size_stride(permute_5, (128, 6, 1024, 128), (786432, 128, 768, 1))
    assert_size_stride(getitem, (128, 6, 1024, 128), (786432, 128, 768, 1))
    assert_size_stride(getitem_1, (128, 6, 1024), (6144, 1024, 1))
    assert_size_stride(getitem_6, (2, ), (1, ))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(mm_3, (131072, 768), (768, 1))
    assert_size_stride(rsqrt_2, (128, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_12, (131072, 768), (768, 1))
    assert_size_stride(mm_4, (131072, 3072), (3072, 1))
    assert_size_stride(view_14, (131072, 3072), (3072, 1))
    assert_size_stride(add_8, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(rsqrt_3, (128, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_12, (768, 3072), (3072, 1))
    assert_size_stride(permute_16, (3072, 768), (768, 1))
    assert_size_stride(permute_20, (768, 768), (768, 1))
    assert_size_stride(permute_28, (768, 768), (768, 1))
    assert_size_stride(permute_32, (768, 768), (768, 1))
    assert_size_stride(permute_36, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (128, 1024, 768), (786432, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 1024, 1), (1024, 1, 131072), torch.float32)
        buf1 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.div, aten.pow, aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mul_pow_sum_0.run(tangents_1, add_8, rsqrt_3, buf0, buf1, 131072, 768, stream=stream0)
        buf2 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (768, 131072), (1, 768), 0), view_14, out=buf2)
        del view_14
        buf3 = empty_strided_cuda((131072, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (131072, 768), (768, 1), 0), permute_12, out=buf3)
        del permute_12
        buf4 = empty_strided_cuda((768, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf2, buf4, 2359296, stream=stream0)
        buf5 = reinterpret_tensor(buf3, (128, 1024, 3072), (3145728, 3072, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [relu, square], Original ATen: [aten._to_copy, aten.relu, aten.pow, aten.mul, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_pow_relu_threshold_backward_2.run(buf5, mm_4, 402653184, stream=stream0)
        del mm_4
        buf6 = reinterpret_tensor(buf2, (3072, 768), (768, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (3072, 131072), (1, 3072), 0), view_12, out=buf6)
        del view_12
        buf7 = reinterpret_tensor(buf1, (131072, 768), (768, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (131072, 3072), (3072, 1), 0), permute_16, out=buf7)
        del buf5
        del permute_16
        buf8 = empty_strided_cuda((3072, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf6, buf8, 2359296, stream=stream0)
        del buf6
        buf10 = primals_8; del primals_8  # reuse
        buf11 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [add_4], Original ATen: [aten.mul, aten.div, aten.pow, aten.add, aten._to_copy, aten.sum, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_3.run(buf10, buf7, mm_3, tangents_1, rsqrt_3, buf0, add_8, rsqrt_2, buf11, 131072, 768, stream=stream0)
        del add_8
        del buf0
        del mm_3
        del rsqrt_2
        del rsqrt_3
        del tangents_1
        buf12 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (768, 131072), (1, 768), 0), reinterpret_tensor(getitem, (131072, 768), (768, 1), 0), out=buf12)
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, permute_20, out=buf13)
        del buf11
        del permute_20
        buf14 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf12, buf14, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf15 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf13, (128, 6, 1024, 128), (786432, 128, 768, 1), 0), permute_3, permute_4, permute_5, getitem, getitem_1, None, None, 1024, 1024, 0.0, True, getitem_6, getitem_7, scale=0.08838834764831843)
        del buf13
        del getitem
        del getitem_1
        del getitem_6
        del getitem_7
        del permute_3
        del permute_4
        del permute_5
        buf16 = buf15[0]
        assert_size_stride(buf16, (128, 6, 1024, 128), (786432, 128, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf17 = buf15[1]
        assert_size_stride(buf17, (128, 6, 1024, 128), (786432, 128, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf17, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf18 = buf15[2]
        assert_size_stride(buf18, (128, 6, 1024, 128), (786432, 128, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf15
        buf26 = reinterpret_tensor(mm_1, (128, 1024, 6, 128), (786432, 768, 128, 1), 0); del mm_1  # reuse
        # Topologically Sorted Source Nodes: [neg, rms_norm_45], Original ATen: [aten.mul, aten.neg, aten.add, aten.slice_backward, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5.run(buf26, buf17, primals_5, primals_6, rsqrt_1, 786432, 128, stream=stream0)
        del buf17
        del rsqrt_1
        buf30 = reinterpret_tensor(mm, (128, 1024, 6, 128), (786432, 768, 128, 1), 0); del mm  # reuse
        # Topologically Sorted Source Nodes: [neg, rms_norm], Original ATen: [aten.neg, aten.slice_backward, aten.mul, aten.add, aten._to_copy, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mul_neg_pow_slice_backward_sum_5.run(buf30, buf16, primals_5, primals_6, rsqrt, 786432, 128, stream=stream0)
        del primals_5
        del primals_6
        del rsqrt
        buf23 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (768, 131072), (1, 768), 0), view, out=buf23)
        buf24 = reinterpret_tensor(buf16, (131072, 768), (768, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (131072, 768), (768, 1), 0), permute_28, out=buf24)
        del permute_28
        buf25 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf23, buf25, 589824, stream=stream0)
        buf27 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (768, 131072), (1, 768), 0), view, out=buf27)
        buf28 = reinterpret_tensor(buf18, (131072, 768), (768, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (131072, 768), (768, 1), 0), permute_32, out=buf28)
        del permute_32
        buf29 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf27, buf29, 589824, stream=stream0)
        buf31 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (768, 131072), (1, 768), 0), view, out=buf31)
        del view
        buf32 = reinterpret_tensor(buf26, (131072, 768), (768, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (131072, 768), (768, 1), 0), permute_36, out=buf32)
        del buf30
        del permute_36
        buf33 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_6.run(buf24, buf28, buf32, buf33, 100663296, stream=stream0)
        del buf24
        del buf28
        del buf32
        buf34 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf31, buf34, 589824, stream=stream0)
        del buf31
    return (buf33, buf34, buf29, buf25, None, None, buf14, buf10, buf8, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_5 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((1024, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_1 = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((128, 1024, 6, 1), (6144, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((128, 1024, 6, 1), (6144, 6, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((128, 6, 1024, 128), (786432, 128, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_4 = rand_strided((128, 6, 1024, 128), (786432, 128, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_5 = rand_strided((128, 6, 1024, 128), (786432, 128, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((128, 6, 1024, 128), (786432, 128, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((128, 6, 1024), (6144, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.uint64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.uint64)
    mm_3 = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((131072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    view_14 = rand_strided((131072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_8 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_12 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_16 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_20 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_32 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_5, primals_6, primals_8, view, mm, mm_1, rsqrt, rsqrt_1, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, mm_3, rsqrt_2, view_12, mm_4, view_14, add_8, rsqrt_3, permute_12, permute_16, permute_20, permute_28, permute_32, permute_36, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
