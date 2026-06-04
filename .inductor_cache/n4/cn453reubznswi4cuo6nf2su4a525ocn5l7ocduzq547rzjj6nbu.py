# AOT ID: ['14_inference']
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


# kernel path: /workspace/modded-nanogpt/.inductor_cache/kc/ckco6ozectdiwe5ysn23yyixi6bzf7ia2wnhssdnrbuer6mkaurd.py
# Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type
#   norm => convert_element_type_1, pow_1, sum_1
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, None), kwargs = {})
triton_red_fused__to_copy_linalg_vector_norm_0 = async_compile.triton('triton_red_fused__to_copy_linalg_vector_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 576, 'r0_': 2359296}}
)
@triton.jit
def triton_red_fused__to_copy_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 72
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/by/cby5x72zvebe2ih3wejyjdtgs2icl3ihtfr3fzwayadthnzrcnof.py
# Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   X => convert_element_type
#   norm => convert_element_type_1, pow_1, sum_1
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, None), kwargs = {})
triton_per_fused__to_copy_linalg_vector_norm_1 = async_compile.triton('triton_per_fused__to_copy_linalg_vector_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 288}}
)
@triton.jit
def triton_per_fused__to_copy_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 72
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/ew/cewl3tcvzjjoie4txxe24rya4t47tuakicylaghcw3y3ueuvudbv.py
# Topologically Sorted Source Nodes: [X, norm, add, X_1], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div]
# Source node to ATen node mapping:
#   X => convert_element_type
#   X_1 => div
#   add => add
#   norm => convert_element_type_2, pow_2
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_2, torch.bfloat16), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1e-07), kwargs = {})
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add), kwargs = {})
triton_poi_fused__to_copy_add_div_linalg_vector_norm_2 = async_compile.triton('triton_poi_fused__to_copy_add_div_linalg_vector_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_linalg_vector_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_linalg_vector_norm_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp4 = libdevice.sqrt(tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1e-07
    tmp7 = tmp5 + tmp6
    tmp8 = (tmp1 / tmp7)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/au/cauejhguwirjkrrhijgd3suybsgekax4enq2236wjhvpryb7hqum.py
# Topologically Sorted Source Nodes: [mul_2], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_2 => mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 2.0315), kwargs = {})
triton_poi_fused_mul_3 = async_compile.triton('triton_poi_fused_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3538944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_3(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 2.0315
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/si/csibi5pylq2ibsl2nqz77p4d4x2tx2ca4nhugoxhtvat3q2dcfuz.py
# Topologically Sorted Source Nodes: [mul, mul_1, add_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
#   mul => mul
#   mul_1 => mul_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 3.4445), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_1, -4.775), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 3.4445
    tmp2 = tmp0 * tmp1
    tmp4 = -4.775
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (768, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((72, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_linalg_vector_norm_0.run(arg0_1, buf1, 72, 8192, stream=stream0)
        buf2 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [X, norm], Original ATen: [aten._to_copy, aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_linalg_vector_norm_1.run(buf1, buf2, 1, 72, stream=stream0)
        del buf1
        buf3 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [X, norm, add, X_1], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_linalg_vector_norm_2.run(arg0_1, buf2, buf3, 589824, stream=stream0)
        del arg0_1
        del buf2
        buf4 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [A], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(buf3, (768, 768), (1, 768), 0), out=buf4)
        buf5 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [B], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf3, out=buf5)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [mul_2], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf6, 589824, stream=stream0)
        buf7 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [mul, mul_1, add_1], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf7, buf5, 589824, stream=stream0)
        buf8 = empty_strided_cuda((768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, mul_1, add_1, mul_2], Original ATen: [aten.mul, aten.add]
        extern_kernels.addmm(buf7, buf6, buf5, alpha=1, beta=1, out=buf8)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [A_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(buf8, (768, 768), (1, 768), 0), out=buf9)
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [B_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, buf8, out=buf10)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [mul_5], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf11, 589824, stream=stream0)
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [mul_3, mul_4, add_3], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf12, buf10, 589824, stream=stream0)
        buf13 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [mul_3, mul_4, add_3, mul_5], Original ATen: [aten.mul, aten.add]
        extern_kernels.addmm(buf12, buf11, buf10, alpha=1, beta=1, out=buf13)
        buf14 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [A_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf13, reinterpret_tensor(buf13, (768, 768), (1, 768), 0), out=buf14)
        buf15 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [B_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, buf13, out=buf15)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [mul_8], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf16, 589824, stream=stream0)
        buf17 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [mul_6, mul_7, add_5], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf17, buf15, 589824, stream=stream0)
        buf18 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [mul_6, mul_7, add_5, mul_8], Original ATen: [aten.mul, aten.add]
        extern_kernels.addmm(buf17, buf16, buf15, alpha=1, beta=1, out=buf18)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [A_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(buf18, (768, 768), (1, 768), 0), out=buf19)
        buf20 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [B_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, buf18, out=buf20)
        buf21 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [mul_11], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf21, 589824, stream=stream0)
        buf22 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [mul_9, mul_10, add_7], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf22, buf20, 589824, stream=stream0)
        buf23 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [mul_9, mul_10, add_7, mul_11], Original ATen: [aten.mul, aten.add]
        extern_kernels.addmm(buf22, buf21, buf20, alpha=1, beta=1, out=buf23)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [A_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, reinterpret_tensor(buf23, (768, 768), (1, 768), 0), out=buf24)
        buf25 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [B_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, buf23, out=buf25)
        buf26 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [mul_14], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf26, 589824, stream=stream0)
        buf27 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [mul_12, mul_13, add_9], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(buf27, buf25, 589824, stream=stream0)
        buf28 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [mul_12, mul_13, add_9, mul_14], Original ATen: [aten.mul, aten.add]
        extern_kernels.addmm(buf27, buf26, buf25, alpha=1, beta=1, out=buf28)
        del buf25
        del buf26
        del buf27
    return (buf28, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
