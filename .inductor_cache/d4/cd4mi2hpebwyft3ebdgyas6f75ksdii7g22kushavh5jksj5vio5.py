# AOT ID: ['0_backward']
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


# kernel path: /workspace/modded-nanogpt/.inductor_cache/n7/cn76ckcizjkv3v6lyjm7u3bprxy3ej3spzpnbtvpdwgsq7jkbhae.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([50304, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_0 = async_compile.triton('triton_poi_fused_embedding_dense_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 309067776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38633472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/xg/cxgfw2oxdbnkbfteypfybt53tqfanlosd6fgneaxuxvgrsop6jjb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %embedding), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %rsqrt), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [2], True), kwargs = {dtype: torch.float32})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %mul_2), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 768), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 1.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_3, 2.0), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_6), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze, %full_default, %add_2), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([50304, 768], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_1, [%primals_1], %where, True), kwargs = {})
triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_1 = async_compile.triton('triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_1', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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
    tmp15 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([R0_BLOCK], 50304, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 50304), "index out of bounds: 0 <= tmp11 < 50304")
    tmp13 = tl.full([1], -1, tl.int64)
    tmp14 = tmp7 == tmp13
    tmp17 = tmp0 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp6 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 0.0013020833333333333
    tmp25 = tmp23 * tmp24
    tmp26 = 2.0
    tmp27 = tmp1 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp14, tmp30, tmp29)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(r0_1 + 768*tmp11, [R0_BLOCK])), tmp31, r0_mask, sem='relaxed')
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, embedding, rsqrt, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (128, 1024), (1024, 1))
    assert_size_stride(embedding, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(rsqrt, (128, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (128, 1024, 768), (786432, 768, 1))
    assert_size_stride(tangents_2, (128, 1024, 768), (786432, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((50304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_0.run(buf1, 38633472, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_embedding_dense_backward_mul_pow_sum_1.run(tangents_1, embedding, primals_1, tangents_2, rsqrt, buf1, 131072, 768, stream=stream0)
        del embedding
        del primals_1
        del rsqrt
        del tangents_1
        del tangents_2
    return (None, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((128, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((128, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, embedding, rsqrt, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
