# AOT ID: ['13_backward']
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


# kernel path: /workspace/modded-nanogpt/.inductor_cache/m4/cm45f6fzar2skwpgxkas72tbztxydmboefeaiwmfr5ius3ack6aj.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data, aten._to_copy]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_5), kwargs = {})
#   %ne_3 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %unsqueeze_1, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [131072, 50304], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_3, %div_1, %full_default_1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1], True), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.bfloat16), kwargs = {})
triton_red_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_red_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i64', 'r0_numel': 'i64', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2097152, 'r0_': 39560675328}}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 50304
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -1, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = -1.0
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp14 = (tmp11 / tmp13)
        tmp15 = tl.where(tmp2, tmp14, tmp8)
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp38 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp36 = tl.load(in_out_ptr0 + (r0_1 + 50304*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], -1, tl.int64)
        tmp21 = tmp0 != tmp20
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tl.where(tmp21, tmp0, tmp22)
        tmp24 = r0_1
        tmp25 = tmp23 == tmp24
        tmp26 = -1.0
        tmp27 = 0.0
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp33 = (tmp30 / tmp32)
        tmp34 = tl.where(tmp21, tmp33, tmp27)
        tmp35 = tmp28 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp39 = tmp37 - tmp38
        tmp41 = tmp39 - tmp40
        tmp42 = tl_math.exp(tmp41)
        tmp43 = tmp42 * tmp18
        tmp44 = tmp35 - tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 50304*x0), tmp45, r0_mask)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/qe/cqe74ouf5ka4lv27ymdxrhmfpotuot5wyvzzqtwwnfq7i7ba4s3i.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1006632960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /workspace/modded-nanogpt/.inductor_cache/kk/ckkilqwr5ppquctp4gj7kunyx6lz6sezmarpigxtztej2vsrdbgm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '0E240AAAE3E9D1980383EC2FCF1E55EF51BD69D1F11DF86C6B99F17FF7E9B37A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 386334720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38633472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, view, mm, amax, log, convert_element_type_5, permute_3, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (128, 1024), (1024, 1))
    assert_size_stride(view, (131072, 768), (768, 1))
    assert_size_stride(mm, (131072, 50304), (50304, 1))
    assert_size_stride(amax, (131072, 1), (1, 1))
    assert_size_stride(log, (131072, 1), (1, 1))
    assert_size_stride(convert_element_type_5, (), ())
    assert_size_stride(permute_3, (50304, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = reinterpret_tensor(mm, (128, 1024, 50304), (51511296, 50304, 1), 0); del mm  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0.run(buf1, primals_3, tangents_1, convert_element_type_5, amax, log, 131072, 50304, stream=stream0)
        del amax
        del convert_element_type_5
        del log
        del primals_3
        del tangents_1
        buf2 = empty_strided_cuda((50304, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (50304, 131072), (1, 50304), 0), view, out=buf2)
        del view
        buf3 = empty_strided_cuda((131072, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (131072, 50304), (50304, 1), 0), permute_3, out=buf3)
        del buf1
        del permute_3
        buf4 = empty_strided_cuda((128, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf3, buf4, 100663296, stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((50304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf2, buf5, 38633472, stream=stream0)
        del buf2
    return (buf4, buf5, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((131072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((131072, 50304), (50304, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((131072, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((131072, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((50304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_3, view, mm, amax, log, convert_element_type_5, permute_3, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
