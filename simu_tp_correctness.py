# test_tp_correctness.py
# torchrun --nproc_per_node=2 --master_port=31415 test_tp_correctness.py
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import os
#determind settings
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 必须在import torch前!
os.environ["PYTHONHASHSEED"] = "0"

import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import time
import math

def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29511'

    print(rank, local_rank, world_size, 'YYYYYYYYYYYYYYYYYYYYYYYY')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    print(rank, local_rank, world_size, 'ZZZZZZZZZZZZZZZZZZZZZZZZ')
    torch.cuda.set_device(local_rank)

    print(rank, local_rank, world_size, 'ZZZZZZZZZZZZZZZZZZZZZZZZ')

    return rank, local_rank, world_size

def test_tp_linear():
    """测试TP线性层的正确性"""
    rank, local_rank, world_size = dist.get_rank(), dist.get_rank(), dist.get_world_size()

    # 测试参数
    batch_size = 4
    in_features = 4096
    out_features = 4096
    dtype = torch.float16  # 测试fp16

    # 随机种子，确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if rank == 0:
        print(f"\n=== TP Correctness Test ===")
        print(f"Batch: {batch_size}, In: {in_features}, Out: {out_features}")
        print(f"Dtype: {dtype}, World size: {world_size}")

    # =========== 单卡计算（rank 0上执行） ===========
    if rank == 0:
        # 生成输入和权重
        x_single = torch.randn(batch_size, in_features, dtype=dtype, device='cuda:0')
        weight_single = torch.randn(out_features, in_features, dtype=dtype, device='cuda:0')
        bias_single = torch.randn(out_features, dtype=dtype, device='cuda:0')

        # 单卡前向计算
        output_single = F.linear(x_single, weight_single, bias_single)

        # 保存结果供对比
        torch.save({
            'x': x_single.cpu(),
            'weight': weight_single.cpu(),
            'bias': bias_single.cpu(),
            'output': output_single.cpu()
        }, 'single_gpu_result.pt')

        print("Single GPU computation completed")

    dist.barrier()

    # =========== TP分布式计算 ===========
    # 广播输入数据（所有rank使用相同的输入）
    if rank == 0:
        x_tp = x_single
        weight_tp = weight_single
        bias_tp = bias_single
    else:
        x_tp = torch.empty(batch_size, in_features, dtype=dtype, device=f'cuda:{local_rank}')
        weight_tp = torch.empty(out_features, in_features, dtype=dtype, device=f'cuda:{local_rank}')
        bias_tp = torch.empty(out_features, dtype=dtype, device=f'cuda:{local_rank}')

    # 广播数据到所有rank
    if rank == 0:
        dist.broadcast(x_tp, src=0)
        dist.broadcast(weight_tp, src=0)
        dist.broadcast(bias_tp, src=0)
    else:
        dist.broadcast(x_tp, src=0)
        dist.broadcast(weight_tp, src=0)
        dist.broadcast(bias_tp, src=0)

    # 按列切分权重和偏置（TP切分）
    split_dim = 0  # 按输出维度切分
    weight_chunks = torch.chunk(weight_tp, world_size, dim=split_dim)
    bias_chunks = torch.chunk(bias_tp, world_size, dim=split_dim)

    # 每个rank处理自己的部分
    local_weight = weight_chunks[rank].contiguous()
    local_bias = bias_chunks[rank].contiguous()

    # 局部计算
    local_output = F.linear(x_tp, local_weight, local_bias)

    # 收集所有rank的结果
    output_chunks = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(output_chunks, local_output)

    # 拼接得到完整输出
    output_tp = torch.cat(output_chunks, dim=-1)  # 沿最后一个维度拼接

    # =========== 验证结果 ===========
    if rank == 0:
        # 加载单卡结果
        single_data = torch.load('single_gpu_result.pt')
        output_single = single_data['output'].cuda()

        # 计算数值差异
        diff = torch.abs(output_tp - output_single)

        print(f"\n=== Linear单步验证结果 ===")
        print(f"Local计算输出形状: {local_output.shape}")
        print(f"单卡输出形状: {output_single.shape}")

        # 统计指标
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()

        print(f"最大绝对误差: {max_diff:.6e}")
        print(f"平均绝对误差: {mean_diff:.6e}")
        print(f"误差标准差: {std_diff:.6e}")

        # 相对误差
        rel_diff = diff / (torch.abs(output_single) + 1e-8)
        max_rel_diff = rel_diff.max().item()

        print(f"最大相对误差: {max_rel_diff:.6e}")

        # 检查是否在可接受范围内
        # fp16的典型误差范围：1e-3 到 1e-4
        tolerance = 1e-3 if dtype == torch.float16 else 1e-5

        if max_diff < tolerance:
            print("✅ TP计算结果与单卡一致，测试通过！")
        else:
            print("❌ TP计算结果与单卡有显著差异！")

        return max_diff < tolerance

    return True

def test_tp_matmul_precision():
    """测试不同精度下的矩阵乘法"""
    rank, local_rank, world_size = dist.get_rank(), dist.get_rank(), dist.get_world_size()

    if rank == 0:
        print(f"\n=== 不同精度测试 ===")

    dtypes = [
        (torch.float32, "fp32"),
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16")
    ]

    results = {}

    for dtype, dtype_name in dtypes:
        # 生成测试数据
        torch.manual_seed(42)
        A = torch.randn(1024, 1024, dtype=dtype, device=f'cuda:{local_rank}') +1.
        B = torch.randn(1024, 1024, dtype=dtype, device=f'cuda:{local_rank}') +1.

        # 单卡计算（在rank 0上）
        if rank == 0:
            C_single = torch.mm(A, B)

        # TP计算：切分矩阵B
        B_chunks = torch.chunk(B, world_size, dim=1)  # 按列切分
        local_B = B_chunks[rank]

        # 局部计算
        local_C = torch.mm(A, local_B)

        # All-Gather结果
        C_chunks = [torch.empty_like(local_C) for _ in range(world_size)]
        dist.all_gather(C_chunks, local_C)

        # 拼接
        C_tp = torch.cat(C_chunks, dim=1)

        # 验证
        if rank == 0:
            print(f"\n=== Matmal AxB {dtype_name} - 验证结果 ===")
            print(f"A, local_B形状: {A.shape, local_B.shape}")
            print(f"Local计算输出形状: {local_C.shape}")
            print(f"单卡输出形状: {C_tp.shape}")
            diff = torch.abs(C_tp - C_single)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"{dtype_name}: 最大误差={max_diff:.2e}, 平均误差={mean_diff:.2e}")

            diff = torch.abs(diff / C_tp)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"{dtype_name}: 最大相对误差={max_diff:.2e}, 平均相对误差={mean_diff:.2e}")

            results[dtype_name] = max_diff

            # 保存用于分析
            torch.save({
                'single': C_single.cpu(),
                'tp': C_tp.cpu(),
                'diff': diff.cpu()
            }, f'precision_test_{dtype_name}.pt')

        dist.barrier()

    return results

def test_llm_style_tp_3matmul(
    matrix_size: int = 1024,
    num_iterations: int = 5,
    save_results: bool = True,
    amplify_values: bool = False,
    use_bf16: bool = False  # 是否使用BF16（Ampere+架构专用）
) -> Dict[str, Dict[str, List[float]]]:
    """
    测试LLM风格的Tensor Parallelism (A@B@C)
    切分策略:
      - B按列切分 (dim=1) -> 每个rank得到B_i
      - C按行切分 (dim=0) -> 每个rank得到C_i
      - 每个rank计算: Y_i = A @ B_i @ C_i
      - All-Reduce (SUM) 聚合: Y_tp = sum(Y_i)

    参数:
        matrix_size: 方阵维度 (N x N)
        num_iterations: 重复测试次数 (验证确定性)
        save_results: 保存详细结果
        amplify_values: 放大数值范围暴露精度问题
        use_bf16: 是否测试bf16 (需要Ampere+ GPU)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"LLM风格Tensor Parallelism测试 (A@B@C)")
        print(f"切分策略: B按列切分 | C按行切分 | 最终All-Reduce(SUM)")
        print(f"矩阵尺寸: {matrix_size}x{matrix_size} | 卡数: {world_size} | 迭代次数: {num_iterations}")
        print(f"放大数值范围: {'启用' if amplify_values else '禁用'}")
        print(f"{'='*60}\n")

    # 支持的数据类型 (根据use_bf16动态调整)
    dtypes = [(torch.float32, "fp32"), (torch.float16, "fp16")]
    if use_bf16 and torch.cuda.is_bf16_supported():
        dtypes.append((torch.bfloat16, "bf16"))

    # 存储结果
    all_results = {name: {"abs_err": [], "rel_err": []} for _, name in dtypes}

    # 固定种子 (所有rank一致)
    # # determind settings
    torch.manual_seed(42)
    np.random.seed(42)

    cons_mean = 1.

 # === 关键修复: 在GPU上生成和广播数据 ===
    # 所有rank创建相同形状的GPU张量 (float32作为基础类型)
    A = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float32)
    B = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float32)
    C = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float32)

    # 仅rank 0生成数据
    if rank == 0:
        torch.manual_seed(42)
        np.random.seed(42)

        # 在CPU生成再移到GPU (确保确定性)
        scale = math.sqrt(2 / matrix_size)  # He初始化标准差
        A_cpu = torch.randn(matrix_size, matrix_size) + cons_mean
        B_cpu = torch.randn(matrix_size, matrix_size) * scale
        C_cpu = torch.randn(matrix_size, matrix_size) * scale

        if amplify_values:
            scale = 1000
            offset = 500
            A_cpu = A_cpu * scale + offset
            B_cpu = B_cpu * scale + offset
            C_cpu = C_cpu * scale + offset

        # 复制到GPU张量
        A.copy_(A_cpu.to(device))
        B.copy_(B_cpu.to(device))
        C.copy_(C_cpu.to(device))

    # === 广播GPU张量 (NCCL要求) ===
    dist.broadcast(A, src=0)
    dist.broadcast(B, src=0)
    dist.broadcast(C, src=0)

    # 测试每种数据类型
    for dtype, dtype_name in dtypes:
        if rank == 0:
            print(f"\n{'-'*45}")
            print(f" 精度类型: {dtype_name.upper()} ".center(45))
            print(f"{'-'*45}")

               # 转换为当前测试精度
        A_test = A.to(dtype=dtype)
        B_test = B.to(dtype=dtype)
        C_test = C.to(dtype=dtype)

        # 单卡参考结果 (仅rank0计算)
        if rank == 0:
            Y_single = A_test @ B_test @ C_test

        # 多次迭代测试
        for iter_idx in range(num_iterations):
            if rank == 0 and num_iterations > 1:
                print(f"\n>>> 迭代 {iter_idx+1}/{num_iterations}")

            # === LLM风格TP计算 ===
            # 1. 切分B (按列) 和 C (按行)
            B_chunks = torch.chunk(B_test, world_size, dim=1)
            C_chunks = torch.chunk(C_test, world_size, dim=0)

            # 2. 获取当前rank的切片
            local_B = B_chunks[rank]
            local_C = C_chunks[rank]

            # 3. 尺寸分析 (仅首次迭代的rank0)
            if rank == 0 and iter_idx == 0:
                print(f"\n[尺寸分析] Rank 0:")
                print(f"  A: {list(A_test.shape)} | 完整输入矩阵")
                print(f"  B_chunks[0] (local_B): {list(local_B.shape)} | B的第0个列切片")
                print(f"  B_chunks[1] (其他rank): {list(B_chunks[1].shape) if world_size > 1 else 'N/A'}")
                print(f"  C_chunks[0] (local_C): {list(local_C.shape)} | C的第0个行切片")
                print(f"  C_chunks[1] (其他rank): {list(C_chunks[1].shape) if world_size > 1 else 'N/A'}")

            # 4. 本地计算: Y_i = A @ B_i @ C_i
            local_Y = A_test @ local_B @ local_C

            if rank == 0 and iter_idx == 0:
                print(f"  local_Y (A@B_i@C_i): {list(local_Y.shape)} | 本地中间结果")

            # 5. All-Reduce (SUM) 聚合最终结果
            dist.all_reduce(local_Y, op=dist.ReduceOp.SUM)
            Y_tp = local_Y

            if rank == 0 and iter_idx == 0:
                print(f"  Y_tp (All-Reduce后): {list(Y_tp.shape)} | 完整输出矩阵")

            # 6. 验证结果 (仅rank0)
            if rank == 0:
                # 确保比较精度一致
                Y_single_iter = Y_single.to(dtype=dtype)
                Y_tp_iter = Y_tp.to(dtype=torch.float32)

                # 计算误差
                abs_diff = torch.abs(Y_tp_iter - Y_single_iter)
                max_abs_err = abs_diff.max().item()
                mean_abs_err = abs_diff.mean().item()

                # 避免除零
                eps = 1e-8 if dtype == torch.float32 else 1e-4
                rel_diff = abs_diff / (torch.abs(Y_single_iter) + eps)
                max_rel_err = rel_diff.max().item()
                mean_rel_err = rel_diff.mean().item()

                # 记录结果
                all_results[dtype_name]["abs_err"].append(max_abs_err)
                all_results[dtype_name]["rel_err"].append(max_rel_err)

                # 打印结果
                print(f"\n[迭代 {iter_idx+1}] {dtype_name} 精度结果:")
                print(f"  最大绝对误差: {max_abs_err:.4e} | 平均绝对误差: {mean_abs_err:.4e}")
                print(f"  最大相对误差: {max_rel_err:.4e} | 平均相对误差: {mean_rel_err:.4e}")

                # 保存详细结果 (仅最后一次迭代)
                if save_results and iter_idx == num_iterations - 1:
                    # 创建保存目录
                    os.makedirs("tp_results", exist_ok=True)
                    save_path = f'tp_results/llm_tp_{dtype_name}_w{world_size}_iter{num_iterations}.pt'

                    # 为保存准备CPU张量
                    torch.save({
                        'config': {
                            'matrix_size': matrix_size,
                            'world_size': world_size,
                            'dtype': dtype_name,
                            'amplify_values': amplify_values,
                            'tp_strategy': 'B_col_split_C_row_split'
                        },
                        'results': {
                            'Y_single': Y_single.cpu(),
                            'Y_tp': Y_tp.cpu(),
                            'abs_diff': abs_diff.cpu(),
                            'rel_diff': rel_diff.cpu()
                        },
                        'errors': {
                            'max_abs_err': max_abs_err,
                            'mean_abs_err': mean_abs_err,
                            'max_rel_err': max_rel_err,
                            'mean_rel_err': mean_rel_err
                        }
                    }, save_path)
                    print(f"  详细结果已保存至: {save_path}")

            dist.barrier()

        # 打印统计摘要
        if rank == 0:
            abs_errs = all_results[dtype_name]["abs_err"]
            rel_errs = all_results[dtype_name]["rel_err"]

            print(f"\n{'='*35}")
            print(f" {dtype_name.upper()} 精度统计摘要 ".center(35))
            print(f"{'='*35}")
            print(f"绝对误差 (最大值):")
            print(f"  平均: {np.mean(abs_errs):.4e} | 标准差: {np.std(abs_errs):.4e}")
            print(f"  范围: [{min(abs_errs):.4e}, {max(abs_errs):.4e}]")
            print(f"相对误差 (最大值):")
            print(f"  平均: {np.mean(rel_errs):.4e} | 标准差: {np.std(rel_errs):.4e}")
            print(f"  范围: [{min(rel_errs):.4e}, {max(rel_errs):.4e}]")

        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print("LLM风格TP测试完成! 结果摘要:")
        for dtype_name in all_results.keys():
            abs_errs = all_results[dtype_name]["abs_err"]
            rel_errs = all_results[dtype_name]["rel_err"]
            print(f"\n{dtype_name.upper()}:")
            print(f"  最大绝对误差范围: [{min(abs_errs):.4e}, {max(abs_errs):.4e}]")
            print(f"  最大相对误差范围: [{min(rel_errs):.4e}, {max(rel_errs):.4e}]")
        print(f"{'='*60}\n")

    return all_results








def main():

    # print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


    rank, local_rank, world_size = setup_distributed()

    print(f"Rank {rank} (local rank {local_rank}) initialized")

    try:
        # 运行测试
        test_tp_linear()

        test_tp_matmul_precision()

        # test_tp_with_gradient()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # 确保确定性 (重要!)
        # determind settings

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # 运行测试
        results = test_llm_style_tp_3matmul(
            matrix_size=1024,
            num_iterations=5,
            save_results=False,
            amplify_values=False,  # 启用放大以暴露精度问题
            use_bf16=True         # 如果GPU支持bf16 (Ampere+)
        )


    except Exception as e:
        print(f"Rank {rank}: Error - {e}")
        import traceback
        traceback.print_exc()
    finally:
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()