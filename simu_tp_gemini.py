# torchrun --nproc_per_node=2 --master_port=31415 test_tp_gemini.py

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import os
import math
import time
from typing import Dict, List, Optional, Callable

# 尝试导入表格库，如果没有则降级处理
try:
    import pandas as pd
    from tabulate import tabulate
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ================= 基础工具函数 =================
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import binned_statistic
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from typing import Union


def plot_error_distribution(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title: str,
    save_dir: str = "tp_plots",
    smooth_sigma: float = 3.0,
    max_samples: int = 100000,
    n_bins: int = 50,
    min_valid_error: float = 1e-9,
    log_base: float = 10.0,
) -> None:
    """
    绘制输出幅度 vs 最大相对误差的包络图（Envelope Plot）

    参数:
        y_true (torch.Tensor): 真实标签。
        y_pred (torch.Tensor): 预测值。
        title (str): 图表标题。
        save_dir (str): 保存目录。
        smooth_sigma (float): 高斯平滑标准差。
        max_samples (int): 散点采样数。
        n_bins (int): 分箱数。
        min_valid_error (float): 有效误差下限。
        log_base (float): 对数底数。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # === 1. 数据准备 ===
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes do not match: {y_true.shape} vs {y_pred.shape}")

    y_t = y_true.float().cpu().numpy().flatten()
    y_p = y_pred.float().cpu().numpy().flatten()

    x_val = np.abs(y_t) + 1e-12
    y_val = np.abs(y_t - y_p) / (np.abs(y_t) + 1e-9)

    # === 2. 散点采样 ===
    if len(x_val) > max_samples:
        idx = np.random.choice(len(x_val), max_samples, replace=False)
        x_scatter, y_scatter = x_val[idx], y_val[idx]
    else:
        x_scatter, y_scatter = x_val, y_val

    # === 3. 计算最大包络 ===
    min_exp = np.floor(np.log10(x_val.min()))
    max_exp = np.ceil(np.log10(x_val.max()))
    bins = np.logspace(min_exp, max_exp, n_bins)

    bin_maxs, bin_edges, _ = binned_statistic(x_val, y_val, statistic='max', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 排除 NaN 和 Inf
    valid_mask = (~np.isnan(bin_maxs)) & (~np.isinf(bin_maxs)) & (bin_maxs > 0)
    if not np.any(valid_mask):
        print(f"⚠️ 无有效误差数据，跳过绘图: {title}")
        return

    x_valid = bin_centers[valid_mask]
    y_valid = bin_maxs[valid_mask]

    # 限制最小有效误差
    y_valid[y_valid < min_valid_error] = min_valid_error

    # === 4. 平滑曲线 ===
    y_smooth = gaussian_filter1d(y_valid, sigma=smooth_sigma)
    x_smooth = x_valid

    # === 5. 绘图 ===
    plt.figure(figsize=(10, 6), dpi=150)

    # 背景散点
    plt.scatter(x_scatter, y_scatter, s=3, alpha=0.15, color='#1f77b4', rasterized=True, label='Raw Errors')

    # 包络线
    plt.plot(x_smooth, y_smooth, color='#D62728', linewidth=3, alpha=0.8, label='Max Error Envelope (Smoothed)')

    # 设置坐标轴
    plt.xscale('log')
    plt.yscale('log')

    # 标注
    plt.xlabel('Output Magnitude (|y|) - Log Scale', fontsize=12)
    plt.ylabel('Relative Error (Max) - Log Scale', fontsize=12)
    plt.title(f'{title}\nMaximum Error Envelope', fontsize=14)
    plt.grid(True, which="major", alpha=0.3)
    plt.legend(loc='upper right')

    # 保存
    clean_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace("@", "at")
    path = os.path.join(save_dir, f"{clean_title}_envelope.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"   📈 包络线绘图完成: {path}")
# def plot_error_distribution(y_true, y_pred, title, save_dir="tp_plots"):
#     """
#     绘制 Output Magnitude vs Relative Error 的散点趋势图

#     参数:
#         y_true: Baseline tensor (理论真值)
#         y_pred: Comparison tensor (TP模拟值)
#         title: 图表标题
#         save_dir: 保存目录
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 1. 数据预处理
#     # 转为 CPU numpy，展平
#     y_t = y_true.detach().float().cpu().numpy().flatten()
#     y_p = y_pred.detach().float().cpu().numpy().flatten()

#     # 计算绝对值大小 (X轴) 和 相对误差 (Y轴)
#     # 加上极小值 1e-12 防止 log(0) 报错
#     abs_values = np.abs(y_t) + 1e-12
#     abs_diff = np.abs(y_t - y_p)
#     rel_err = abs_diff / abs_values

#     # 2. 采样 (Downsample)
#     # 100万个点画散点图太慢，随机采样 50,000 个点即可看清分布
#     n_samples = 100000
#     if len(y_t) > n_samples:
#         print('XXXXX   --  ', len(y_t))
#         indices = np.random.choice(len(y_t), n_samples, replace=False)
#         x_plot = abs_values[indices]
#         y_plot = rel_err[indices]
#     else:
#         x_plot = abs_values
#         y_plot = rel_err

#     # 3. 计算趋势线 (Binned Median)
#     # 将 X 轴数据在 Log 空间均匀切分为 50 个桶，看每个桶里的误差中位数
#     # 使用 Log space bins
#     min_exp = np.floor(np.log10(x_plot.min()))
#     max_exp = np.ceil(np.log10(x_plot.max()))
#     bins = np.logspace(min_exp, max_exp, 50)

#     bin_means, bin_edges, _ = binned_statistic(x_plot, y_plot, statistic='max', bins=bins)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#     # 4. 开始绘图
#     plt.figure(figsize=(10, 6), dpi=150)

#     # A. 绘制散点 (浅蓝色，透明度高，作为背景)
#     plt.scatter(x_plot, y_plot, alpha=0.3, s=3, color='#1f77b4', label='Individual Points', rasterized=True)

#     # B. 绘制趋势线 (红色，加粗，作为核心结论)
#     plt.plot(bin_centers, bin_means, 'r-', linewidth=2.5, label='Maximum Error Trend')

#     # C. 设置坐标轴为对数坐标
#     plt.xscale('log')
#     plt.yscale('log')

#     # D. 装饰
#     plt.xlabel('Output Magnitude (|y|) - Log Scale', fontsize=12)
#     plt.ylabel('Relative Error - Log Scale', fontsize=12)
#     plt.title(f'{title}\nError vs Magnitude Analysis', fontsize=14)
#     plt.grid(True, which="both", ls="-", alpha=0.2)
#     plt.legend(loc='upper right')

#     # E. 标注精度参考线 (可选)
#     # plt.axhline(y=1e-3, color='g', linestyle='--', alpha=0.5, label='FP16 Precision (~1e-3)')

#     # 保存
#     clean_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace("@", "at")
#     save_path = os.path.join(save_dir, f"{clean_title}_dist.png")
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"   📈 绘图完成: {save_path}")

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' not in os.environ:
        # 本地调试用默认值
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        print("⚠️ 未检测到分布式环境，使用单机模拟模式 (World Size=1)")

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def set_deterministic(seed=42):
    """强制确定性，确保 Baseline 和 TP 的输入初始化完全一致"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ================= 核心测试框架类 =================

class TPBenchmarkRunner:
    def __init__(self, save_dir="tp_artifacts"):
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.results_log = [] # 存储所有测试结果
        self.save_dir = save_dir

        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"🚀 TP Benchmark Runner Initialized")
            print(f"   World Size: {self.world_size} | Device: {self.device}")
            print(f"   Artifacts Dir: {self.save_dir}")
            print(f"{'='*60}\n")

    def _reset_seeds(self, seed=42):
        """每次测试前重置种子"""
        set_deterministic(seed)

    def _record_result(self, test_name, dtype, stage, abs_diff,abs_diff_mean, rel_diff, rel_diff_mean, output_tensor=None):
        """记录单次测试结果"""
        self.results_log.append({
            "Test Name": test_name,
            "Dtype": str(dtype).split('.')[-1],
            "Comparison": stage,
            "Max Abs Diff": abs_diff,
            "Mean Abs Diff": abs_diff_mean,
            "Max Rel Diff": rel_diff,
            "Mean Rel Diff": rel_diff_mean
        })

        # 保存 Tensor 用于溯源 (可选，仅在 Rank 0)
        if self.rank == 0 and output_tensor is not None:
            fname = f"{test_name}_{str(dtype).split('.')[-1]}_{stage.replace(' ', '_')}.pt"
            path = os.path.join(self.save_dir, fname)
            # torch.save(output_tensor.cpu(), path)

    def _compare_tensors(self, name, dtype, tensor_a, tensor_b, tag_a, tag_b):
        """对比两个 Tensor 并返回误差"""
        if self.rank != 0: return

        # 统一转 float32 比较以免溢出
        a_f32 = tensor_a.float()
        b_f32 = tensor_b.float()

        abs_diff = (a_f32 - b_f32).abs().max().item()
        abs_diff_mean = (a_f32 - b_f32).abs().mean().item()
        # 相对误差处理除零
        denominator = b_f32.abs() + 1e-8
        rel_diff = ((a_f32 - b_f32).abs() / denominator).max().item()
        rel_diff_mean = ((a_f32 - b_f32).abs() / denominator).mean().item()

        # === 新增：只有当有显著误差时，才画图分析 ===
        # 例如：如果是 Simulated vs Real TP (全是0)，就没必要画了，浪费时间
        # 但 Baseline vs Simulated (有数学误差) 一定要画
        if "Baseline" in tag_a and abs_diff > 1e-6:
            plot_title = f"{name} [{str(dtype).split('.')[-1]}]"
            # 调用绘图函数
            plot_error_distribution(a_f32, b_f32, plot_title, self.save_dir)
        # ==========================================

        self._record_result(name, dtype, f"{tag_a} vs {tag_b}", abs_diff,abs_diff_mean, rel_diff, rel_diff_mean, tensor_b)
        return abs_diff

    # ================= 测试场景 1: Linear (Column Parallel) =================
    # 对应你原来的 test_tp_linear
    # 场景: 输出维度被切分，结果需要 All-Gather 拼接

    def run_case_linear_col_parallel(self, batch_size=4, in_feat=4096, out_feat=4096, dtype=torch.float16):
        test_name = "Linear(ColParallel)"
        if self.rank == 0: print(f"RUNNING: {test_name} [{dtype}]...")
        self._reset_seeds()

        # 1. 准备全局数据 (Rank 0 生成，广播)
        if self.rank == 0:
            x = torch.randn(batch_size, in_feat, device=self.device, dtype=torch.float32)
            w = torch.randn(out_feat, in_feat, device=self.device, dtype=torch.float32)
            b = torch.randn(out_feat, device=self.device, dtype=torch.float32)
        else:
            x = torch.zeros(batch_size, in_feat, device=self.device, dtype=torch.float32)
            w = torch.zeros(out_feat, in_feat, device=self.device, dtype=torch.float32)
            b = torch.zeros(out_feat, device=self.device, dtype=torch.float32)

        dist.broadcast(x, 0); dist.broadcast(w, 0); dist.broadcast(b, 0)
        x, w, b = x.to(dtype), w.to(dtype), b.to(dtype)

        # --- A. Baseline (单卡标准) ---
        res_baseline = None
        if self.rank == 0:
            res_baseline = F.linear(x, w, b)

        # --- B. Simulation (单卡模拟切分) ---
        res_sim = None
        if self.rank == 0:
            # 模拟 Column Parallel: 切分 Weight 的 output dim (dim 0), Bias 也要切
            w_chunks = w.chunk(self.world_size, dim=0)
            b_chunks = b.chunk(self.world_size, dim=0)

            sim_outputs = []
            for i in range(self.world_size):
                # 模拟不同 Rank 的计算
                sim_outputs.append(F.linear(x, w_chunks[i], b_chunks[i]))

            # Column Parallel 的合并方式是 Concat
            res_sim = torch.cat(sim_outputs, dim=-1)

        # --- C. Real TP (真实分布式) ---
        # 1. 切分数据
        w_local_chunks = w.chunk(self.world_size, dim=0)
        b_local_chunks = b.chunk(self.world_size, dim=0)
        w_local = w_local_chunks[self.rank]
        b_local = b_local_chunks[self.rank]

        # 2. 本地计算
        y_local = F.linear(x, w_local, b_local)

        # 3. 通信 (All-Gather)
        y_gathered_list = [torch.zeros_like(y_local) for _ in range(self.world_size)]
        dist.all_gather(y_gathered_list, y_local)
        res_real = torch.cat(y_gathered_list, dim=-1)

        # --- 验证 ---
        if self.rank == 0:
            self._compare_tensors(test_name, dtype, res_baseline, res_sim, "Baseline(TP1)", "Simulated")
            self._compare_tensors(test_name, dtype, res_sim, res_real, "Simulated", "Real TP")

    # ================= 测试场景 2: MatMul (Row Parallel Sum) =================
    # 对应你原来的 test_tp_matmul_precision
    # 场景: 输入维度被切分 (K维)，结果需要 All-Reduce (Sum)

    def run_case_matmul_row_parallel(self, N=1024, mean = 0., dtype=torch.float16):
        test_name = f"MatMul(A@B), Mean = {mean}"
        if self.rank == 0: print(f"RUNNING: {test_name} [{dtype}]...")
        self._reset_seeds()

        # Data Generation
        if self.rank == 0:
            A = torch.randn(N, N, device=self.device).to(dtype) + mean
            B = torch.randn(N, N, device=self.device).to(dtype)
        else:
            A = torch.zeros(N, N, device=self.device, dtype=dtype)
            B = torch.zeros(N, N, device=self.device, dtype=dtype)
        dist.broadcast(A, 0); dist.broadcast(B, 0)

        # --- A. Baseline ---
        res_base = None
        if self.rank == 0:
            res_base = torch.mm(A, B)

        # --- B. Simulation ---
        res_sim = None
        if self.rank == 0:
            # Row Parallel: A 按列切 (dim 1), B 按行切 (dim 0) -> 结果 Sum
            A_chunks = A.chunk(self.world_size, dim=1)
            B_chunks = B.chunk(self.world_size, dim=0)

            partials = []
            for i in range(self.world_size):
                partials.append(torch.mm(A_chunks[i], B_chunks[i]))
            res_sim = sum(partials) # 模拟 Reduce Sum

        # --- C. Real TP ---
        # 准备分片
        A_local = A.chunk(self.world_size, dim=1)[self.rank]
        B_local = B.chunk(self.world_size, dim=0)[self.rank]

        # 计算与通信
        y_local = torch.mm(A_local, B_local)
        dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
        res_real = y_local

        # --- 验证 ---
        if self.rank == 0:
            self._compare_tensors(test_name, dtype, res_base, res_sim, "Baseline(TP1)", "Simulated")
            self._compare_tensors(test_name, dtype, res_sim, res_real, "Simulated", "Real TP")

    # ================= 测试场景 3: MLP Chain (A@B@C) =================
    # 对应你原来的 test_llm_style_tp_3matmul (MLP Style)
    # 流程: X -> [Col Split] -> Y_mid -> [Row Split] -> Y_out -> AllReduce

    def run_case_mlp_chain(self, size=1024, use_relu=False, mean = 0., dtype=torch.bfloat16):
        test_name = f"MLP( sigma(x@A)@B ), Mean = {mean}"
        if self.rank == 0: print(f"RUNNING: {test_name} [{dtype}]...")
        self._reset_seeds()

        # X, A(Up_proj), B(Down_proj)
        # 假设 A 是 expand (1024->4096), B 是 shrink (4096->1024)
        # 为了简化和你之前的例子一致，我们用 size x size
        hidden_size = size
        inter_size = size

        if self.rank == 0:
            X = torch.randn(1, hidden_size, device=self.device).to(dtype) + mean
            W_up = torch.randn(hidden_size, inter_size, device=self.device).to(dtype) * math.sqrt(2.0 / hidden_size) # A
            W_down = torch.randn(inter_size, hidden_size, device=self.device).to(dtype) * math.sqrt(2.0 / hidden_size)# B
        else:
            X = torch.zeros(1, hidden_size, device=self.device, dtype=dtype)
            W_up = torch.zeros(hidden_size, inter_size, device=self.device, dtype=dtype)
            W_down = torch.zeros(inter_size, hidden_size, device=self.device, dtype=dtype)

        dist.broadcast(X, 0); dist.broadcast(W_up, 0); dist.broadcast(W_down, 0)

        # --- A. Baseline ---
        res_base = None
        if self.rank == 0:
            # X @ A @ B
            mid = torch.mm(X, W_up)
            if use_relu:
                mid = F.relu(mid)
            res_base = torch.mm(mid, W_down)

        # --- B. Simulation ---
        res_sim = None
        if self.rank == 0:
            # 1. Col Parallel (W_up): W按列切
            W_up_chunks = W_up.chunk(self.world_size, dim=1)
            # 2. Row Parallel (W_down): W按行切
            W_down_chunks = W_down.chunk(self.world_size, dim=0)

            partials = []
            for i in range(self.world_size):
                # Local Path: X @ W_up_i -> W_down_i
                mid_i = torch.mm(X, W_up_chunks[i])
                if use_relu:
                    mid_i = F.relu(mid_i) # 激活函数本地做
                out_i = torch.mm(mid_i, W_down_chunks[i])
                partials.append(out_i)

            res_sim = sum(partials) # AllReduce Sum

        # --- C. Real TP ---
        # 准备分片
        W_up_local = W_up.chunk(self.world_size, dim=1)[self.rank]
        W_down_local = W_down.chunk(self.world_size, dim=0)[self.rank]

        # 本地计算
        mid_local = torch.mm(X, W_up_local)
        if use_relu:
            mid_local = F.relu(mid_local)
        out_local = torch.mm(mid_local, W_down_local)

        # 通信
        dist.all_reduce(out_local, op=dist.ReduceOp.SUM)
        res_real = out_local

        # --- 验证 ---
        if self.rank == 0:
            self._compare_tensors(test_name, dtype, res_base, res_sim, "Baseline(TP1)", "Simulated")
            self._compare_tensors(test_name, dtype, res_sim, res_real, "Simulated", "Real TP")

    # ================= 报告生成 =================

    def generate_report(self):
        if self.rank != 0: return

        print(f"\n{'='*30} TEST REPORT {'='*30}")

        if HAS_PANDAS:
            df = pd.DataFrame(self.results_log)
            # 格式化一下数字显示
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2e", showindex=False))

            # 保存 CSV
            csv_path = os.path.join(self.save_dir, "tp_benchmark_summary.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n📊 详细报告已保存至: {csv_path}")
            print(f"💾 中间 Tensor 已保存至: {self.save_dir}/")
        else:
            # 简单的 Fallback 打印
            print(f"{'Test':<20} | {'Dtype':<6} | {'Compare':<20} | {'Max Abs Err':<12} | {'Mean Abs Err':<12} | {'Max Rel Err':<12} {'Mean Rel Err':<12} |")
            print("-" * 80)
            for res in self.results_log:
                print(f"{res['Test Name']:<20} | {res['Dtype']:<6} | {res['Comparison']:<20} | {res['Max Abs Diff']:.2e}     |  {res['Mean Abs Diff']:.2e}     |   {res['Max Rel Diff']:.2e}  | {res['Mean Rel Diff']:.2e}     |")

# ================= 主程序入口 =================

def main():
    runner = TPBenchmarkRunner()

    # 1. 运行 Linear 测试 (Col Parallel)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        runner.run_case_linear_col_parallel(dtype=dtype)

    # 2. 运行 MatMul 测试 (Row Parallel)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        runner.run_case_matmul_row_parallel(dtype=dtype)
        runner.run_case_matmul_row_parallel(mean= 1., dtype=dtype)



    # 3. 运行 MLP Chain 测试
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        runner.run_case_mlp_chain(use_relu=True, mean=0., dtype=dtype)
        runner.run_case_mlp_chain(use_relu=True, mean= .1, dtype=dtype)




    # 4. 生成汇总表格
    runner.generate_report()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()