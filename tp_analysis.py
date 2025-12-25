import numpy as np
import torch

import os
# from vllm import LLM, SamplingParams
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_and_visualize_logits(df_merged, tp1_name='TP1', tp2_name='TP2'):
    """
    分析和可视化两个时间点(TP1, TP2)之间的logits差异

    Parameters:
    df_merged: 包含logits数据的DataFrame
    tp1_name: 第一个时间点的名称，默认为'TP1'
    tp2_name: 第二个时间点的名称，默认为'TP2'

    Returns:
    None (直接显示图表和打印报告)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # === 3. 计算核心指标 ===

    # A. 计算 Logits 绝对误差 (数值差异)
    # 我们对比 Top1_Logprob 的差异
    df_merged['Logits_Diff'] = (df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']).abs()

    # B. 检查 Token ID 是否翻转 (排序差异)
    # 如果 ID 不一样，说明微小的误差导致模型选了不同的词
    df_merged['Token_Mismatch'] = df_merged[f'Top1_ID_{tp1_name}'] != df_merged[f'Top1_ID_{tp2_name}']

    # C. 为了画图方便，创建一个全局的 Step 计数 (Global Step)
    df_merged['Global_Step'] = df_merged.index

    # === 4. 可视化绘图 ===
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 10))

    # --- 图 1: Logits 误差散点图 (The "Noise" Plot) ---
    plt.subplot(2, 1, 1)

    # 画散点，颜色根据 Question_ID 区分
    scatter = sns.scatterplot(
        data=df_merged,
        x='Global_Step',
        y='Logits_Diff',
        hue='Question_ID',
        palette='tab10',
        s=60,
        alpha=0.7,
        edgecolor='w'
    )

    # 关键：使用对数坐标轴，因为误差通常极小
    plt.yscale('log')

    plt.title(f' {tp1_name} vs {tp2_name} Logits Difference (Floating Point Error Analysis)', fontsize=15)
    plt.ylabel('Abs Difference (Log Scale)', fontsize=12)
    plt.xlabel('Token Sequence (Across all questions)', fontsize=12)
    plt.axhline(y=1e-5, color='r', linestyle='--', label='1e-5 Threshold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Question ID")
    plt.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # === 5. 文字报告 ===
    print("====== 📝 实验结果分析报告 ======")
    print(f"总计分析 Token 数: {len(df_merged)}")
    print(f"最大 Logits 误差: {df_merged['Logits_Diff'].max():.2e}")
    print(f"平均 Logits 误差: {df_merged['Logits_Diff'].mean():.2e}")
    print("-" * 30)

    mismatches = df_merged[df_merged['Token_Mismatch']]
    if len(mismatches) > 0:
        print(f"⚠️ 警告: 发现 {len(mismatches)} 个 Token 发生了选择翻转 (Butterfly Effect)!")
        print("翻转详情 (前5个):")
        print(mismatches[['Question_ID', 'Step_Index', f'Top1_Text_{tp1_name}', f'Top1_Text_{tp2_name}', 'Logits_Diff']].head())
    else:
        print("✅ 完美一致: 尽管存在浮点误差，但 {} 和 {} 选择的 Token 序列完全一样 (0 翻转)。".format(tp1_name, tp2_name))
        print("结论: 这种微小的误差 (Atomic Add 导致) 未影响生成结果。")

def compare_logprob_and_prob(df_merged, tp1_name='TP1', tp2_name='TP2'):
    """
    比较两个时间点(TP1, TP2)之间的logprob和probability差异

    Parameters:
    df_merged: 包含logits数据的DataFrame
    tp1_name: 第一个时间点的名称，默认为'TP1'
    tp2_name: 第二个时间点的名称，默认为'TP2'

    Returns:
    None (直接显示图表和打印报告)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 确保 df_merged 还在内存里，如果不在请重新运行上一段的"读取"部分
    if df_merged is None:
        print("请先提供 df_merged 数据！")
        return

    # 1. 计算线性概率 (Probability = exp(Logprob))
    # 这代表模型认为这个词出现的真实概率 (0% - 100%)
    df_merged[f'Prob_{tp1_name}'] = np.exp(df_merged[f'Top1_Logprob_{tp1_name}'])
    df_merged[f'Prob_{tp2_name}'] = np.exp(df_merged[f'Top1_Logprob_{tp2_name}'])

    # 2. 绘图
    plt.figure(figsize=(16, 7))

    # --- 左图: Logprob Scatter (对数空间) ---
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=f'Top1_Logprob_{tp1_name}',
        y=f'Top1_Logprob_{tp2_name}',
        data=df_merged,
        alpha=0.6,
        edgecolor=None,
        s=30,
        color='blue'
    )

    # 画一条 y=x 的红线作为基准
    min_val = min(df_merged[f'Top1_Logprob_{tp1_name}'].min(), df_merged[f'Top1_Logprob_{tp2_name}'].min())
    max_val = max(df_merged[f'Top1_Logprob_{tp1_name}'].max(), df_merged[f'Top1_Logprob_{tp2_name}'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect Match (y=x)')

    plt.title(f'Log Space: {tp1_name} vs {tp2_name} (Logprobs)', fontsize=14)
    plt.xlabel(f'{tp1_name} Logprob (Values < 0)')
    plt.ylabel(f'{tp2_name} Logprob (Values < 0)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 右图: Probability Scatter (线性空间) ---
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=f'Prob_{tp1_name}',
        y=f'Prob_{tp2_name}',
        data=df_merged,
        alpha=0.6,
        edgecolor=None,
        s=30,
        color='green'
    )

    # 画一条 y=x 的红线
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect Match (y=x)')

    plt.title(f'Linear Space: {tp1_name} vs {tp2_name} (Probabilities)', fontsize=14)
    plt.xlabel(f'{tp1_name} Probability (0.0 - 1.0)')
    plt.ylabel(f'{tp2_name} Probability (0.0 - 1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 简单的统计
    print(f"Logprob 相关系数: {df_merged[f'Top1_Logprob_{tp1_name}'].corr(df_merged[f'Top1_Logprob_{tp2_name}']):.8f}")
    print(f"Probability 相关系数: {df_merged[f'Prob_{tp1_name}'].corr(df_merged[f'Prob_{tp2_name}']):.8f}")
    print()

    # 设置一个极小值防止除零
    epsilon = 1e-9

    # === 1. 计算 MSE (均方误差) ===
    mse_logprob = ((df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']) ** 2).mean()
    mse_prob    = ((df_merged[f'Prob_{tp1_name}'] - df_merged[f'Prob_{tp2_name}']) ** 2).mean()

    # === 2. 计算 Relative Error (相对误差) ===
    # 公式: |TP1 - TP2| / (|TP1| + epsilon)
    # Logprob 空间
    rel_err_log = (df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']).abs() / (df_merged[f'Top1_Logprob_{tp1_name}'].abs() + epsilon)
    # Probability 空间
    rel_err_prob = (df_merged[f'Prob_{tp1_name}'] - df_merged[f'Prob_{tp2_name}']).abs

def calculate_error_metrics(df_merged, tp1_name='TP1', tp2_name='TP2'):
    """
    计算两个时间点(TP1, TP2)之间的误差指标

    Parameters:
    df_merged: 包含logits数据的DataFrame
    tp1_name: 第一个时间点的名称，默认为'TP1'
    tp2_name: 第二个时间点的名称，默认为'TP2'

    Returns:
    dict: 包含各种误差指标的字典
    """
    import numpy as np

    # 设置一个极小值防止除零
    epsilon = 1e-9

    # === 1. 计算 MSE (均方误差) ===
    mse_logprob = ((df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']) ** 2).mean()
    mse_prob    = ((df_merged[f'Prob_{tp1_name}'] - df_merged[f'Prob_{tp2_name}']) ** 2).mean()

    # === 2. 计算 Relative Error (相对误差) ===
    # 公式: |TP1 - TP2| / (|TP1| + epsilon)
    # Logprob 空间
    rel_err_log = (df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']).abs() / (df_merged[f'Top1_Logprob_{tp1_name}'].abs() + epsilon)
    # Probability 空间
    rel_err_prob = (df_merged[f'Prob_{tp1_name}'] - df_merged[f'Prob_{tp2_name}']).abs() / (df_merged[f'Prob_{tp1_name}'] + epsilon)

    # === 3. 格式化输出 ===
    print(f"====== 📉 误差统计分析 (Metrics) ======")
    print(f"MSE (Logprob空间):      {mse_logprob:.5e}")
    print(f"MSE (Probability空间):  {mse_prob:.5e}")
    print("-" * 40)
    # print(f"平均相对误差 (Logprob):     {rel_err_log.mean():.6%}  (Max: {rel_err_log.max():.4%})")
    print(f"平均相对误差 (Probability): {rel_err_prob.mean():.6%}  (Max: {rel_err_prob.max():.4%})")

    # 如果你想看相对误差最大的前3个样本
    print(f"\n====== ⚠️ 相对误差(Prob)最大的 Top 3 样本 ======")
    df_merged['Rel_Err_Prob_Val'] = rel_err_prob
    top_errors = df_merged.nlargest(3, 'Rel_Err_Prob_Val')
    for i, row in top_errors.iterrows():
        print(f"ID: {row['Question_ID']} | Token: {row[f'Top1_Text_{tp1_name}']} | {tp1_name}_Prob: {row[f'Prob_{tp1_name}']:.4f} | {tp2_name}_Prob: {row[f'Prob_{tp2_name}']:.4f} | Err: {row['Rel_Err_Prob_Val']:.2%}")

    # 返回误差指标字典
    return {
        'mse_logprob': mse_logprob,
        'mse_prob': mse_prob,
        'mean_rel_err_log': rel_err_log.mean(),
        'max_rel_err_log': rel_err_log.max(),
        'mean_rel_err_prob': rel_err_prob.mean(),
        'max_rel_err_prob': rel_err_prob.max(),
        'rel_err_log': rel_err_log,
        'rel_err_prob': rel_err_prob
    }

def analyze_divergence_tracking(df_merged, tp1_name='TP1', tp2_name='TP2'):
    """
    分析两个时间点(TP1, TP2)之间的路径偏离情况

    Parameters:
    df_merged: 包含logits数据的DataFrame
    tp1_name: 第一个时间点的名称，默认为'TP1'
    tp2_name: 第二个时间点的名称，默认为'TP2'

    Returns:
    dict: 包含分析结果的字典
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # ==========================================
    # 1. 核心逻辑：识别并标记"分水岭" (Divergence Tracking)
    # ==========================================

    # A. 计算单点 Token 是否不匹配
    df_merged['Token_Mismatch'] = df_merged[f'Top1_ID_{tp1_name}'] != df_merged[f'Top1_ID_{tp2_name}']

    # B. 计算分水岭状态 (Is_Diverged)
    # 逻辑：对于每个问题，一旦出现过 Mismatch，后续所有 Token 都标记为 Diverged
    df_merged['Is_Diverged'] = False

    for q_id in df_merged['Question_ID'].unique():
        # 获取当前问题的掩码
        q_mask = df_merged['Question_ID'] == q_id
        q_data = df_merged[q_mask].sort_values('Step_Index')

        # 找到第一个不匹配的索引
        mismatch_steps = q_data[q_data['Token_Mismatch']]['Step_Index']

        if not mismatch_steps.empty:
            first_mismatch_step = mismatch_steps.min()
            # 将该 step 及其之后的 token 全部标记为已偏离
            df_merged.loc[q_mask & (df_merged['Step_Index'] >= first_mismatch_step), 'Is_Diverged'] = True

    # 创建可读的状态标签用于绘图
    df_merged['Diverge_Status'] = df_merged['Is_Diverged'].map({False: 'Consistent (Pre-diverge)', True: 'Diverged (Post-diverge)'})

    # ==========================================
    # 2. 重新计算核心指标 (Metrics)
    # ==========================================

    # 计算概率空间值
    df_merged[f'Prob_{tp1_name}'] = np.exp(df_merged[f'Top1_Logprob_{tp1_name}'])
    df_merged[f'Prob_{tp2_name}'] = np.exp(df_merged[f'Top1_Logprob_{tp2_name}'])
    df_merged['Logits_Diff'] = (df_merged[f'Top1_Logprob_{tp1_name}'] - df_merged[f'Top1_Logprob_{tp2_name}']).abs()

    # 分组计算指标：我们主要关注"一致阶段"的微小误差
    stats_consistent = df_merged[~df_merged['Is_Diverged']]
    stats_diverged = df_merged[df_merged['Is_Diverged']]

    # ==========================================
    # 3. 可视化绘图 (Improved Scatter Plot)
    # ==========================================

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 8))

    # --- 右图 (按您的要求修改的 Prob 散点图) ---
    plt.subplot(1, 2, 1)

    # 使用不同颜色区分一致前和差别后
    # 绿色代表路径一致时的微小浮点误差，红色代表路径分叉后的巨大差异
    palette_colors = {"Consistent (Pre-diverge)": "#2ecc71", "Diverged (Post-diverge)": "#e74c3c"}

    sns.scatterplot(
        x=f'Prob_{tp1_name}',
        y=f'Prob_{tp2_name}',
        hue='Diverge_Status',
        data=df_merged,
        palette=palette_colors,
        alpha=0.6,
        edgecolor=None,
        s=40
    )

    # 画一条 y=x 的对角线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Match (y=x)')

    plt.title(f'Probability Distribution: {tp1_name} vs {tp2_name} (Pre vs Post Divergence)', fontsize=15)
    plt.xlabel(f'{tp1_name} Probability', fontsize=12)
    plt.ylabel(f'{tp2_name} Probability', fontsize=12)
    plt.legend(title="Generation Status")

    # --- 左图: Logits 随时间的变化 (显示误差积累) ---
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=df_merged.index,
        y='Logits_Diff',
        hue='Diverge_Status',
        data=df_merged,
        palette=palette_colors,
        s=40,
        alpha=0.7
    )
    plt.yscale('log') # 对数坐标更易观察 1e-6 级别的误差
    plt.axhline(y=1e-5, color='blue', linestyle=':', label='Common Float16 Noise Threshold')
    plt.title('Logits Absolute Difference (Temporal View)', fontsize=15)
    plt.ylabel('Abs Diff (Log Scale)', fontsize=12)
    plt.xlabel('Global Token Sequence', fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 4. 详细指标报告
    # ==========================================
    print(f"      ====== 📊 {tp1_name} vs {tp2_name} 深度差异报告 ======      ")
    print(f"总分析 Token 数: {len(df_merged)}")
    print(f"发生路径偏离的 Token 数: {len(stats_diverged)} (占 {len(stats_diverged)/len(df_merged):.2%})")
    print("-" * 50)

    print(f"【阶段 A: 路径一致时 (Consistent)】")
    if not stats_consistent.empty:
        print(f" -> 平均 Logits 误差: {stats_consistent['Logits_Diff'].mean():.2e}")
        print(f" -> 最大 Logits 误差: {stats_consistent['Logits_Diff'].max():.2e}")
        print(f" -> MSE (Probability): {((stats_consistent[f'Prob_{tp1_name}'] - stats_consistent[f'Prob_{tp2_name}'])**2).mean():.2e}")
    else:
        print(" -> (无数据)")

    print(f"\n【阶段 B: 路径偏离后 (Diverged)】")
    if not stats_diverged.empty:
        print(f" -> 平均 Logits 误差: {stats_diverged['Logits_Diff'].mean():.2e} (由于输入不同，误差天然变大)")
        print(f" -> 路径偏离的首个 ID 示例:")
        first_diffs = df_merged[df_merged['Token_Mismatch']].groupby('Question_ID').first()
        print(first_diffs[[f'Top1_Text_{tp1_name}', f'Top1_Text_{tp2_name}', 'Logits_Diff']].head())
    else:
        print(f" -> ✅ 恭喜：所有样本路径完全一致，未发生 Divergence。")

    print("-" * 50)

    # 返回分析结果字典
    return {
        'stats_consistent': stats_consistent,
        'stats_diverged': stats_diverged,
        'total_tokens': len(df_merged),
        'diverged_tokens': len(stats_diverged),
        'divergence_ratio': len(stats_diverged)/len(df_merged) if len(df_merged) > 0 else 0,
        'mean_logits_diff_consistent': stats_consistent['Logits_Diff'].mean() if not stats_consistent.empty else 0,
        'max_logits_diff_consistent': stats_consistent['Logits_Diff'].max() if not stats_consistent.empty else 0,
        'mse_prob_consistent': ((stats_consistent[f'Prob_{tp1_name}'] - stats_consistent[f'Prob_{tp2_name}'])**2).mean() if not stats_consistent.empty else 0
    }

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional

# ==========================================
# 函数 1: 提取 .json 文件为字典
# ==========================================
def load_debug_json(filepath: str) -> Dict[int, dict]:
    """
    读取 JSONL 格式的文件，并按 index 组织成字典。

    Args:
        filepath: JSON 文件路径 (例如 'layer_13_post_attn_tp1.json')
    Returns:
        Dict: { index: { 'shape': list, 'data': np.array } }
    """
    data_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            idx = entry['index']
            # 将列表转换为 numpy 数组以便后续计算，同时保持原始维度
            data_map[idx] = {
                'shape': entry['shape'],
                'data': np.array(entry['data'], dtype=np.float32)
            }
    print(f"[Loader] 成功加载 {filepath}, 包含 {len(data_map)} 个 Step 数据。")
    return data_map

# ==========================================
# 函数 2: 比较不同配置之间的相似度
# ==========================================
def compare_tp_configs(dict_ref: Dict[int, dict], dict_target: Dict[int, dict]) -> pd.DataFrame:
    """
    比较两个字典中相同 index 和 shape 的数据相似度。
    使用三种指标：Cosine Similarity, Pearson Correlation, MSE (均方误差)。

    Returns:
        pd.DataFrame: 包含 Index, Shape, 各项指标的 Mean/Max/Min
    """
    results = []

    # 获取交集索引并排序
    common_indices = sorted(set(dict_ref.keys()) & set(dict_target.keys()))

    for idx in common_indices:
        data1 = dict_ref[idx]['data']
        data2 = dict_target[idx]['data']
        shape1 = dict_ref[idx]['shape']
        shape2 = dict_target[idx]['shape']

        # 1. 检查 Shape 是否一致 (如果不一致，通常是 prefill/decode 阶段不对应，跳过)
        if shape1 != shape2:
            continue

        # 将数据展平为 [N, D] 形式，其中 D 是最后一个维度（Hidden Dim）
        # 如果是 [10, 2048]，则 N=10, D=2048
        v1 = data1.reshape(-1, data1.shape[-1])
        v2 = data2.reshape(-1, data2.shape[-1])

        cos_list, pearson_list, mse_list = [], [], []

        # 2. 逐向量计算 (以 2048 维度为例)
        for i in range(v1.shape[0]):
            vec1 = v1[i].reshape(1, -1)
            vec2 = v2[i].reshape(1, -1)

            # 指标 A: 余弦相似度
            cos = cosine_similarity(vec1, vec2)[0][0]
            cos_list.append(cos)

            # 指标 B: Pearson 相关系数
            # pearsonr 返回 (correlation, p-value)，由于是浮点对比，重点在 correlation
            corr, _ = pearsonr(v1[i], v2[i])
            pearson_list.append(corr)

            # 指标 C: MSE (均方误差) - 反应数值绝对偏差
            mse = np.mean((v1[i] - v2[i])**2)
            mse_list.append(mse)

        # 3. 汇总当前 Step 的指标
        results.append({
            'index': idx,
            'shape': str(shape1),
            'cos_mean': np.mean(cos_list),
            'cos_max': np.max(cos_list),
            'cos_min': np.min(cos_list),
            'pearson_mean': np.mean(pearson_list),
            'pearson_max': np.max(pearson_list),
            'pearson_min': np.min(pearson_list),
            'mse_mean': np.mean(mse_list),
            'mse_max': np.max(mse_list)
        })

    df = pd.DataFrame(results)
    return df

# ==========================================
# 函数 3: 绘制相似度变化曲线
# ==========================================
def plot_similarity_report(df: pd.DataFrame, metric_type: str = 'cos', title: str = "TP Consistency Analysis", plot_min_max: bool = True):
    """
    绘制指定指标随 Index 变化的趋势图。

    Args:
        df: compare_tp_configs 生成的 DataFrame
        metric_type: 'cos', 'pearson', 或 'mse'
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    # 映射列名
    prefix = metric_type.lower()
    col_mean = f"{prefix}_mean"
    col_max = f"{prefix}_max"
    col_min = f"{prefix}_min"

    if col_mean not in df.columns:
        print(f"错误: 找不到指标 {metric_type}")
        return

    # 绘制均值曲线
    plt.plot(df['index'], df[col_mean], color='#1f77b4', label=f'Mean {metric_type.upper()}', linewidth=2, marker='o', markersize=4)

    # 绘制最大/最小值填充区间 (Shaded Area)
    if col_min in df.columns and plot_min_max:
        plt.fill_between(df['index'], df[col_min], df[col_max], color='#1f77b4', alpha=0.2, label='Max-Min Range')

    # 添加拟合曲线 (Polynomial Fit)
    z = np.polyfit(df['index'], df[col_mean], 3)
    p = np.poly1d(z)
    plt.plot(df['index'], p(df['index']), "r--", alpha=0.8, label='Trend')

    # 美化图表
    plt.title(f'{title}: {metric_type.upper()} over Steps', fontsize=16)
    plt.xlabel('Token Generation Step (Index)', fontsize=12)
    plt.ylabel(f'Similarity Metric ({metric_type.upper()})', fontsize=12)

    # 如果是 Cosine 或 Pearson，固定 Y 轴范围在 [0, 1.05] 方便观察
    if metric_type in ['cos', 'pearson'] and plot_min_max:
        current_min = df[col_min].min()
        plt.ylim(max(0, current_min - 0.05), 1.05)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


