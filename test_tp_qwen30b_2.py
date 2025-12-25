# python test_tp_qwen30b_2.py --tp 1  --output result_detailed_tp1.xlsx
# python test_tp_qwen30b_2.py --tp 2  --output result_detailed_tp2.xlsx
#  用于检测不同TP size得到的测试集合的问题输出不同；

import argparse
import pandas as pd
import time
from vllm import LLM, SamplingParams

# === 1. 保持一致的测试集 ===
TEST_PROMPTS = [
    "法国的首都是哪里？请介绍一下这个城市的历史和未来战略",
    "请分析一下中国的人口发展趋势，历史原因以及可能的问题和解决办法",
    "如果所有的猫都喜欢吃猫草，而汤姆是一只猫，那么汤姆喜欢吃什么？分析一下吃素的优缺点",
    "请用Python写一个Hello World函数，并介绍一下python的发展历史",
    "请模仿李白的风格写两句关于月亮和家乡的诗，并作出分析",
    "太阳系中体积最大的行星是哪颗？ 求介绍一下这颗行星",
    "Translate 'Artificial Intelligence is the future' into Chinese and introduce some knowledge of it in chinese.",
    "这句话的情感是正面还是负面：'今天丢了钱包，心情不太好。'请提供详细的分析",
    "第二次世界大战是在哪一年结束的？请详细介绍一下历史原因和人们悼念的活动",
    "为什么苹果会往地上掉而不是往天上飞？请详细解释并提供应用例子。"
]

def run_detailed_benchmark(tp_size, output_file, model_path):
    print(f"=== 开始详细测试 (Top-5 Logits): TP_SIZE={tp_size} ===")

    # === 2. 这里的 logprobs 设为 5 ===
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,          # 控制导出多少参数
        logprobs=5,            # <--- 关键修改：获取前5个
        seed=42
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        seed=42,
        enforce_eager=True,
        max_num_batched_tokens=8192,   # 允许一次 prefill 最多 4096 tokens
        max_num_seqs=20,
    )


    _ = llm.generate("Hello", SamplingParams(max_tokens=1))  # 预热

    start_time = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    end_time = time.time()

    # === 3. 数据处理：转为“长格式” ===
    # 目标：一行一个 Token，记录 Top 5 信息
    detailed_data = []

    for q_id, output in enumerate(outputs):
        prompt_text = output.prompt
        # output.outputs[0].logprobs 是一个列表，列表长度 = 生成的 token 数量
        # 列表中的每个元素是一个字典：{token_id: LogprobObject, ...}
        generated_logprobs_list = output.outputs[0].logprobs
        generated_token_ids = output.outputs[0].token_ids

        for step, (step_logprobs, actual_token_id) in enumerate(zip(generated_logprobs_list, generated_token_ids)):
            # step_logprobs 是类似 {id1: logprobObj, id2: logprobObj...} 的字典
            # 我们需要按 logprob 从大到小排序，提取前 5 个
            # 注意：vLLM返回的字典通常已经是Top K，但为了保险我们手动排一下

            # 提取 (token_id, logprob) 对
            top_candidates = []
            for tid, logprob_obj in step_logprobs.items():
                top_candidates.append((tid, logprob_obj.logprob, logprob_obj.decoded_token))

            # 按分数降序排列
            top_candidates.sort(key=lambda x: x[1], reverse=True)

            # 构建这一行的数据
            row = {
                "Question_ID": q_id,
                "Prompt_Snippet": prompt_text[:10], # 只存前10个字方便识别
                "Step_Index": step,                 # 第几个 Token
                "Actual_Token_ID": actual_token_id  # 实际选中的那个
            }

            # 填入 Top 1 到 Top 5 的数据
            for rank in range(5):
                if rank < len(top_candidates):
                    tid, score, token_text = top_candidates[rank]
                    row[f"Top{rank+1}_ID"] = tid
                    row[f"Top{rank+1}_Logprob"] = score
                    # 也可以存 token 文本，方便肉眼看，但在excel里可能有特殊字符乱码风险，这里先选存
                    row[f"Top{rank+1}_Text"] = token_text
                else:
                    # 万一不足5个
                    row[f"Top{rank+1}_ID"] = None
                    row[f"Top{rank+1}_Logprob"] = None
                    row[f"Top{rank+1}_Text"] = None

            detailed_data.append(row)

    # === 4. 保存为 Excel ===
    df = pd.DataFrame(detailed_data)

    # 调整一下列的顺序，把 Logprob 放一起方便看
    cols = ['Question_ID', 'Prompt_Snippet', 'Step_Index', 'Actual_Token_ID']
    for i in range(1, 6):
        cols.extend([f'Top{i}_ID', f'Top{i}_Logprob', f'Top{i}_Text'])

    df = df[cols]

    df.to_excel(output_file, index=False)

    print(f"=== 测试完成，耗时 {end_time - start_time:.2f}s ===")
    print(f"详细数据 (Rows={len(df)}) 已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--output", type=str, required=True, help="Output xlsx filename")
    # 请确保这里路径是你服务器上的真实路径
    parser.add_argument("--model", type=str, default="/public/data_science/models/downloaded_models/Qwen3-30B-A3B-Instruct-2507", help="Model path")

    args = parser.parse_args()
    run_detailed_benchmark(args.tp, args.output, args.model)