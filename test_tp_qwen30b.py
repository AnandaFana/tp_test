# python test_tp_qwen30b.py --tp 2 --output result_tp2_run2.json

import os
import argparse
import json
import time
from vllm import LLM, SamplingParams

# === 1. 定义多领域测试集 (10个问题) ===
TEST_PROMPTS = [
    # 基础常识
    "法国的首都是哪里？",
    # 数学计算
    "25 * 25 等于多少？请直接输出数字。",
    # 逻辑推理
    "如果所有的猫都喜欢吃鱼，而汤姆是一只猫，那么汤姆喜欢吃什么？",
    # 代码能力
    "请用Python写一个Hello World函数。",
    # 文学创作
    "请模仿李白的风格写两句关于月亮的诗。",
    # 天文地理
    "太阳系中体积最大的行星是哪颗？",
    # 翻译
    "Translate 'Artificial Intelligence is the future' into Chinese.",
    # 情感分析
    "这句话的情感是正面还是负面：'今天丢了钱包，心情糟透了。'",
    # 历史
    "第二次世界大战是在哪一年结束的？",
    # 物理常识
    "为什么苹果会往地上掉而不是往天上飞？简要解释。"
]

def run_benchmark(tp_size, output_file, model_path):
    print(f"=== 开始测试: TP_SIZE={tp_size} ===")

    # === 2. 强制确定性设置 ===
    # 必须设置 seed，并且使用 temperature=0 进行贪婪解码
    sampling_params = SamplingParams(
        temperature=0.0,       # 贪婪解码，消除采样随机性
        max_tokens=100,        # 限制输出长度
        logprobs=1,            # 输出 Logits (用于数值对比)
        seed=42                # 固定随机种子
    )

    # === 3. 加载模型 ===
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        seed=42
    )

    # === 4. 执行推理 ===
    start_time = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    end_time = time.time()

    # === 5. 格式化结果 ===
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        # 获取第一个生成token的logprob作为数值参考
        # 注意：这里我们只取第一个token的logprob做简单对比，取全部太大了
        first_token_logprob = output.outputs[0].logprobs[0]
        # first_token_logprob 是一个字典 {token_id: logprob_object}
        token_id = list(first_token_logprob.keys())[0]
        score = first_token_logprob[token_id].logprob

        results.append({
            "id": i,
            "prompt": output.prompt,
            "text": generated_text,
            "first_token_logprob": score  # 关键数值指标
        })

    # === 6. 保存到文件 ===
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"=== 测试完成，耗时 {end_time - start_time:.2f}s ===")
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--output", type=str, required=True, help="Output json filename")
    parser.add_argument("--model", type=str, default="/public/data_science/models/downloaded_models/Qwen3-30B-A3B-Instruct-2507", help="Model path")

    args = parser.parse_args()
    run_benchmark(args.tp, args.output, args.model)