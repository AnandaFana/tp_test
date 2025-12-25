import torch
import torch.distributed as dist
import os
import json
from datetime import datetime

class DebugDumper:
    """
    用于在 vLLM 推理过程中 Dump 中间层 Tensor 的辅助工具类。
    支持追加存储多个张量数据到 JSON 文件中，并可跳过 prefill 阶段的大张量
    """

    # 配置 Dump 的根目录
    DUMP_ROOT = "/public/data_science/saultysoup/tp_test/vllm_debug_dump"
    # 是否开启 Dump (可以通过修改此变量快速开关)
    ENABLED = True
    # 跳过 prefill 阶段的阈值，如果第一个维度大于此值则跳过
    PREFILL_SKIP_THRESHOLD = 10

    _initialized = False
    _tp_size = 1

    @classmethod
    def init_dump_dir(cls):
        if not cls.ENABLED:
            return
        if not cls._initialized:
            # 获取 TP size
            try:
                cls._tp_size = dist.get_world_size() if dist.is_initialized() else 1
            except:
                cls._tp_size = 1

            # 只在 Rank 0 创建目录，避免冲突
            if cls.is_rank_0():
                os.makedirs(cls.DUMP_ROOT, exist_ok=True)
                print(f"[DebugDumper] Output directory: {cls.DUMP_ROOT}, TP Size: {cls._tp_size}")

            cls._initialized = True

    @staticmethod
    def is_rank_0():
        """检查当前是否为主进程 (Rank 0)"""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    @classmethod
    def _get_json_filepath(cls, layer_idx: int, tag: str) -> str:
        """获取对应的 JSON 文件路径"""
        filename = f"layer_{layer_idx:02d}_{tag}_tp{cls._tp_size}.json"  #change .json" #
        return os.path.join(cls.DUMP_ROOT, filename)

    @classmethod
    def _get_next_index(cls, filepath: str) -> int:
        """快速获取下一个索引，只需读取最后一行"""
        if not os.path.exists(filepath):
            return 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        last_data = json.loads(last_line)
                        return last_data.get('index', 0) + 1
        except:
            pass
        return 0

    @classmethod
    def save(cls, tensor: torch.Tensor, layer_idx: int, tag: str, skip_prefill: bool = True):
        return  # changed @@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        保存 Tensor 到 JSON 文件，支持追加模式和跳过 prefill 阶段。
        Args:
            tensor: 需要保存的 Tensor
            layer_idx: 当前层号
            tag: 标签，如 'post_attn', 'post_mlp'
            skip_prefill: 是否跳过 prefill 阶段的大张量（第一个维度 > threshold）
        """
        if layer_idx % 8 !=0:
            return

        if not cls.ENABLED:
            return

        # 仅在 Rank 0 保存
        if not cls.is_rank_0():
            return

        # 检查是否需要跳过 prefill 阶段
        if skip_prefill and len(tensor.shape) > 0 and tensor.shape[0] > cls.PREFILL_SKIP_THRESHOLD:
            print(f"[DebugDumper] Skipping prefill tensor with shape {tensor.shape}")
            return

        # 确保初始化
        cls.init_dump_dir()

        # 获取对应的 JSON 文件路径
        filepath =   cls._get_json_filepath(layer_idx, tag)

        try:
            # 获取下一个索引
            next_index = cls._get_next_index(filepath)

            # 准备要保存的数据
            tensor_data = {
                'index': next_index,
                'timestamp': datetime.now().isoformat(),
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'data':  tensor.detach().clone().cpu().tolist()  # 转换为 Python 列表  #change
            }

            # 使用追加模式写入单行 JSON
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(tensor_data) + '\n')

            print(f"[DebugDumper] Saved tensor #{next_index} with shape {tensor.shape} to {os.path.basename(filepath)}")

        except Exception as e:
            print(f"[DebugDumper] Error saving tensor to {filepath}: {e}")

    @classmethod
    def save_with_limit(cls, tensor: torch.Tensor, layer_idx: int, tag: str, max_entries: int = 100, skip_prefill: bool = True):  # change to False for skip
        """
        保存 Tensor 到 JSON 文件，带最大条目限制和 prefill 跳过功能。
        Args:
            tensor: 需要保存的 Tensor
            layer_idx: 当前层号
            tag: 标签
            max_entries: 最大存储条目数，超过时会删除最早的条目
            skip_prefill: 是否跳过 prefill 阶段的大张量
        """
        if not cls.ENABLED:
            return

        if not cls.is_rank_0():
            return

        # 检查是否需要跳过 prefill 阶段
        if skip_prefill and len(tensor.shape) > 0 and tensor.shape[0] > cls.PREFILL_SKIP_THRESHOLD:
            print(f"[DebugDumper] Skipping prefill tensor with shape {tensor.shape}")
            return

        cls.init_dump_dir()
        filepath = cls._get_json_filepath(layer_idx, tag)

        try:
            # 获取下一个索引
            next_index = cls._get_next_index(filepath)

            tensor_data = {
                'index': next_index,
                'timestamp': datetime.now().isoformat(),
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'data': tensor.detach().clone().cpu().tolist()
            }

            # 先追加新数据
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(tensor_data) + '\n')

            # 如果需要限制条目数，读取并清理文件
            if max_entries > 0:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if len(lines) > max_entries:
                    # 保留最新的 max_entries 条记录
                    lines = lines[-max_entries:]

                    # 重写文件
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

            print(f"[DebugDumper] Saved tensor #{next_index} with shape {tensor.shape} to {os.path.basename(filepath)}")

        except Exception as e:
            print(f"[DebugDumper] Error saving tensor to {filepath}: {e}")

    @classmethod
    def read_data(cls, layer_idx: int, tag: str) -> list:
        """
        读取指定层和标签的 JSON 数据
        """
        filepath = cls._get_json_filepath(layer_idx, tag)
        if os.path.exists(filepath):
            try:
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data
            except json.JSONDecodeError:
                print(f"[DebugDumper] Error reading {filepath}")
                return []
        return []



# import torch
# import torch.distributed as dist
# import os
# from datetime import datetime

# class DebugDumper:
#     """
#     用于在 vLLM 推理过程中 Dump 中间层 Tensor 的辅助工具类。
#     """

#     # 配置 Dump 的根目录
#     DUMP_ROOT = "/public/data_science/saultysoup/tp_test/vllm_debug_dump"
#     # 是否开启 Dump (可以通过修改此变量快速开关)
#     ENABLED = True

#     _initialized = False
#     _tp_size = 1

#     @classmethod
#     def init_dump_dir(cls):
#         if not cls.ENABLED:
#             return
#         if not cls._initialized:
#             # 获取 TP size
#             try:
#                 cls._tp_size = dist.get_world_size() if dist.is_initialized() else 1
#             except:
#                 cls._tp_size = 1

#             # 只在 Rank 0 创建目录，避免冲突
#             if cls.is_rank_0():
#                 os.makedirs(cls.DUMP_ROOT, exist_ok=True)
#                 print(f"[DebugDumper] Output directory: {cls.DUMP_ROOT}, TP Size: {cls._tp_size}")

#             cls._initialized = True

#     @staticmethod
#     def is_rank_0():
#         """检查当前是否为主进程 (Rank 0)"""
#         if not dist.is_initialized():
#             return True
#         return dist.get_rank() == 0

#     @classmethod
#     def save(cls, tensor: torch.Tensor, layer_idx: int, tag: str):
#         """
#         保存 Tensor 到文件。
#         Args:
#             tensor: 需要保存的 Tensor
#             layer_idx: 当前层号
#             tag: 标签，如 'post_attn', 'post_mlp'
#         """
#         if not cls.ENABLED:
#             return

#         # 1. 仅在 Rank 0 保存 (假设 TP 模式下我们只关心同步后的结果，或者你也可以去掉这个限制来对比不同卡的数据)
#         if not cls.is_rank_0():
#             return

#         # 确保初始化
#         cls.init_dump_dir()

#         # 2. 构造文件名: layer_{idx}_{tag}_tp{size}.pt
#         filename = f"layer_{layer_idx:02d}_{tag}_tp{cls._tp_size}.pt"
#         filepath = os.path.join(cls.DUMP_ROOT, filename)

#         # 3. 核心：Clone -> Detach -> CPU -> Save
#         # 必须 clone 和 detach，否则可能保存到后续被 inplace 修改的值
#         # 必须转 CPU，否则会阻塞 GPU 流水线或导致 OOM
#         try:
#             data_to_save = tensor.detach().clone().cpu()
#             torch.save(data_to_save, filepath)
#             # print(f"[Debug] Saved {filename}") # 打印太多会刷屏，建议注释掉
#         except Exception as e:
#             print(f"[DebugDumper] Error saving {filename}: {e}")
