# test_all_gather.py
#  torchrun --nproc_per_node=2 --master_port=0 test_all_gather.py


import torch
import torch.distributed as dist
import os

def init_process(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def test_all_gather():
    """测试All-Gather操作"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个rank创建不同的数据
    tensor_size = 1024 * 1024  # 1MB数据
    local_tensor = torch.ones(tensor_size, device=f'cuda:{rank}') * (rank + 1)

    # 准备接收缓冲区
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]

    # 执行All-Gather
    dist.all_gather(gathered_tensors, local_tensor)

    # 验证结果
    for i, tensor in enumerate(gathered_tensors):
        expected_value = i + 1
        if not torch.all(tensor == expected_value):
            print(f"Rank {rank}: Error in tensor from rank {i}")
            return False

    print(f"Rank {rank}: All-Gather test passed!")
    return True

if __name__ == "__main__":
    # 通过torchrun启动
    # 命令: torchrun --nproc_per_node=2 test_all_gather.py
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    init_process(rank, world_size)
    test_all_gather()
    dist.destroy_process_group()