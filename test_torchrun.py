# test_torchrun.py
import os

def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"Rank {rank}, Local Rank {local_rank}, World Size {world_size} --> YYYYYYYYYY")

if __name__ == "__main__":
    main()