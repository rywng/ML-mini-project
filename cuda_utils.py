import torch


def get_least_used_gpu() -> str:
    """Returns the index of the least used CUDA GPU."""
    assert torch.cuda.is_available()

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return "cpu"  # Cuda support is not available
    else:
        memory_free = []
        for i in range(device_count):
            mem_used = torch.cuda.mem_get_info(i)[0]
            memory_free.append(mem_used)
        max_free = max(memory_free)
        gpu = memory_free.index(max_free)
        load_ratio = 1 - (max_free / torch.cuda.mem_get_info(gpu)[1])
        if load_ratio > 0.7:
            print("All GPUs are under heavy load.")
            print(
                f"the least used GPU is {gpu} with {max_free / 1024 / 1024} MB available, and {load_ratio}% load"
            )
            ans = input("Continue? (y/N) ")
            if not ans.lower().startswith("y"):
                print("Exiting")
                exit(0)
        return "cuda:" + str(gpu)


if __name__ == "__main__":
    least_used_gpu = get_least_used_gpu()
    if least_used_gpu is None:
        print("No CUDA GPUs found.")
    else:
        print(least_used_gpu)
