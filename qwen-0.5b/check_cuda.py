import torch
import sys

def check_pytorch_cuda():
    """
    检查 PyTorch 和 CUDA 的兼容性并打印相关环境信息。
    """
    print(f"--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print("-" * 30)

    # 1. 核心检查：CUDA 是否对 PyTorch 可用
    # 这是最关键的一步，如果返回 True，说明 PyTorch 能够成功调用 GPU。
    is_available = torch.cuda.is_available()
    
    print(f"Is CUDA available for PyTorch? -> {is_available}")

    if not is_available:
        print("\n[Warning] PyTorch cannot find a compatible CUDA setup.")
        print("This could be due to several reasons:")
        print("  1. Your NVIDIA drivers are not installed or need an update.")
        print("  2. You installed a CPU-only version of PyTorch.")
        print("     (To fix, reinstall PyTorch with a specific CUDA version, e.g., 'pip install torch --index-url https://download.pytorch.org/whl/cu121')")
        print("  3. The installed CUDA Toolkit version is not compatible with your PyTorch version or NVIDIA driver.")
        print("-" * 30)
        return

    print("-" * 30)
    # 2. 获取 PyTorch 编译时所依赖的 CUDA 版本
    print(f"PyTorch was compiled with CUDA version: {torch.version.cuda}")

    # 3. 获取检测到的 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    if gpu_count > 0:
        print("-" * 30)
        # 4. 遍历并打印每个 GPU 的信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"--- GPU {i} ---")
            print(f"  Name: {gpu_name}")
            print(f"  Total Memory: {total_mem:.2f} GB")
    
    print("\n[Success] Your PyTorch and CUDA setup appears to be correct!")
    print("-" * 30)

if __name__ == "__main__":
    check_pytorch_cuda()
