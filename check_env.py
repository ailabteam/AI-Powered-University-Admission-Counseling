import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version used by PyTorch: {torch.version.cuda}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
