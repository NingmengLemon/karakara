import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Print the number of available GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # Print the name of the current (or first) GPU
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    # Print the CUDA compute capability of the first GPU
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")
