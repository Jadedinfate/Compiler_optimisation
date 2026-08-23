import torch
import torch_geometric

print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))