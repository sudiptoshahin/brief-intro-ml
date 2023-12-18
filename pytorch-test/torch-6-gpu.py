import torch

"""
    GPU access
    
"""

# check
# print(f'cuda available: {torch.cuda.is_available()}')
# print(f'cuda devicecount: {torch.cuda.device_count()}')

CPU = torch.device('cpu')
CUDA = torch.device('cuda')

tensor = torch.rand([1, 2, 3, 4], device=CPU)
