
import torch


"""
    Reproducibility
    use manual_seed()    
"""

random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

# print(random_tensor_A, '\n', random_tensor_B)
# print(random_tensor_A == random_tensor_B)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
D = torch.rand(3, 4)
print(C, '\n', D)
print(C == D)