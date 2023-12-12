
import torch

X = torch.rand(size=(20, 5), dtype=torch.float16)
Y = torch.rand(size=(240, 85), dtype=torch.float16)

results = torch.matmul(X, Y)
print(torch.shape)