
import torch
import numpy as np

np_array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(np_array).type(torch.float32)

print(f'numpy array type: {np_array.dtype}')
print(f'tensor type: {tensor.dtype}')

# change the value of numpy array doesn't change the tensor array
np_array = np_array + 1
print(np_array, tensor)

# tensor to numpy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor, numpy_tensor.dtype)

# change the tensor -> numpy does not change
tensor = tensor + 1
print(tensor, numpy_tensor)

"""
 Reproducbility (trying to take random out of random)
 how neural networks learn
 
 starts with random number -> tensor operations -> update random number
 to try and make them better representations of the data -> and again....
 
 to reduce the randomness in neural networks and pytorch comes
 the concept of a random seed
"""

random_torch = torch.rand(3, 3)

