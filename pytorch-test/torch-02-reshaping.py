import torch
import cv2
import numpy as np

# RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'

# torch.set_default_tensor_type(torch.cuda.HalfTensor)
# we can use torch.float32 for use the above code

X = torch.rand(size=(20, 5), dtype=torch.float32)
Y = torch.rand(size=(5, 240), dtype=torch.float32)

# results = torch.matmul(X, Y)
# print(results.shape)


"""
find the min max, mean, sum (tensor aggregation)
"""
x = torch.arange(0, 100, 10, dtype=torch.float32)
torch.min(x)

# print(x.max())
# print(x.mean())
#
# print(torch.sum(x))

# method has return the position of min and max position

# print(torch.argmin(x, dim=(si)))
# print(x.argmin())
#
# print(x.argmax())

"""
  # reshaping, stacking, squeezing and unsqueezing
  
  * reshaping - reshapes an input tensor a defined shape
    reshaping must be match with length of original tensors
  * view - return a view of an input tensor of certain shape
            but keep the same memory as the original tensor
  * Stacking - combine multiple tensors on top of each other (vstack)
                or side bu side (hstack)
  * squeeze - remove all 1 dimensions from a tensor
  * unsqueeze - add a 1 dimension to a target tensor 
  * permute - return a view of the input with dimension permuted(swapped) in a certain way
"""

X = torch.arange(1., 13.)
# print(X, X.shape)

X_reshaped = X.reshape(3, 2, 2)
# print(X_reshaped, '\n', X_reshaped.shape)

# view shares the same memory of the tensors
# pointer concept for programming
X = torch.arange(1., 10.)
z = X.view(1, 9)

# if we changes an element of z it will changes in X
z[:, 0] = 5.
# print(z, X)

# Stack tensor of top of each other
X_stacked = torch.stack([X, X, X], dim=1)
# print(X_stacked)

# torch.squeeze() - removes all single dimensions from the target tensor
x = torch.arange(1., 10.)
x_reshaped = x.reshape(1, 9)

# print(f'previous tensor: {x_reshaped}')
# print(f'\nprevious tensor shape: {x_reshaped.shape}')

x_squeezed = x_reshaped.squeeze()
# print(f'\nnew tensor: {x_squeezed}')
# print(f'\nnew tensor shape: {x_squeezed.shape}')

## torch.unsqueeze()
# adds a single dimension to a target tensor at a specific dimension
# print(f'previous tensor: {x_squeezed}')
# print(f'\nprevious tensor shape: {x_squeezed.shape}')

x_unsqueezed = x_squeezed.unsqueeze(dim=1)
# print(f'\nnew tensor: {x_unsqueezed}')
# print(f'\nnew tensor shape: {x_unsqueezed.shape}')


## torch.permute - rearranges the dimensions of a target tensor in specified order
x_original = torch.rand(size=(224, 224, 3)) # height, width, color_channels

# shift axis 0 ->  1, 1 -> 2, 2 -> 0
x_permuted = x_original.permute(2, 0, 1) # color_channels, height, width
print(x_permuted.shape)
