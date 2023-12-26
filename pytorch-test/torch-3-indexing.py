import torch

"""
    Indexing (selecting images from tensors)
    similar indexing as numpy
"""

x = torch.arange(1, 10).reshape(1, 3, 3)
print(f'original tensor: {x}')
print(f'original tensor shape: {x.shape}')

# [
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ]
# ]

# index on the middle brackets
print(x[0][0])

# index on the most inner bracket (last dimension)
print(x[0][1][1])

# select all of the target dimension
print(x[:, 1])

# get all values of 0th and 1st dimension but only index 1 of 2nd dimension
print(x[:, :, 1])















