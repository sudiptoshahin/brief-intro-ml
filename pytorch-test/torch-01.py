

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cv2

print(torch.__version__)


# #### Tensor
# 
# Naming convention:
# * Scalers and verctor in lowercase
# * and MATRIX and TENSORS should be in uppercase


# scaler tensor
scaler = torch.tensor(7)
print(scaler.ndim)
print(scaler.item())


# vector
vector = torch.tensor([3, 2])
print(vector.ndim)
print(vector.shape)

# Matrix
MATRIX = torch.tensor([
    [1, 2],
    [3, 4]
])

print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX[0:, 0:1])

# Tensor
TENSOR = torch.tensor([[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]])

# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR.shape) 

# TENSOR[0:, 0:, 0:1]

TENSOR2 = torch.tensor([
    [
        [1, 2, 2],
        [2, 3, 4],
        [4, 5, 6]
    ],
    [
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ],
    [
        [7, 8, 9],
        [1, 2, 3],
        [10, 11, 12]
    ],
    [
        [3, 2, 1],
        [7, 6, 8],
        [0, 1, 1]
    ]
])

print(TENSOR2.shape)

# numpy array to tensor

np_arr = np.array([
    [
        [1, 2, 2],
        [2, 3, 4],
        [4, 5, 6]
    ],
    [
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ],
    [
        [7, 8, 9],
        [1, 2, 3],
        [10, 11, 12]
    ],
    [
        [3, 2, 1],
        [7, 6, 8],
        [0, 1, 1]
    ]
])

tnsr = torch.as_tensor(np_arr)

print(type(np_arr))
print(type(tnsr))

torch.zeros([3, 2])
cuda0 = torch.device('cuda:0')
cuda_tens = torch.ones([2, 4], dtype=torch.float64, device=cuda0)

x = torch.tensor([
    [1, -1],
    [1, 1]
], dtype=torch.float16, requires_grad=True)
out = x.pow(2).sum()
out.backward()
x = x.grad
# transpose
x.T

# stored on CPU/GPU
print(cuda_tens.is_cuda)
print(x.is_cuda)

print('device:cuda:- ', cuda_tens.device)
print('device:cpu:-', x.device)


# Random tensors
# Random tensors are important because many neural networks lears is that they starts with tensors full of random numbers and then adjust those random numbers to better represent the data
# 
# `starts with random numbers => look at data => update random numbers => look at data => update random numbers`

random_tensor = torch.rand(3, 2)

# create a random tensor with 
# similar shape to an image tensor
# random_image_tensor = torch.rand(size=(224, 224, 3))
random_image_tensor = torch.rand(size=(3, 224, 224))

# random_image_tensor.shape, random_image_tensor.ndim
# random_image_tensor

# zeros and ones
zeros = torch.zeros(size=(3, 2))

ones = torch.ones(size=(2, 3))
ones.dtype

# creating a range of tensors and tensors-like

one_to_ten = torch.arange(start=0, end=100, step=10)

# tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros


# #### Tensors dtypes
# 
# **Note:** Tensor data type is one of 3 big errors we will run into pytorch and deep learning:
# 1. Tensors not right datatype
# 2. tensors not right shape
# 3. tensors not on the right device

float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=None, # datatype 
    device=None, # what device is your tensor on
    requires_grad=False # weather or not to track gradients with this tensor operations
)
print(float_32_tensor.dtype)
# convert datatype
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor.dtype)


# #### Getting information from tensors
# ### Attributes

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)

float_32_tensor * int_32_tensor


# 1. Tensors not right datatype -> get datatype => `tensor.dtype`
# 2. tensors not right shape -> get shape => `tensor.shape`
# 3. tensors not on the right device -> get device => `tensor.device`

some_tensor = torch.rand(size=(3, 4))
print(some_tensor)

print(f'Datatype: {some_tensor.dtype}')
print(f'Shape: {some_tensor.shape}')
print(f'Device: {some_tensor.device}')


# #### Manipulating / Operations
# * Addition
# * Subtraction
# * Multiplication ( element-wise )
# * Division
# * Matrix multiplication
#

# tnsr = torch.tensor([1, 2, 3])

tnsr = torch.rand(100)
add_tnsr = tnsr + 10

mul_tnsr = tnsr * 10

sub_tnsr = tnsr - 10

# built-in function
torch.mul(tnsr, 10)


# #### Matrix multiplication
# 
# 1. element-wise multiplication
# 2. matrix multiplication (dot product)


print(tnsr ,'*', tnsr)
print(f'equals: {tnsr*tnsr}')

# matrix multiplication
torch.matmul(tnsr, tnsr)

import time
st = time.time()
val = 0
for i in range(len(tnsr)):
    val += tnsr[i] * tnsr[i]

print(val)
print(f'{round(((time.time() - st) * 10**3), 500)}\'ms')

st = time.time()

mat_mul = torch.matmul(tnsr, tnsr)
print(mat_mul)

print(f'{round(((time.time() - st) * 10**3), 500)}\'ms')

import time

st = time.time()

mat_mul = torch.matmul(tnsr, tnsr)
print(mat_mul)

print(f'{round(((time.time() - st) * 10**3), 120)}\'ms')

A = torch.rand(size=(30, 5)) # A's column
B = torch.rand(size=(5, 32)) # b's row must be same

matrix_mul = torch.matmul(A, B)
print(matrix_mul.shape)

A = torch.rand(size=(7, 80))
B = torch.rand(size=(80, 3))

# print(f' {A.shape} \n\t\tx\n {B.shape} \n-----------------------------------\n {torch.matmul(A, B).shape}')

# check shape
def check_and_print(A, B):
    if len(A.shape) != 2 or len(B.shape) != 2:
        print('Please enter matrix....')
        exit(1)
    else:
        # if A.shape[1] != B.shape[0]:
        #     print('Please enter a VALID matrix....1')
        #     quit()
        # mat_mul = torch.matmul(A, B)
        try:
            mat_mul = torch.matmul(A, B)
            print(f' {A.shape} \n\t\tx\n {B.shape} \n-----------------------------------\n result will be in\n {mat_mul.shape} shape')
        except RuntimeError:
            # quit()
            print('Please enter a VALID matrix....2')

check_and_print(A, B)


A = torch.rand(size=(50, 2), dtype=torch.float32)
B = torch.rand(size=(20, 2), dtype=torch.float32)

# torch.matmul(A, B).shape
# matrix multiplication will work when tensor_B is TRANSPOSED
torch.matmul(A, B.T).shape


