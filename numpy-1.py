#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

print(np.__version__)


# In[10]:


arr = np.array(42)
np.array([[1, 2, 3], [3, 4, 5]])
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)


# ## np dimensions

# In[30]:


a = np.array(43)
b = np.array([1, 2, 3, 4])
c = np.array([[1, 2], [4, 5]])
d = np.array([[[1, 2, 3], [4, 5, 7]], [[9, 10, 11], [12, 13, 14]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

# create a array with 5 dimensions
arr = np.array([1, 2, 3, 4], ndmin=5)

arr1 = np.array([[1, 2], [3, 4], [5,6]])
print(arr1[:2])


# ## 3D array

# In[40]:


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr)
# print(arr.ndim)
print('\n')
print(arr[0:2:2])


# ### Numpy array slicing

# In[46]:


arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])
print(arr[:4])

# [start: end]
# [start: end: step]
# step
print(arr[0: 5: 3])


# ### Slicing 2D arrays

# In[113]:


arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr)
print('\n')

res = arr[1, 1:4]
# print(res, '\n')

res = arr[0:, 4]
print(res, '\n')

print(arr[0:2, 1:4])


# In[142]:


arr = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(arr, '\n')

# arr[[rows], [columns]]
# arr[0:, 1:] for [col2] = 2, 4, 6

col1 = arr[0:, :1]
print(col1.shape)


# In[162]:


arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

print(arr, '\n')

col1 = arr[0:, :1]
col2 = arr[0:, 1:2]
col3 = arr[0:, 2:3]

row1 = arr[0]
row2 = arr[1]
row3 = arr[2]
row4 = arr[3]

print(f'col1: \n{col1}')
print(f'col2: \n{col2}')
print(f'col3: \n{col3}')
print('\n')
print(f'row1: \n{row1}')
print(f'row2: \n{row2}')
print(f'row3: \n{row3}')
print(f'row4: \n{row4}')


# ### Datatypes
# * i - integer
# * 
# b - boolea
# * 
# u - unsigned integ
# * f - float
# * c - complex float
# * m - timedelta
# * M - datetime
# * O - object
# * S - string
# * U - unicode string
# * V - fixed chunk of memory for other type (void)e ( void )

# narr = np.array([1, 2, 3], dtype='i4')
# starr = np.array([['apple', 'bananna', 'cherry']])
# 
# print(narr.dtype)
# print(starr.dtype)
# 
# # converting images to another
# floatArr = np.array([1.1, 2.001, 3.3321, 4.31231])
# # newarr = floatArr.astype('i')
# newarr = floatArr.astype(int)
# 
# boolArr = np.array([1, 0, 3, 0, 0, 1, 0])
# newarr = boolArr.astype(bool)
# 
# # copy array
# boolarrcopy = boolArr.copy()
# 
# print(newarr)
# 

# ### Shape / Reshape

# In[190]:


arr = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

dim = arr.shape
print(dim)
# reshape array
newarr = arr.copy()

newarr = newarr.reshape(2, 3)
print(newarr)

newarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
rearr = newarr.reshape(4, 3)
print(rearr)

# Flattening array
flatarr = rearr.reshape(-1)
print(flatarr)


# In[195]:


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
    for y in x:
        print(y)

arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)


# ### Array join

# In[209]:


arr1 = np.array([1, 2, 3])
arr2 = np.array([3, 2, 4])

arra = np.stack((arr2, arr1), axis=0)

# split
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 4)[2]

print(newarr)


# ### numpy random

# In[307]:


from numpy import random

x = random.randint(100)
randfloat = random.rand(3, 2)
# print(randfloat)
y = random.randint(2, size=(5,2))

y


# In[310]:


# Data distribution
randChoice = random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.6, 0.0], size=10)

x = random.choice([2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.4], size=(3, 5))
x


# ### Visualisation

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

# sns.distplot([0, 1, 2, 3, 4, 5])

x = random.rand(3, 2)

y = random.rand(3, 1)
# sns.displot(x+y)

sns.displot([1])


# ### Normal Distribution

# In[63]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


x = np.random.normal(size=(2, 3))

x = np.random.normal(loc=1, scale=3, size=(2, 3))
normal1 = np.random.normal(size=1000)


sns.distplot(normal1, hist=False)
# plt.show()


# ### Bionomial distribution
# 
# 
# Binomial Distribution is a Discrete Distribution. It describes the outcome of binary scenarios, 
# <br>**e.g. toss of a coin, it will either be head or tails.
# 
# It has 3 parameters:
# * n - number of trials.
# 
# * p - probability of occurence of each trial 
# <br>**(e.g. for toss of a coin 0.5 each).
# 
# * size - The shape of the returned array.
# 

# In[69]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = np.random.binomial(n = 10, p=0.5, size=10000)

sns.distplot(x, hist=True, kde=False)

plt.show()


# ### Poison distribution
# Poisson Distribution is a Discrete Distribution. It estimates how many times an event can happen in a specified time. 
# 
# ex. If someone eats twice a day what is the probability he will eat thrice?
# 
# It has 2 parameters:
# * lam - rate or known number of occurrences 
# <br>*e.g. 2 for above problem.
# 
# * size - The shape of the returned array.

# In[75]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.random.poisson(lam=2, size=1000)

# sns.distplot(x, kde=False)

# difference between normal & poisson distribution
# a = np.random.normal(loc=50, scale=7, size=1000)
# b = np.random.poisson(lam=50, size=1000)

# difference between binomial & poisson distribution
a = np.random.binomial(n=1000, p=0.01, size=1000)
b = np.random.poisson(lam=50, size=1000)

# sns.distplot(a, hist=False, label='normal')
sns.distplot(a, hist=False, label='binomial')
sns.distplot(b, hist=False, label='poisson')
sns.distplot()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




