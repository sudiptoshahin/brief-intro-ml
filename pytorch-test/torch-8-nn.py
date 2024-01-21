import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

n_samples = 100

# create circle
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
# print(len(X), len(y))
print(f'First 5 samples of X: {X[:5]}')
print(f'First 5 sample of y: {y[:5]}')

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "y": y
})


# # visualize
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y)
# plt.show()


"""
1. Check input and output shapes
2. view the first example of features
"""

X_sample = X[0]
y_sample = y[0]

print(f'Values for one sample of X: {X_sample}')
print(f'Values of one sample of y: {y_sample}')

""" 3. Turn data into tensors and create train test splits """

_X = torch.tensor(torch.from_numpy(X), dtype=torch.float16, requires_grad=False)
_X.detach().clone()
_y = torch.tensor(torch.from_numpy(y), dtype=torch.int8, requires_grad=False)
_y.detach().clone()

# float16, int8

""" Split data into testing and testing
    sklearn random_state and torch random_state must be same 
"""
torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(_X, _y,
                                                    test_size=0.2,
                                                    random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))

"""
    Build the model
    --------------------
    1. classify blue and red dots
    2. setup device agnostic code (CPU/GPU)
    3. construct the model (subclassing 'nn.Module')
    4. Define loss function and optimizer
    5. Creating and test loop
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
    1. Subclass 'nn.Module'
    2. Create 2 'nn.Linear' layers that capable of handling the
    shapes of the data.
    3. Define a `forward` method that outlines the forward pass
    4. Instantiate an instance of our model class and sent it to the
    target device.
"""
# construct model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # create nn layers
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

        # upgrade the class with built-in forward
        self.two_linear_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )

    def forward(self, x):
        # x -> layer1 -> layer2 -> output
        # return self.layer_2(self.layer_1(x))
        return self.two_linear_layers


# Instantiate model
model0 = CircleModelV0().to(device)
print(next(model0.parameters()).device)

# replicate the above model by using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# print(model_0.state_dict())

# Make predictions
# untrained_preds = model_0(X_test.to(device))
# print(f'Lengths of predictions: {len(untrained_preds)}\n Shape: {untrained_preds.shape}')
















