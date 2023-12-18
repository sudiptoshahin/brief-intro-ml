import torch
from torch import nn
import matplotlib.pyplot as plt

"""
    Data -> preparing and loading
    * spereadsheet
    * images
    * videos
    * audio
    * DNA
    * Text
    
    In machinelearning
    1. Get data in numerical representation
    2. Build a model to learn patterns in that numerical representation
"""

# Linear Regression
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print(f'X[:10] = {X[: 10]}')
# print(f'y[: 10] = {y[: 10]}')

"""
    Spliting data into training and testing
    
"""
train_split = int(0.8 * len(X))
X_train, y_train = X[: train_split], y[: train_split]
X_test, y_test = X[train_split:], y[train_split:]

"""  Visualize the data  """


def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    # plot training in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')

    # plot testing in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Test data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()


""" 
    BUILD MODEL
    
    * torch.nn - contains all the building blocks for computation graph
    * torch.nn.Parameter - what parameters should our model try and learning often a torch layer
    * torch.optim - this is where the optimizers in pytorch
    * def forward() - all nn.Module subclasses needs to override the forward() method
"""


class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# Create random seed
torch.manual_seed(42)

model0 = LinearRegressionModel()

# get the list of our model parameters
print(list(model0.parameters()))

# get the list of named parameters
print(model0.state_dict())

# making prediction using `torch.inference_mode()`
# check how well it predicts `y_test` based on `X_test`
# When we pass data through our model, its going to run it with forward() method

# inference_mode() disable the gradient / all other stuffs
# and making it faster
# with torch.inference_mode():
#     y_preds = model0(X_test)

# only disable the no_grad()
with torch.no_grad():
    y_preds = model0(X_test)

# print(y_preds)
# plot_predictions(predictions=y_preds)
"""
    TRAIN A MODEL
    --------------
    The whole idea of training is for a model to move from some unknown parameters to some
    known parameters. Or in other words a poor representation of the data to a better 
    representation of the data.
    
    How wrong the models predictions are is to calculate its LOSS funciton
    # THINGS WE NEED TO TRAIN:
    * LOSS function
    * OPTIMIZER: Takes into account the loss of a model and adjusts the models parameter(Weights and bias)
    
    # Specifically for PyTorch: we need
    * A training loop
    * A testing loop
"""

# loss function
loss_fn = nn.L1Loss()

# setup optimizers
_params = model0.parameters()
optimizer = torch.optim.SGD(params=_params, lr=0.05)

""" 
    Build a training & testing loop
    1. loop through data
    2. Forward pass - forward propagation
    3. Calculate the loss
    4. Optimizer zero grad
    5. Loss backward - move backwards through the network to calculate the gradients of
    each of the parameters of out model with respect to the loss
    6. Optimizer step - use optimizer to adjust our models parameters tor try and improve the loss
"""

# and epoch is one loop through the data
# it is a hyperparameter because we set it
epochs = 1

# Training
# 1. loop through the data
for epoch in range(epochs):
    # set the model in training mode
    # train mode in pytorch sets all parameters that require gradients to require gradients
    model0.train()

    # turns off gradient tracking
    model0.eval()

    # 2. forward pass
    y_pred = model0(X_train)

    # 3. calculate loss
    loss = loss_fn(y_pred, y_train)

    # 4. optimizer zero grad
    optimizer.zero_grad()  # by default how the optimizer changes will acclumate

    # 5. perform backpropagation on the loss with respect to the parameters of the model

    # 6. step the optimizer (perform gradient descent)
    optimizer.step()

    # turns off hte gradient tracking
    model0.eval()


# 6.25.05