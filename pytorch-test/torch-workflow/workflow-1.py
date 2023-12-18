import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
"""
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. 
That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to 
ensure that only a single OpenMP runtime is linked into the process, 
e.g. by avoiding static linking of the OpenMP runtime in any library. 
As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to 
allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
For more information, please see http://www.intel.com/software/products/support/.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
""" ENDS """

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
X = torch.arange(start, end, step, device=torch.device('cpu')).unsqueeze(dim=1)

print('device-type: ', torch.get_device(X))

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

    # train_data = train_data.numpy()
    # train_labels = train_labels.numpy()
    #
    # test_data = test_data.numpy()
    # test_labels = test_labels.numpy()

    # plot training in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')

    # plot testing in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Test data')

    if predictions is not None:
        # print(f'{test_data}\n type: {type(test_data)}')
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()
    pass

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
        self.weights = nn.Parameter(torch.randn(
            1,
            requires_grad=True,
            dtype=torch.float
        ))
        self.bias = nn.Parameter(torch.randn(
            1,
            requires_grad=True,
            dtype=torch.float
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.weights, x.T) + self.bias


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
    * The whole idea of training is for a model to move from some unknown parameters to some
    known parameters. 
    Or,
    in other words a poor representation of the data to a better 
    representation of the data.
    
    * How wrong the models predictions are is to calculate its LOSS funciton
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
epochs = 100

# track
epoch_count = []
loss_values = []
test_loss_values = []


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
    y_pred = y_pred.unsqueeze(dim=1)
    loss = loss_fn(y_pred, y_train)
    print(f'Loss: {loss}')

    # 4. optimizer zero grad
    optimizer.zero_grad()  # by default how the optimizer changes will acclumate

    # 5. perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 6. step the optimizer (perform gradient descent)
    optimizer.step()

    # turns off different settings in the model not need for evaluation / testing
    # dropout / batch norm layers
    model0.eval()

    # TEST-MODEL
    # turns off hte gradient tracking and many other things
    with torch.inference_mode():
        # 1. forward pass
        test_pred = model0(X_test)
        # 2. calculate loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        print('model parameters: \n', model0.state_dict())

""" Ends Model """


# print(f"weight: {weight}, bias: {bias}")

with torch.no_grad():
    y_preds_new = model0(X_test)

plot_predictions(predictions=y_preds)

plot_predictions(predictions=y_preds_new)
# print(y_preds_new)

""" Plot the loss curves """
np_loss_values = np.array(torch.tensor(loss_values).cpu().numpy())
np_test_loss_values = np.array(torch.tensor(test_loss_values).cpu().numpy())

plt.plot(epoch_count, np_loss_values, label="Train Loss")
plt.plot(epoch_count, np_test_loss_values, label="Test Loss")
plt.title('Training and testing loss curves')
plt.ylabel("Loss")
plt.xlabel('Epochs')
plt.legend()
plt.show()

"""
    Saving the model
    There are 3 methods for saving and loading the model
    1. torch.save() -> allows to save a pytorch object to pythons pickels format
    2. torch.load() -> allows to load a saved model
    3. torch.nn.Module.load_save_dict() -> this allows to load a models saved state dictionary
"""

from pathlib import Path

# 1. create models directory
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. create model save
MODEL_NAME = '01_torch_linear_model.pt'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
# print(f'Saving the model to {MODEL_SAVE_PATH}')
# torch.save(
#     obj=model0.state_dict(),
#     f=MODEL_SAVE_PATH
# )

"""
    Load the saved model
"""
loaded_model_0 = LinearRegressionModel()

# Load the saved state dict of model0 (this will update the new instance with updated parameter)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(f'saved model state dict: {model0.state_dict()}')
print(f'\nloaded model state dict: {loaded_model_0.state_dict()}')

# Run testing for checking
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)


with torch.inference_mode():
    y_preds = model0(X_test)

print(y_preds == loaded_model_preds)

# 7.41.14