"""
    Putting all together
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

""" 
    CREATE DEVICE AGNOSTIC CODE
    If we have a GPU, code will use it, if not then use CPU 
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# DATA
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])

# SPLIT
train_split = int(0.8 * len(X))

X_train, y_train = X[: train_split], y[: train_split]
X_test, y_test = X[train_split:], y[train_split:]


# ******** PLOT THE DATA *********
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
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training images')

    # plot testing in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Test images')

    if predictions is not None:
        # print(f'{test_data}\n type: {type(test_data)}')
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()
    pass


# ******* ENDS ******

""" BUILD LINEAR MODEL """


class LinearRegressionModel(nn.Module):

    def __int__(self):
        super().__int__()
        # create model parameters
        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionModel()
print(model_1, model_1.state_dict())

# check current device
# next(model_1.parameters()).device

# set the model to use the targeted device
# TRAINING
# setup log
loss_fn = nn.L1Loss()
# optimizer
optimizer = torch.optim.SGD(lr= 0.001, params=model_1.parameters())

# training loop
torch.manual_seed(42)
epoches = 42

for epoch in range(epoches):
    # 1. Forward pass

    y_pred = model_1(X_train)
    # 2. calculate loss
    train_loss_value = loss_fn(y_pred, y_train)
    # 3. optimizer zero grad
    optimizer.zero_grad()

    model_1.backward()

    torch.manual_seed(42)

    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss_values = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {train_loss_value} | Test loss: {test_loss_values}')



