import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torchvision.models as models
# import torch.nn.functional as F


training_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

print(training_dataset.data.shape)
print(training_dataset.targets.shape)

# display the first 5 images
# plt.figure(figsize=(10, 10))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(training_dataset.data[i])
#     plt.title(training_dataset.targets[i])
#     plt.axis("off")
# plt.show()

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(test_dataset.data.shape)
print(test_dataset.targets.shape)

# load the data into a dataloader
train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)



# display the first 5 images
# plt.figure(figsize=(10, 10))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(training_dataset.data[i])
#     plt.title(training_dataset.targets[i])
#     plt.axis("off")
# plt.show()

# define the device
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# define the model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = get_device()
model = NeuralNetwork().to(device)
print(model)



learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
     # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward pass
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # print the loss every 100 batches
        if batch % 100 == 0:
            # display only 5 images in matplotlib
            if batch == 0:  # only display the first batch  
                plt.figure(figsize=(10, 10))
                for i in range(5):
                    plt.subplot(1, 5, i + 1)
                    plt.imshow(X[i].cpu().numpy().reshape(28, 28))
                    plt.title(f"Pred: {pred[i].argmax(0)}, Actual: {y[i]}")
                    plt.axis("off") 
                plt.show()
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train_loop(train_dataloader, model, loss_fn, optimizer)
test_loop(test_dataloader, model, loss_fn)

# save the model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print(model)

# modify the model
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(model)

# save the model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")