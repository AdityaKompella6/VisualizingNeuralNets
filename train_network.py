import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        return out

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = MLP(input_size=784, hidden_size=256, output_size=10).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

transformation = transforms.Compose(
    [
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(10, (0.01, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

transformation_eval = transforms.Compose(
    [
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
n_epochs = 10
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 1e-5)

dataset = MNIST(root="data/", download=True, transform=transforms.ToTensor())
# Determine the sizes of the train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test sets
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False
)
for epoch in range(n_epochs):
    loss_epoch = 0
    for batch in train_loader:
        # Forward pass
        X, y = batch
        X = transformation(X)
        X = X.to(device)
        y = y.to(device)
        X = X.squeeze(1)
        X = X.view(-1, 28 * 28)
        with torch.cuda.amp.autocast():
            outputs = model(X)
            loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_epoch += loss.item()

    if (epoch + 1) % 1 == 0:
        print(
            f"Epoch [{epoch + 1}/{n_epochs}], Loss: {(loss_epoch/len(train_loader)):.4f}"
        )

num_correct = 0
model.eval()
for batch in test_loader:
    # Forward pass
    X, y = batch
    X = transformation_eval(X)
    X = X.to(device)
    y = y.to(device)
    X = X.squeeze(1)
    X = X.view(-1, 28 * 28)
    with torch.cuda.amp.autocast():
        outputs = model(X)
    preds = torch.argmax(outputs, dim=1)
    correct = torch.sum(preds == y)
    num_correct += correct
print(f"Total Accuracy: {num_correct/len(test_dataset)}")

print("Saving Model......")
torch.save(model.state_dict(), "model.pth")
