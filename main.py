import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# import concepts of Pytorch


# create a tensor


x = torch.tensor([1, 2, 3])

# print(x)

# create a tensor of zeros and onces

zeros_tensor = torch.zeros(2, 4)  # 2,4 are rows and columns respectively
# print(zeros_tensor)

ones_tensors = torch.ones(2, 4)

# print(ones_tensors)


# get shape of tensors

# print(ones_tensors.shape,ones_tensors.dtype)

# Gradient

t = torch.tensor(2.0, requires_grad=True)

# print(t)

y = t**2 + 3 * t + 1  # t^2 +3t + 1    (quadratic equation)
# print(y)

y.backward()  #  computes dy/dt    differentiation of y wrt  t

print(t.grad)  # t*2+3 = 7

"""Building a Neural Network with Pytorch"""


# Transform : Convert image to tensor and normalize

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


# Download training and test data

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Build Neural Network


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 digits (0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten 28x28 images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DigitClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training Loop

for epoch in range(10):  # train for 5 epochs
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# Evaluate on Test Data


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
