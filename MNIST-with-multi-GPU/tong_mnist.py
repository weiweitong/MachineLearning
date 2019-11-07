import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# ************************************************************************************************

# Parameters and Dataloaders

train_batch_size = 64
test_batch_size = 1000

epochs = 20
learning_rate = 0.01
momentum = 0.5
random_seed = 1

log_interval = 200


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ************************************************************************************************

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True,
                   download=True, transform=transform),
    batch_size=train_batch_size, shuffle=True, num_workers=16)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False,
                   download=True, transform=transform),
    batch_size=test_batch_size, shuffle=True, num_workers=16)


# ************************************************************************************************


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = Net()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

net.to(device)

# ************************************************************************************************


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

train_loss, train_correct = [], []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0.0
    start = time.time()
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data  # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # net.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # print statistics
        running_loss += loss.item()
    correct = correct / len(train_loader.dataset)
    running_loss = running_loss / len(train_loader.dataset)
    print('%d loss: %.3f' % (epoch + 1, running_loss))
    train_correct.append(correct)
    train_loss.append(running_loss)

end = time.time()
print("time used:" + str(end-start))

print('Finished Training\n')
plt.plot(train_loss, label='train_loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(train_correct, label='train_corrects')
plt.legend(loc='upper right')
plt.show()
# ************************************************************************************************

net.eval()
test_loss = 0.0
correct = 0.0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

