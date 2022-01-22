import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
import numpy as np

# This model is meant to be the same used in this paper: https://arxiv.org/abs/1711.00867
# It consists of 26,914,826 parameters and may be incorrect, but it does obtain the accuracy of 98.4% as described
# in the paper, consider emailing the authors to elaborate on their architecture
class model3(torch.nn.Module):
    def __init__(self, input_shape=(1,28,28), classes=10):
        super(model3, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_shape[0], 1024, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.classifier = torch.nn.Linear(802816, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.log_softmax(self.classifier(x), dim=1)
        return(x)


def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(params)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('/home/eric/Datasets', train=True, download=False, transform=transform)
    test_data = datasets.MNIST('/home/eric/Datasets', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                num_workers=1, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                num_workers=1, pin_memory=True, shuffle=True)

    model = model3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in range(1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Count # of model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)