import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import os
import argparse
import sys
import json

import dataset_MNIST

from model1 import model1
from model2 import model2
from model3 import model3


def train(model, device, train_loader, optimizer, epoch, writer, verbose=False):
    log_interval = 10

    # Set model to train mode (enables dropout etc.)
    model.train()

    # For each batch,
    # 1. Move data to GPU
    # 2. Zero gradients
    # 3. Generate predictions (forward pass)
    # 4. Calculate loss
    # 5. Propogate loss backwards
    # 6. Run Optimizer (step)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        writer.add_scalar("Loss/train", loss, epoch*len(train_loader)+batch_idx)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

# Calculate test set accuracy & loss
def test(model, device, test_loader):
    # Set model to eval mode (disables dropout etc.)
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

    return 100. * correct / len(test_loader.dataset)

def main(config_file):

    # Get dictionary of configuration parameters
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

    # Create directory for model & other files based on time ran
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_path = os.path.join(config_dict['save_path'], time_now)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize writer to save data to tensorboard
    writer = SummaryWriter(save_path)

    MNIST_shape = (1, 28, 28)
    FINAL_shape = (1, 2*config_dict['padding'] + 28, 2*config_dict['padding'] + 28,)

    # Set device (cuda or cpu)
    device = torch.device(config_dict['device'])

    # Set keyword arguments for data loaders
    train_kwargs = {'batch_size': config_dict['batch_size']}
    test_kwargs = {'batch_size': config_dict['batch_size']}
    if device == 'cuda':
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Define data transformations applied to dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(FINAL_shape[1:]),
        transforms.ToTensor()
    ])

    # Define datasets
    train_data = dataset_MNIST.MNIST(root=config_dict['data_path'], train=True, padding=config_dict['padding'],
                                     didactic=config_dict['didactic'], noise=config_dict['noise'],
                                     unique_didactic=config_dict['unique_didactic'],
                                     false_didactic=config_dict['false_didactic'], transform=transform)
    test_data = dataset_MNIST.MNIST(root=config_dict['data_path'], train=False, padding=config_dict['padding'],
                                    didactic=config_dict['didactic'], noise=config_dict['noise'],
                                    unique_didactic=config_dict['unique_didactic'],
                                    false_didactic=config_dict['false_didactic'], transform=transform)

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    # Initialize model & allocate to GPU
    model = getattr(sys.modules[__name__], config_dict['model'])(input_shape=FINAL_shape, classes=10).to(device)

    # Declare optimizer to use
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['l_r'])

    # Train for given epochs
    for epoch in range(config_dict['epochs']):
        train(model, device, train_loader, optimizer, epoch, writer, verbose=True)
        test(model, device, test_loader)
    print("total # of trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    writer.flush()
    writer.close()

    # Reinitialize specific layer
    # torch.nn.init.xavier_uniform(model.classifier.weight)

    # Test model accuracy
    test_acc = test(model, device, test_loader)

    # Save model in directory dependent on time trained
    model_path = os.path.join(save_path, 'final_model.h5')
    print("saving model to:", model_path)
    torch.save(model.state_dict(), model_path)

    # Save config file to directory to show what was used to train & results
    config_dict['test_acc'] = test_acc
    config_dict['model_path'] = model_path
    config_dict['save_path'] = save_path
    out_file = os.path.join(save_path, "config.json")
    with open(out_file, 'w') as outfile:
        json.dump(config_dict, outfile, indent = 4)
    print("out_file:", out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('-config_file', help='path to the configuration file (.json)')

    args = parser.parse_args()
    main(args.config_file)


    #

