import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from captum.attr import *

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import json

import dataset_MNIST

from model1 import model1
from model2 import model2


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

def format_img(img):
    img = np.squeeze(img)
    # print(np.amax(img))
    if len(img.shape) > 2:
        img = np.swapaxes(img,0,2)
    # print(img.shape)
    # print(np.amax(img))

    return img

def main(config_file):
    # Get dictionary of configuration parameters
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            config_dict = json.load(json_file)

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cpu")

    MNIST_shape = (1, 28, 28)
    FINAL_shape = (1, 2 * config_dict['padding'] + 28, 2 * config_dict['padding'] + 28,)

    model = getattr(sys.modules[__name__], config_dict['model'])(input_shape=FINAL_shape, classes=10).to(device)
    model.load_state_dict(torch.load(config_dict['model_path']))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

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

    # Define inference dataset & data loader
    test_data = dataset_MNIST.MNIST(root=config_dict['data_path'], train=False, padding=config_dict['padding'],
                                    didactic=config_dict['didactic'], noise=config_dict['noise'],
                                    unique_didactic=config_dict['unique_didactic'],
                                    false_didactic=config_dict['false_didactic'],
                                    fixed_didactic=config_dict['fixed_didactic'],
                                    random_labels=config_dict['random_labels'], transform=transform)

    # Define data loaders
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model.eval()
    test(model, device, test_loader)

    # Objects needed for XAI methods
    grad = Saliency(model)
    smooth_grad = NoiseTunnel(grad)
    lrp = LRP(model)
    ig = IntegratedGradients(model)

    # Create figure, rows = # of samples, cols = # of XAI tests + 1
    N = 4
    seed = args.seed
    fig, ax = plt.subplots(nrows=N, ncols=6, figsize=(16, 12))
    np.random.seed(seed)
    for i in range(N):
        # Randomly select sample, allow gradient flow
        rint = np.random.randint(test_data.__len__())
        images, labels = test_data.__getitem__(rint)
        images = torch.unsqueeze(images, 0).to(device)
        images.requires_grad_()

        # Show Label & Prediction
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        im = ax[i, 0].text(0.5,0.5, "pred: {}\nlabel: {}".format(pred.detach().cpu().numpy()[0,0], labels))
        ax[i,0].axis('off')

        # Show original image
        tmp_img = images
        tmp_img = tmp_img.detach().numpy()
        im = ax[i, 1].imshow(format_img(tmp_img), cmap='gray', interpolation='nearest')
        plt.colorbar(im, ax=ax[i, 1])

        # Calculate & show gradient results
        attr = grad.attribute(images, target=labels)
        attr_result = attr.detach().numpy()
        im = ax[i, 2].imshow(format_img(attr_result))
        plt.colorbar(im, ax=ax[i, 2])

        # Calculate & show smoothgrad results
        attr = smooth_grad.attribute(images, nt_type='smoothgrad', nt_samples=50, target=labels)
        attr_result = attr.detach().numpy()
        im = ax[i, 3].imshow(format_img(attr_result))
        plt.colorbar(im, ax=ax[i, 3])

        # Calculate & show lrp results
        attr = lrp.attribute(images, target=labels)
        attr_result = attr.detach().numpy()
        im = ax[i, 4].imshow(format_img(attr_result))
        plt.colorbar(im, ax=ax[i, 4])

        # Calculate & show integrated gradient results
        attr, delta = ig.attribute(images, target=1, return_convergence_delta=True)
        attr_result = attr.detach().numpy()
        im = ax[i, 5].imshow(format_img(attr_result))
        plt.colorbar(im, ax=ax[i, 5])

    ax[0, 2].set_title('gradient', rotation=45)
    ax[0, 3].set_title('smoothgrad', rotation=45)
    ax[0, 4].set_title('lrp', rotation=45)
    ax[0, 5].set_title('ig', rotation=45)

    fig.tight_layout()
    plt.savefig(os.path.join(config_dict['save_path'], 'evaluations_{}.png'.format(args.seed)))
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('-config_file', help='path to the configuration file (.json)')
    parser.add_argument('-seed', default=42, help="random seed to determine which samples to use")
    args = parser.parse_args()

    main(args.config_file)

