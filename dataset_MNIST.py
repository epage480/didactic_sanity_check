from __future__ import print_function, division
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets
from PIL import Image
import numpy as np
from typing import Any, Tuple
from matplotlib import pyplot as plt
import cv2

def get_nonoverlapping_coords(dim, padding):
    coords = []
    # Top Row
    for col in range(0, dim - padding):
        coords.append((0, col))
    # Bottom Row
    for col in range(0, dim - padding):
        coords.append((dim - padding, col))
    # Left Column
    for row in range(0, dim - padding):
        coords.append((row, 0))
    # Right Column
    for row in range(0, dim - padding):
        coords.append((row, dim - padding))
    return coords

def remove_overlapping(did_row, did_col, poss_coords, padding, dim):
    for row in range(max(0, did_row - padding), min(dim - padding + 1, did_row + padding)):
        for col in range(max(0, did_col - padding), min(dim - padding + 1, did_col + padding)):
            if (row, col) in poss_coords:
                poss_coords.remove((row, col))
    return poss_coords

# TODO: just a note that the images here are not normalized and have values between 0 & 255
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
class MNIST(datasets.MNIST):
    def __init__(self, padding=9, didactic=True, noise=True, unique_didactic=False, false_didactic=False,
                 fixed_didactic=True, random_labels=False, **kwargs):
        """

        :param padding: # of pixels to padd the image, also the width/height of the didactic & noise signals
        :param didactic: Adds a didactic signal to the original image
        :param noise: When true adds a noise signal which does not overlap with the didactic signal
        :param unique_didactic: Each label has its own didactic signal
        :param false_didactic: Adds a panda with s&p noise in other didactic position
        :param fixed_didactic: Flag indicating the didactic signal is in a fixed location for each label
        :param random_labels: Randomizes label regardless of actual label
        :param kwargs:
        """
        super(MNIST, self).__init__(**kwargs)

        # Initialization is same but 3 additional arguments
        'Initialization'
        self.padding = padding
        self.didactic = didactic
        self.noise = noise
        self.false_didactic = false_didactic
        self.unique_didactic = unique_didactic
        self.fixed_didactic = fixed_didactic
        self.random_labels = random_labels
        self.dim = 28+padding*2

        # Find coordinates to place images (4 images along top/bottom 3 along sides)
        dim = self.dim
        rows = [0, dim // 2 - padding // 2, dim - padding]
        cols = [0, dim // 2 - (padding + 2), dim // 2 + 2, dim - padding]

        if didactic:
            # Load didactic signal as either a single image (unique_didactic=False)
            # or a list of images corresponding to labels (unique_didactic=True)
            if unique_didactic:
                self.didactic_img = []
                for i in range(10):
                    didactic_img = imread('/home/eric/PycharmProjects/pytorch-XAI/Signal_Images/{}.png'.format(i))
                    didactic_img = resize(didactic_img, (self.padding, self.padding))
                    didactic_img = rgb2gray(didactic_img)
                    didactic_img = 255 * didactic_img
                    self.didactic_img.append(torch.from_numpy(didactic_img))
            else:
                didactic_img = imread('/home/eric/PycharmProjects/pytorch-XAI/Signal_Images/panda-face.png')
                didactic_img = resize(didactic_img, (self.padding, self.padding))
                didactic_img = rgb2gray(didactic_img)
                didactic_img = 255 * didactic_img
                self.didactic_img = torch.from_numpy(didactic_img)

            # Pre-compute coordinates for didactic signal if the locations are fixed (fixed_didactic=True)
            # or pre-compute potential coordinates which don't overlap original image (fixed_didactic=False)
            if self.fixed_didactic:
                self.didactic_coords = {}
                for i in range(3):  # 0,1,2
                    self.didactic_coords[i] = [rows[i], cols[0]]
                for i in range(1, 4):  # 3,4,5
                    self.didactic_coords[len(self.didactic_coords)] = [rows[-1], cols[i]]
                for i in range(1, -1, -1):  # 6,7
                    self.didactic_coords[len(self.didactic_coords)] = [rows[i], cols[-1]]
                for i in range(2, 0, -1):  # 8,9
                    self.didactic_coords[len(self.didactic_coords)] = [rows[0], cols[i]]
            else:
                self.poss_didactic_coords = get_nonoverlapping_coords(dim, padding)

        # Read in noise picture and pre-compute possible noise coordinates if the didactic
        # signal is in a fixed location (fixed_didactic=True)
        if noise:
            noise = imread('/home/eric/PycharmProjects/XAI/poop-emoji.png')
            noise_resized = resize(noise, (self.padding, self.padding))
            noise_resized = rgb2gray(noise_resized)
            noise_resized = 255 * noise_resized
            self.noise_img = torch.from_numpy(noise_resized)

            if fixed_didactic:
                # Find all locations which don't overlap with original image
                nonoverlapping_coords = get_nonoverlapping_coords(dim, padding)
                self.poss_noise_coords = {i:nonoverlapping_coords.copy() for i in range(10)}
                for i in range(10):
                    # Remove any coordinates which overlap with the didactic signal
                    did_row, did_col = self.didactic_coords[i]
                    self.poss_noise_coords[i] = remove_overlapping(did_row, did_col,self.poss_noise_coords[i],
                                                                padding, dim)
            else:
                self.poss_noise_coords = get_nonoverlapping_coords(dim, padding)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # Randomize label if desired
        if self.random_labels:
            target = np.random.randint(10)

        # Add black borders to data
        img = torch.nn.ZeroPad2d(self.padding)(img)

        # Add didactic signal to input image
        did_row, did_col = None, None
        if self.didactic:
            # Put panda picture in specific spot for each label if didactic signal locations are fixed
            # (fixed_didactic=True) or randomly select from possible locations (fixed_didactic=False)
            if self.fixed_didactic:
                did_row, did_col = self.didactic_coords[target]

                # Add corresponding didactic image for a given label if each label has its own didactic signal
                # (unique_didactic=True) or use the same image for all labels (unique_didactic=False)
                if self.unique_didactic:
                    img[did_row:did_row + self.padding, did_col:did_col + self.padding] = self.didactic_img[target]
                else:
                    img[did_row:did_row + self.padding, did_col:did_col + self.padding] = self.didactic_img

            else:
                did_row, did_col = self.poss_didactic_coords[np.random.choice(len(self.poss_didactic_coords))]
                if self.unique_didactic:
                    img[did_row:did_row + self.padding, did_col:did_col + self.padding] = self.didactic_img[target]
                else:
                    img[did_row:did_row + self.padding, did_col:did_col + self.padding] = self.didactic_img

        # Add a fake (noisy) didactic signal
        false_row, false_col = None, None
        if self.false_didactic:

            # Decide the target label of the false didactic signal
            false_label = None
            if self.unique_didactic or self.fixed_didactic:
                false_label_choices = [tgt for tgt in range(10) if tgt != target]
                false_label = np.random.choice(false_label_choices)

            # Select location to place false didactic signal
            if self.fixed_didactic:
                false_row, false_col = self.didactic_coords[false_label]
            else:
                poss_false_coords = self.poss_didactic_coords.copy()
                poss_false_coords = remove_overlapping(did_row, did_col, poss_false_coords,
                                                       self.padding, self.dim)
                false_row, false_col = poss_false_coords[np.random.choice(len(poss_false_coords))]

            # Select image to use (unique_didactic)
            if self.unique_didactic:
                noisy_didactic_img = torch.tensor(self.didactic_img[false_label])
            else:
                noisy_didactic_img = torch.tensor(self.didactic_img)

            # Add noise to the didactic image
            rows, cols = noisy_didactic_img.shape
            # Salt mode
            for i in range(10):
                y_coord = np.random.randint(0, rows - 1)
                x_coord = np.random.randint(0, cols - 1)
                # Color that pixel to white
                noisy_didactic_img[y_coord][x_coord] = 255

            # Pepper mode
            for i in range(10):
                y_coord = np.random.randint(0, rows - 1)
                x_coord = np.random.randint(0, cols - 1)
                # Color that pixel to white
                noisy_didactic_img[y_coord][x_coord] = 0

            img[false_row:false_row + self.padding, false_col:false_col + self.padding] = noisy_didactic_img

        # Add noise signal to the image
        if self.noise:

            # If the coordinates of the didactic signal are known prior, randomly select location
            # (fixed_didactic=True) otherwise find coordinates which will not overlap and randomly
            # select (fixed_didactic=False)
            if self.fixed_didactic:
                row, col = self.poss_noise_coords[target][np.random.choice(len(self.poss_noise_coords[target]))]

            else:
                # Remove any coordinates which overlap with the didactic signal
                poss_noise_coords = self.poss_noise_coords.copy()
                poss_noise_coords = remove_overlapping(did_row, did_col, poss_noise_coords,
                                   self.padding, self.dim)
                # Randomly select noise coordinate
                row, col = poss_noise_coords[np.random.choice(len(poss_noise_coords))]

            img[row:row+self.padding,col:col+self.padding] = self.noise_img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def plot_images(savename, dataset):
    fig, ax = plt.subplots(nrows=5, ncols=2)

    for i in range(10):
        idx = 0
        while (True):
            images, labels = dataset.__getitem__(idx)
            if labels == i:
                ax[i % 5, i // 5].imshow(images, cmap='gray', interpolation='nearest')
                break
            idx += 1
    fig.tight_layout()
    plt.savefig(savename)

if __name__ == "__main__":
    # Experiment 0
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=False,
                       noise=False, unique_didactic=False, false_didactic=False, fixed_didactic=False,
                       random_labels=False)
    plot_images('./sample_plots/experiment_0.png', train_data)

    # Experiment 1
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=False, false_didactic=False, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_1.png', train_data)

    # Experiment 2
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=False, false_didactic=False, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_2.png', train_data)

    # Experiment 3
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=False, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_3.png', train_data)

    # Experiment 4
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=True, false_didactic=False, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_4.png', train_data)

    # Experiment 5
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=False, fixed_didactic=False,
                       random_labels=False)
    plot_images('./sample_plots/experiment_5.png', train_data)

    # Experiment 6
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=True, false_didactic=False, fixed_didactic=False,
                       random_labels=False)
    plot_images('./sample_plots/experiment_6.png', train_data)

    # Experiment 7
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=False, false_didactic=True, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_7.png', train_data)

    # Experiment 8
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=True, fixed_didactic=True,
                       random_labels=False)
    plot_images('./sample_plots/experiment_8.png', train_data)

    # Experiment 9
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=True, fixed_didactic=False,
                       random_labels=False)
    plot_images('./sample_plots/experiment_9.png', train_data)

    # Randomization Experiment 1
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=False, false_didactic=False, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_1.png', train_data)

    # Randomization Experiment 2
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=False, false_didactic=False, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_2.png', train_data)

    # Randomization Experiment 3
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=False, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_3.png', train_data)

    # Experiment 4
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=True, false_didactic=False, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_4.png', train_data)

    # Randomization Experiment 5
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=False, fixed_didactic=False,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_5.png', train_data)

    # Randomization Experiment 6
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=True, unique_didactic=True, false_didactic=False, fixed_didactic=False,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_6.png', train_data)

    # Randomization Experiment 7
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=False, false_didactic=True, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_7.png', train_data)

    # Randomization Experiment 8
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=True, fixed_didactic=True,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_8.png', train_data)

    # Randomization Experiment 9
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, didactic=True,
                       noise=False, unique_didactic=True, false_didactic=True, fixed_didactic=False,
                       random_labels=True)
    plot_images('./sample_plots/randomization_experiment_9.png', train_data)

    # plt.show()