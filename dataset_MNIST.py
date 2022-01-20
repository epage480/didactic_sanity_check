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


# TODO: just a note that the images here are not normalized and have values between 0 & 255
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
class MNIST(datasets.MNIST):
    def __init__(self, padding=9, didactic=True, noise=True, unique_didactic=False, false_didactic=False,
                 **kwargs):
        super(MNIST, self).__init__(**kwargs)

        # Initialization is same but 3 additional arguments
        'Initialization'
        self.padding = padding
        self.didactic = didactic
        self.noise = noise
        self.false_didactic = false_didactic
        dim = 28+padding*2

        # Find coordinates to place images (4 images along top/bottom 3 along sides)
        rows = [0, dim // 2 - padding // 2, dim - padding]
        cols = [0, dim // 2 - (padding + 2), dim // 2 + 2, dim - padding]
        self.didactic_coords = {}
        for i in range(3):  # 0,1,2
            self.didactic_coords[i] = [rows[i], cols[0]]
        for i in range(1, 4):  # 3,4,5
            self.didactic_coords[len(self.didactic_coords)] = [rows[-1], cols[i]]
        for i in range(1, -1, -1):  # 6,7
            self.didactic_coords[len(self.didactic_coords)] = [rows[i], cols[-1]]
        for i in range(2, 0, -1):  # 8,9
            self.didactic_coords[len(self.didactic_coords)] = [rows[0], cols[i]]


        # Create specific location of didactic signal for each digit
        if didactic:

            if unique_didactic:
                # Read in unique didactic image for each label
                self.didactic_img = []
                for i in range(10):
                    didactic_img = imread('/home/eric/PycharmProjects/pytorch-XAI/Signal_Images/{}.png'.format(i))
                    didactic_img = resize(didactic_img, (self.padding, self.padding))
                    didactic_img = rgb2gray(didactic_img)
                    didactic_img = 255 * didactic_img
                    self.didactic_img.append(torch.from_numpy(didactic_img))
            else:
                # Read in didactic (panda) image
                didactic_img = imread('/home/eric/PycharmProjects/pytorch-XAI/Signal_Images/panda-face.png')
                didactic_img = resize(didactic_img, (self.padding, self.padding))
                didactic_img = rgb2gray(didactic_img)
                didactic_img = 255 * didactic_img
                self.didactic_img = torch.from_numpy(didactic_img)

        # Find possible locations for noise signal for each didactic signal
        # (doesn't overlap with original image or didactic signal)
        self.noise_img = None
        if noise:
            # Find all locations which don't overlap with original image
            self.poss_noise_coords = {i:[] for i in range(10)}
            for i in range(10):
                # Top Row
                for col in range(0, dim-padding):
                    self.poss_noise_coords[i].append((0, col))
                # Bottom Row
                for col in range(0, dim-padding):
                    self.poss_noise_coords[i].append((dim-padding, col))
                # Left Column
                for row in range(0, dim-padding):
                    self.poss_noise_coords[i].append((row, 0))
                # Right Column
                for row in range(0, dim-padding):
                    self.poss_noise_coords[i].append((row, dim-padding))

                # Remove any coordinates which overlap with the didactic signal
                did_row, did_col = self.didactic_coords[i]
                for row in range(max(0, did_row-padding), min(dim-padding+1, did_row+padding)):
                    for col in range(max(0, did_col - padding), min(dim - padding+1, did_col+padding)):
                        if (row, col) in self.poss_noise_coords[i]:
                            self.poss_noise_coords[i].remove((row, col))

            # Read in noise picture
            noise = imread('/home/eric/PycharmProjects/XAI/poop-emoji.png')
            noise_resized = resize(noise, (self.padding, self.padding))
            noise_resized = rgb2gray(noise_resized)
            noise_resized = 255 * noise_resized
            self.noise_img = torch.from_numpy(noise_resized)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # Add black borders to data
        img = torch.nn.ZeroPad2d(self.padding)(img)

        if self.didactic:
            # Put panda picture in specific spot for each digit
            row, col = self.didactic_coords[target]
            if type(self.didactic_img) is list:
                img[row:row + self.padding, col:col + self.padding] = self.didactic_img[target]
            else:
                img[row:row + self.padding, col:col + self.padding] = self.didactic_img

        if self.noise:
            # Find random spot for noise signal which does not overlap with didactic signal & place
            row, col = self.poss_noise_coords[target][np.random.choice(len(self.poss_noise_coords[target]))]
            img[row:row+self.padding,col:col+self.padding] = self.noise_img

        if self.false_didactic:
            # Select random didactic location different from the real signal
            choices = [tgt for tgt in self.didactic_coords if tgt != target]
            row, col = self.didactic_coords[np.random.choice(choices)]

            # Select image to use (unique_didactic)
            if type(self.didactic_img) is list:
                noisy_didactic_img = torch.tensor(self.didactic_img[target])
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
            # # mean = 0.
            # # var = 32.
            # # sigma = var ** 0.5
            # # gauss = np.random.normal(mean, sigma, noisy_didactic_img.shape)
            # # noisy_didactic_img = noisy_didactic_img + gauss
            img[row:row + self.padding, col:col + self.padding] = noisy_didactic_img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target

if __name__ == "__main__":
    train_data = MNIST(padding=12, root='/home/eric/Datasets', train=True, download=False, noise=False, unique_didactic=False, false_didactic=True)

    # # Create subplots, remove ticks, add titles
    # N = 5
    # fig, ax = plt.subplots(nrows=N, ncols=1)
    #
    # for i in range(N):
    #     rint = np.random.randint(10000)
    #     images, labels = train_data.__getitem__(rint)
    #     # print(np.amax(images))
    #     print(labels)
    #
    #     ax[i].imshow(images, cmap='gray', interpolation='nearest')
    #
    # fig.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=2)

    for i in range(10):
        idx = 0
        while(True):
            images, labels = train_data.__getitem__(idx)
            if labels == i:
                ax[i%5,i//5].imshow(images, cmap='gray', interpolation='nearest')
                break
            idx += 1
    fig.tight_layout()
    plt.show()