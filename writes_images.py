import torch
from model import AutoEncoder
import numpy as np
from random import randint
from torchvision.utils import save_image
import os
import argparse


def transfer(model, x, version):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to('cuda')
    model.eval()
    if version == 'a':
        out = model(x, version='a')
        return torch.cat([x, out])
    elif version == 'b':
        out = model(x, version='b')
        return torch.cat([x, out])


def write_images(model, image_a, image_b, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    compare_x = transfer(model, image_a, 'a')
    save_image(compare_x.data.cpu(), '{}/sample_image_a.png'.format(dir_name))

    compare_x = transfer(model, image_b, 'b')
    save_image(compare_x.data.cpu(), '{}/sample_image_b.png'.format(dir_name))

    compare_x = transfer(model, image_b, 'a')
    save_image(compare_x.data.cpu(), '{}/sample_image_b_to_a.png'.format(dir_name))

    compare_x = transfer(model, image_a, 'b')
    save_image(compare_x.data.cpu(), '{}/sample_image_a_to_b.png'.format(dir_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-saved_dir', type=str, default='saved_images', help='where to save the images')
    parser.add_argument('-model_name', type=str, default='model', help='model name')
    parser.add_argument('-images_a', type=str, default='a.npy', help='dataset of images a')
    parser.add_argument('-images_b', type=str, default='b.npy', help='dataset of images b')

    args = parser.parse_args()

    model = AutoEncoder(image_channels=3).to('cuda')
    model.load_state_dict(torch.load('saved_models/{}.pt'.format(args.model_name)))

    train_dataset_array_a = np.load(args.images_a)
    train_dataset_array_b = np.load(args.images_b)

    x_a = train_dataset_array_a[randint(1, 10)]
    x_b = train_dataset_array_b[randint(1, 10)]

    a_max = len(train_dataset_array_a)
    b_max = len(train_dataset_array_b)

    write_images(model, x_a, x_b, args.saved_dir)

