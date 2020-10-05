import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from Iter import Iterator
from SSIM import SSIM
from torch.autograd import Variable
from model import AutoEncoder, Discriminator
import argparse
import numpy as np
from torchvision import datasets
from torchvision import transforms
import os

parser = argparse.ArgumentParser()

parser.add_argument('-face_a_dir', type=str, default='face_a', help='directory containing aligned faces')
parser.add_argument('-face_b_dir', type=str, default='face_b', help='output data directory')
parser.add_argument('-saved_dir', type=str, default='saved_models', help='training batch size')
parser.add_argument('-batch_size', type=int, default=1, help='training batch size')
parser.add_argument('-n_steps', type=int, default=10000, help='Number of training loops')
parser.add_argument('-save_iter', type=int, default=1000, help='How many train loops until the model is saved')
parser.add_argument('-discriminator', type=bool, default=False, help='Use adversarial loss in training loop')
parser.add_argument('-model_name', type=str, default='model', help='Name of the saved model')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.face_a_dir) or os.path.exists(args.face_b_dir):
    assert FileNotFoundError

if not os.path.exists(args.saved_dir):
    os.makedirs(args.saved_dir)

print('loading data...')
dataset_a = datasets.ImageFolder(root=args.face_a_dir, transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ToTensor(),
]))

dataset_b = datasets.ImageFolder(root=args.face_b_dir, transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ToTensor(),
]))
dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=len(dataset_a), shuffle=True)
dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=len(dataset_b), shuffle=True)


train_dataset_array_a = next(iter(dataloader_a))[0].numpy()
train_dataset_array_b = next(iter(dataloader_b))[0].numpy()

np.save('a.npy', train_dataset_array_a)
np.save('b.npy', train_dataset_array_b)

save = True

itera = iter(Iterator(train_dataset_array_a, args.batch_size))
iterb = iter(Iterator(train_dataset_array_b, args.batch_size))

model = AutoEncoder(image_channels=3).to(device)
discriminator = Discriminator().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_b = torch.optim.Adam(model.parameters(), lr=1e-3)

mse = nn.L1Loss()
ssim_loss = SSIM()


def dis_loss(prob_real_is_real, prob_fake_is_real):
    EPS = 1e-12
    return torch.mean(-(torch.log(prob_real_is_real + EPS) + torch.log(1 - prob_fake_is_real + EPS)))


def gen_loss(original, recon_structed, validity=None):
    ssim_l = -ssim_loss(recon_structed, original)
    if validity:
        gen_loss_GAN = torch.mean(-torch.log(validity + 1e-12))
        # gen_loss_L1 = torch.mean(torch.abs(original - recon_structed))
        return 5 * ssim_l + gen_loss_GAN
    else:
        return ssim_l


def train_step(images, version='a'):
    _decoder_image = model(images, version=version)
    if args.discriminator:
        with torch.no_grad():
            validity = discriminator(_decoder_image)

        _loss = gen_loss(_decoder_image, images, validity)
    else:
        _loss = gen_loss(_decoder_image, images)
    optimizer.zero_grad()
    _loss.backward(retain_graph=True)
    optimizer.step()
    if args.discriminator:
        validity = discriminator(_decoder_image.detach())
        real_dis = discriminator(images)

        d_loss = dis_loss(real_dis, validity)
        optimizer_b.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizer_b.step()
    return _loss


print('training for {} steps'.format(args.n_steps))

for epoch in range(args.n_steps):
    # for idx, (images, _) in enumerate(dataloader):
    a = next(itera)
    b = next(iterb)
    images_a = torch.tensor(a, device=device).float()
    images_a = images_a.to(device)

    images_b = torch.tensor(b, device=device).float()
    images_b = images_b.to(device)
    loss_a = train_step(images_a, version='a')
    loss_b = train_step(images_b, version='b')
    to_print = "Epoch[{}/{}] Loss A:{}, Loss B:{}".format(epoch+1, args.n_steps, loss_a.data, loss_b.data)
    if epoch % 1000 == 0:
        print(to_print)
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, '{}/{}.pt'.format(args.saved_dir, args.model_name))
if save:
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, '{}/model.pt'.format(args.saved_dir))
else:
    model.load_state_dict(torch.load('{}/model.pt'.format(args.saved_dir)))
