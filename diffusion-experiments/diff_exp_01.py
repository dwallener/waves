#!/usr/local/bin/python3
# https://wandb.ai/capecape/miniai_ddpm/reports/Next-Frame-Prediction-Using-Diffusion-The-fastai-Approach--VmlldzozMzcyMTYy
# https://github.com/tcapelle/torch_moving_mnist/blob/main/nbs/01_data.ipynb

# basic imports

import wandb
from time import perf_counter
from fastprogress import progress_bar

import matplotlib.pyplot as plt
from functools import partial
from types import SimpleNamespace

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from data import apply_n_times, padding, RandomTrajectory, MovingMNIST
from utils import show_image, show_images


# pull static MNIST images (this is where we can drop in our own images to animate)

PATH = "."
mnist = MNIST(PATH, download=True)
print(mnist)
print(mnist.data.shape)

# visualize the dataset

show_images(mnist.data[0:5])
plt.show()

# generate moving sequence

import random

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

mnist_stats    = ([0.131], [0.308])
digit = torch.tensor(mnist.data[[0]])
# show_image(digit)
# plt.show()

# test transformation

angle = 12  # rotation in degrees
scale = 1.3 # scaling in percentage (1.0 = no scaling)
translate = (2,3) # translation in pixels
shear = 15 # deformation on the z-plane

# show_image(TF.affine(digit, angle, translate, scale, shear))
# plt.show()

padding(64)
pdigit = TF.pad(digit, padding=18)  #18 give us a 64x64 image (18x2 + 28)
# show_image(pdigit)
# plt.show()

tf = partial(TF.affine, angle=angle, translate=(-7,3), scale=scale, shear=shear)
# show_image(tf(pdigit))
# plt.show()

# show_images(apply_n_times(tf, pdigit, n=5), figsize=(10,20))
# plt.show()

# limit the transformations

affine_params = SimpleNamespace(
    angle=(-4, 4),
    translate=((-5, 5), (-5, 5)),
    scale=(.8, 1.2),
    shear=(-3, 3),
)

# test with randomness

angle     = random.uniform(*affine_params.angle)
translate = (random.uniform(*affine_params.translate[0]), 
             random.uniform(*affine_params.translate[1]))
scale     = random.uniform(*affine_params.scale)
shear     = random.uniform(*affine_params.shear)

tf = partial(TF.affine, angle=angle, translate=translate, scale=scale, shear=shear)

# show_images(apply_n_times(tf, pdigit, n=5), figsize=(10,20))
# plt.show()

traj = RandomTrajectory(affine_params)
print(traj)
# show_images(traj(pdigit))
# plt.show()

# move the image

move = partial(TF.affine, angle=0, scale=1, shear=(0,0))
def random_place(img, img_size=64):
    "Randomly place the digit inside the canvas"
    max_displacement = padding(img_size)
    x = random.uniform(-max_displacement, max_displacement)
    y = random.uniform(-max_displacement, max_displacement)
    return move(img, translate=(x,y))
     

# show_image(random_place(pdigit))
# plt.show()

# ok let's generate some dataset

print(mnist[0])
print(affine_params)
ds = MovingMNIST(affine_params=affine_params, num_frames=5)
show_images(ds._one_moving_digit())
plt.show()
digits = ds[0]
print(digits.shape)
show_images(digits)
plt.show()

# looks good! 

b = ds.get_batch(bs=32)
print(b.shape)
b = ds.get_batch(512)

# not sure what this section is about
# building it piece by piece, waiting for the meltdown

def cycle(bs=512, n=10):
    for _ in progress_bar(range(n), total=n):
        ti = perf_counter()
        b = ds.get_batch(bs)
        tf = perf_counter()
        print(f"Run took: {tf-ti:2.3f}s")
        if wandb.run is not None: wandb.log({"time_per_batch":tf-ti})

config = dict(bs=512, n=10)

with wandb.init(project="miniai_ddpm", job_type="dataloader_perf", config=config):
    cycle(config["bs"], config["n"])

ds.save(n_batches=10, bs=512)

def cycle_dl(bs=512, n=10):
    dl = iter(DataLoader(torch.load("mmnist.pt"), batch_size=bs))
    for _ in progress_bar(range(n), total=n):
        ti = perf_counter()
        b = next(dl)
        tf = perf_counter()
        print(f"Run took: {tf-ti:2.3f}s")
        if wandb.run is not None: wandb.log({"time_per_batch":tf-ti})
     

with wandb.init(project="miniai_ddpm", job_type="dataloader_perf", tags=["from_mem"], config=config):
    cycle_dl()

ds.concat=False

type(ds[0]), ds[0][0].shape

import nbdev; nbdev.nbdev_export()

