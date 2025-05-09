# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_utils.ipynb.

# %% auto 0
__all__ = ['ifnone', 'not_none', 'set_seed', 'to_device', 'show_image', 'show_images']

# %% ../nbs/01_utils.ipynb 3
import math

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

import torch

# %% ../nbs/01_utils.ipynb 5
def ifnone(x, y):
    if x is None:
        return y
    else:
        return x

# %% ../nbs/01_utils.ipynb 11
def not_none(o):
    if isinstance(o, (list, tuple)):
        return all(not_none(x) for x in o)
    return o is not None

# %% ../nbs/01_utils.ipynb 14
def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available) - fastai"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %% ../nbs/01_utils.ipynb 15
def to_device(t, device):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise("Not a Tensor or list of Tensors")
    return t     

# %% ../nbs/01_utils.ipynb 18
def _fig_bounds(x):
    r = x//32
    return min(5, max(1,r))

# "Show a PIL or PyTorch image on `ax`."

def show_image(im, ax=None, figsize=None, title=None, **kwargs):
    # print("Inside show_image")
    cmap=None
    # Handle pytorch axis order
    if isinstance(im, torch.Tensor):
        im = im.data.cpu()
        if im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im, np.ndarray): 
        im=np.array(im)
    # Handle 1-channel images
    if im.shape[-1]==1: 
        cmap = "gray"
        im=im[...,0]
    
    if figsize is None: 
        figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: 
        _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, cmap=cmap, **kwargs)
    if title is not None: 
        ax.set_title(title)
    ax.axis('off')
    return ax

# %% ../nbs/01_utils.ipynb 21
def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`."
    if ncols is None: 
        ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: 
        titles = [None]*len(ims)
    axs = plt.subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): 
        show_image(im, ax=ax, title=t)
    