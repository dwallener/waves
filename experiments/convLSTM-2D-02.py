#!/usr/local/bin/python3
# https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
# just include the class files here
# from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
from Seq2Seq import Seq2Seq

import io
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time

# some run length vars
target_device = 'cpu'
training_size = 10000
val_size = 250
num_epochs = 5
learning_rate = 0.001
# set True for end of epoch display of input/output
visualize = True

# do args stuff
parser = argparse.ArgumentParser()

parser.add_argument('--target', required=False)
parser.add_argument('--ts', required=False)
parser.add_argument('--epochs', required=False)
parser.add_argument('--lr', required=False)

args = parser.parse_args()

if args.target is not None:
    target_device = args.target

if args.ts:
    training_size = int(args.ts)
    val_size = int(args.ts)

if args.epochs:
    num_epochs = int(args.epochs)

if args.lr:
    learning_rate = float(args.lr) 

# set device to proper target

device = 'cpu'

if target_device == 'mps':
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

if target_device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("No CUDA!")

print("Retargetting to: ", device)

wave_file = "wave-disturbance-01-" + str(training_size) + "ts-64px.npy"

# show the run params
print("Run conditions:")
print("Training Size  : ", training_size)
print("Validation Size: ", val_size)
print("Epochs to train: ", num_epochs)
print("Learning Rate  :", learning_rate)
print("Target device  :", device)
print("Dataset        :", wave_file)

# Support function for displaying image sequneces

def display_image_sequence(images, start_index, max_frames):
    """
    Display 20 images in sequence, starting from the given index.
    
    :param images: A numpy array of shape (1000, 64, 64) where each element is a 64x64 image.
    :param start_index: The index of the first image to display.
    """
    
    # Ensure start_index is within the range of the images array
    if start_index < 0 or start_index >= len(images) - max_frames:
        raise ValueError("Start index is out of the valid range.")
    
    # Prepare the figure and axes for animation
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis
    
    # Function to update the frame
    def update(i):
        if images.ndim == 5:
            ax.imshow(images[0, 0, start_index + i], cmap='gray')
        if images.ndim == 4:
            ax.imshow(images[0, start_index + i], cmap='gray')
        if images.ndim == 3:
            ax.imshow(images[start_index + i], cmap='gray')
        ax.set_title(f'Image Index: {start_index + i}')
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=max_frames, interval=250, repeat=False)  # Update every 500 ms
    
    plt.show(block=False)
    plt.pause(3)
    plt.close()


# Original ConvLSTM cell as proposed by Shi et al.
# Class Seq2Seq now tucked away in its own source file

# This was designed around MNIST (Moving NIST)
# This is structured as a 10000 x 20 x 64 x 64 numpy array
# 10000 is the number of unique sequences
# 20 is the number of sequences inside each sequence - this is where next frame matters
# For MNIST, the boundary between 19-20, 39-40, etc, doesn't matter

# So for our purposes on just one long sequence...we should be fine if it's chunked into 20s
# Not sure why it's transposed though...
    
#MovingWave = np.load('wave-disturbance-01-100000ts-64px.npy')
#MovingWave = np.load(wave_file).transpose(1,0,2,3)
MovingWave = np.load('mnist_test_seq.npy').transpose(1,0,2,3)
print("Original wave shape: ", MovingWave.shape)
print("Image shape: ", MovingWave[0,0,:,:].shape)
display_image_sequence(MovingWave, 0, 10)

# that's all debug shit - now clear the var and start for real
#MovingWave = np.load('wave-disturbance-01-10000ts-64px.npy')
MovingWave = None
MovingWave = np.load(wave_file)
print("Original wave shape: ", MovingWave.shape)
print("Image shape: ", MovingWave[0,:,:].shape)
display_image_sequence(MovingWave, 100, 10)

# reshape into 20-slice chunks.
MovingWaveReshape = MovingWave.reshape(int(training_size/20), 20, 64, 64).astype(np.float32)
print("New Wave Shape: ", MovingWaveReshape.shape)
print("New Image shape:", MovingWaveReshape[0,0,:,:])
display_image_sequence(MovingWave, 800, 10)
# the shape is 10000 timesetps X 20 animations X 64 pixels X 64 pixels
# our wave sim is 1000/10000 timesetps x 1 animation X 64 pixels X 64 pixels
#
# for simplicity sake, let's just use the same format so we don't have to 
# reshape anything downstream...at least it allows us to shuffle on the 20
# which should be ok


#np.random.shuffle(MovingWaveReshape)
train_idx = int(training_size/20)
train_frac = int(train_idx/4)
val_idx = train_idx
train_data = MovingWaveReshape[:train_frac*2]
val_data = MovingWaveReshape[train_frac*2:train_frac*3]
test_data = MovingWaveReshape[train_frac*3: train_frac*4]

# TODO: save tensor and reload as tensor, because this is a slooooow step
tensor_save_path = "nist-tensor.pt"

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)
    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10, 20)
    return batch[:, :, rand-10:rand], batch[:, :, rand]


# Training Data Loader
train_loader = DataLoader(train_data, shuffle=False, batch_size=16, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=False, batch_size=16, collate_fn=collate)

# Get a batch
input, _ = next(iter(val_loader))

# Reverse process before displaying
input = input.cpu().numpy() * 255.0     


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(64, 64), num_layers=3, device=device).to(device)

optim = Adam(model.parameters(), lr=learning_rate)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')

num_epochs = num_epochs

# end of epoch image snapshots
sav_input = []
sav_target = []
sav_output = []
sav_input_npy = np.zeros((num_epochs, 64, 64))
sav_target_npy = np.zeros((num_epochs, 64, 64))
sav_output_npy = np.zeros((num_epochs, 64, 64))

# method 1: using DataLoader
print("Dataloader path")
for epoch in range(1, num_epochs+1):
    
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        output = model(input)
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)                       

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:                          
            output = model(input)                                   
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(val_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
    path = "checkpoint-" + str(epoch) + ".pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': train_loss,
            }, path)
    
    # end of epoch...let's take a look-see

    # save the stuffs
    print("Saving for post-processing...")
    sav_input.append(input)
    sav_target.append(target)
    sav_output.append(output)
    sav_input_npy[epoch-1] = input[0, 0, 0, :, :].cpu()
    sav_target_npy[epoch-1] = target[0, 0, :, :].cpu()
    sav_output_npy[epoch-1] = output[0, 0, :, :].cpu()


    # viz the last target/output
    if visualize:
        print("Input shape: ", input.shape)
        print("Target Shape: ", target.shape)
        print("Output Shape: ", output.shape)
        display_image_sequence(input.cpu(), 0, 1)
        display_image_sequence(target.cpu(), 0, 1)
        display_image_sequence(output.cpu(), 0, 1)
    else:
        print("In-training visualization is off")

print("Saving epoch snapshots to snapshots.sav")
print("Saving numpys to input/output/target-snapshots.npy")
np.save("input-snapshot.npy", sav_input_npy)
np.save("target-snapshot.npy", sav_target_npy)
np.save("output-snapshot.npy", sav_output_npy)

