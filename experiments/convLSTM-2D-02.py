#!/usr/local/bin/python3
# https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
# just include the class files here
# from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader

import io
import os
import imageio
from ipywidgets import widgets, HBox
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# some run length vars
target_device = 'cpu'
training_size = 1000
val_size = 250
num_epochs = 5
learning_rate = 0.001
test_skip_rate = 10 # stride size through post-training validation test


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

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
    

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device=device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output


class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])

        return nn.Sigmoid()(output)



#MovingWave = np.load('wave-disturbance-01-100000ts-64px.npy')
MovingWave = np.load(wave_file)
print("Original wave shape: ", MovingWave.shape)

# reshape into 20-slice chunks.
MovingWave = MovingWave.reshape(int(training_size/20), 20, 64, 64).astype(np.float32)
print("Reshaped wave shape:", MovingWave.shape)
# the shape is 10000 timesetps X 20 animations X 64 pixels X 64 pixels
# our wave sim is 1000/10000 timesetps x 1 animation X 64 pixels X 64 pixels
#
# for simplicity sake, let's just use the same format so we don't have to 
# reshape anything downstream...at least it allows us to shuffle on the 20
# which should be ok


np.random.shuffle(MovingWave)
train_idx = int(training_size/20)
train_frac = int(train_idx/4)
val_idx = train_idx
train_data = MovingWave[:train_frac*2]
val_data = MovingWave[train_frac*2:train_frac*3]
test_data = MovingWave[train_frac*3: train_frac*4]

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
train_loader = DataLoader(train_data, shuffle=True, batch_size=16, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True, batch_size=16, collate_fn=collate)

# Get a batch
input, _ = next(iter(val_loader))

# Reverse process before displaying
input = input.cpu().numpy() * 255.0     

for video in input.squeeze(1)[:3]:          # Loop over videos
    with io.BytesIO() as gif:
        imageio.mimsave(gif,video.astype(np.uint8),"GIF",fps=5)
        display(HBox([widgets.Image(value=gif.getvalue())]))

# TODO: replace above with matplotlib because notebooks suck
#fig, ax = plt.subplots()
#ax.axis('off')

#for video in input.squeeze(1)[:3]:  
#    # Loop over videos
#    for i in range (1,10):
#        im = plt.imshow(video[i, :, :])
#        plt.show()


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=1, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(64, 64), num_layers=3).to(device)

optim = Adam(model.parameters(), lr=learning_rate)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')

num_epochs = num_epochs

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


def collate_test(batch):

    # Last 10 frames are target
    target = np.array(batch)[:, 10:]           

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)          
    batch = batch / 255.0                             
    batch = batch.to(device)                          
    return batch, target


# Test Data Loader
test_loader = DataLoader(test_data, shuffle=True, batch_size=3, collate_fn=collate_test)

# Get a batch
batch, target = next(iter(test_loader))

# Initialize output sequence
output = np.zeros(target.shape, dtype=np.uint8)

# Loop over timesteps
for timestep in range(target.shape[1]):
    input = batch[:, :, timestep:timestep+10]   
    output[:, timestep] = (model(input).squeeze(1).cpu() > 0.5)*255.0


fig, (ax1, ax2) = plt.subplots(1, 2)

print("Target shape: ", target.shape)
print("Output shape: ", output.shape)

for i in range(10):
    im = ax1.imshow(target[1, i, :, :])
    im = ax2.imshow(output[1, i, :, :])
    plt.show()

for tgt, out in zip(target, output):       # Loop over samples
    
    # Write target video as gif
    with io.BytesIO() as gif:
        imageio.mimsave(gif, tgt, "GIF", fps=5)    
        target_gif = gif.getvalue()

    # Write output video as gif
    with io.BytesIO() as gif:
        imageio.mimsave(gif, out, "GIF", fps=5)    
        output_gif = gif.getvalue()

    display(HBox([widgets.Image(value=target_gif), 
                  widgets.Image(value=output_gif)]))
    
