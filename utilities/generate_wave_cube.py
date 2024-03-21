#!/usr/local/bin/python3

# generate a cube of numbers representing a wave snapshot
# inputs: 
#
# --wave : which IRL wave to model
# --ts   : how many timesteps to model
# --resX
# --resY
# --resZ : resolution for vertical (Y), front-to-back (Z), wavefront (X)
#

# read arguments

import sys, argparse
import numpy as np

def return_args():
        
    parser = argparse.ArgumentParser(description="default arg parser")

    parser.add_argument("--wave", type=str, help="Accepts a string")
    parser.add_argument("--ts", type=int, help="Accepts an int")
    parser.add_argument("--resX", type=int, help="Accepts an int")
    parser.add_argument("--resY", type=int, help="Accepts an int")
    parser.add_argument("--resZ", type=int, help="Accepts an int")

    args = parser.parse_args()

    wave = args.wave
    ts = args.ts
    resX = args.resX
    resY = args.resY
    resZ = args.resZ

    print ("Wave      : ", wave)
    print ("Timesteps : ", ts)
    print ("X res     : ", resX)
    print ("Y res     : ", resY)
    print ("Z res     : ", resZ)
    print ("")
    print ("**********************************************")

    return wave, ts, resX, resY, resZ


def create_chopes_at_timestep(i):
    print ("Chopes timestep: ", i)


def create_pipe_at_timestep(i):
    print ("Pipe timestep: ", i)


# Special wave - just a constant, straight line, intended for testing/training
def create_constant_at_timestep(i):
    print ("Constant wave at: ", i)


def create_wave_timestep(timestep, wave):

    if wave == "chopes":
        create_chopes_at_timestep(timestep)
    if wave == "pipe":
        create_pipe_at_timestep(timestep)
    if wave == "constant":
        create_constant_at_timestep(timestep)


# pull instructions from command line
wave, ts, resX, resY, resZ = return_args()

# create the wave container
this_wave = np.zeros((resX, resY, resZ))

for i in range(ts):
    create_wave_timestep(i, wave)




