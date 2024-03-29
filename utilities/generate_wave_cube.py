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

# default inputs for wave construction
default_x = 64
default_y = 64
default_z = 64
default_ts = default_x

def return_args():
        
    parser = argparse.ArgumentParser(description="default arg parser")

    parser.add_argument("--wave", type=str, help="Accepts a string")
    parser.add_argument("--ts", type=int, help="Accepts an int")
    parser.add_argument("--resX", type=int, help="Accepts an int")
    parser.add_argument("--resY", type=int, help="Accepts an int")
    parser.add_argument("--resZ", type=int, help="Accepts an int")

    args = parser.parse_args()

    wave = args.wave

    if args.resX:
        resX = args.resX
    else:
        resX = default_x
    if args.resY:
        resY = args.resY
    else:
        resY = default_y
    # if no Z dimension given, assume it is 2D
    if args.resZ:
        resZ = args.resZ
    else:
        resZ = 0
    if args.ts:
        ts = args.ts
    else:
        ts = default_ts

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


def create_jaws_at_timestep(i):
    print ("Pipe timestep: ", i)


def create_nazare_at_timestep(i):
    print ("Pipe timestep: ", i)

# Special wave - just a constant, straight line, intended for testing/training
def create_constant_at_timestep(i):

    print ("Constant wave at: ", i)
    this_ts = np.zeros((resX, resY, resZ))

    # let's just make a half-cube inside the cube
    for x in range(int(resX/4), int(3*resX/4)):
        for y in range(int(resX/4), int(3*resX/4)):
            for z in range(int(resX/4), int(3*resX/4)):
                this_ts[x,y,z] = 1

    print(this_ts)
    return this_ts


def create_wave_timestep(timestep, wave):

    if wave == "chopes":
        create_chopes_at_timestep(timestep)
    if wave == "pipe":
        create_pipe_at_timestep(timestep)
    if wave == "nazare":
        create_nazare_at_timestep(timestep)
    if wave == "jaws":
        create_jaws_at_timestep(timestep)
    if wave == "constant":
        create_constant_at_timestep(timestep)


# pull instructions from command line
wave, ts, resX, resY, resZ = return_args()

# create the wave container
this_wave = np.array([np.zeros((resX, resY, resZ)) for _ in range(ts)])

# generate the timesteps
for i in range(ts):
    this_wave[i] = create_wave_timestep(i, wave)

# tuck it somewhere safe
np.save("generated_wave.npy", this_wave)
# load it with something like...
# loaded_array_3d = np.load("array_3d.npy")




