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

    return wave, ts, resX, resY, resZ


wave, ts, resX, resY, resZ = return_args()





