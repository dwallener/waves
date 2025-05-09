#!/usr/local/bin/python3

import numpy
from matplotlib import pyplot
from matplotlib import cm

np_save_file_1000 = "wave-disturbance-01-1000ts-64px.npy"
np_save_file_10000 = "wave-disturbance-01-10000ts-64px.npy"
np_save_file_100000 = "wave-disturbance-01-100000ts-64px.npy"

Lx = 10 # total width of the pool
Nx = 64 # amount of points in the x direction, the more the better
Ly = 10 # total height of the pool
Ny = 64 # amount of points in the y direction, the more the better

# meshes the x dimension of the domain as being from 0 to Lx and
# containing Nx points. The linspace function returns an array of
# all the points
x_vec = numpy.linspace(0, Lx, Nx)
dx = x_vec[2] - x_vec[1] # defines dx as the space between 2 points in x

# meshes the y dimension of the domain as being from 0 to Ly and
# containing Ny points. The linspace function returns an array of
# all the points
y_vec = numpy.linspace(0, Ly, Ny)
dy = y_vec[2] - y_vec[1] # defines dy as the space between 2 points in y

dt = .025 # the amount of time that will pass after every iteration
dt = .025 # double the time frequency
Nt = 100000 # amount of iterations

print ("Timesteps : ", Nt)
if Nt == 1000:
    np_save_file = np_save_file_1000
if Nt == 10000:
    np_save_file = np_save_file_10000
if Nt == 100000:
    np_save_file = np_save_file_100000

# this means that the simulation will simulate dt*Nt real seconds of water rippling

c = 1 # keeping it simple

# defines a 2 dimensional array that corresponds to the value of u at
# every point in the mesh
u = numpy.zeros([Nt, len(x_vec), len(y_vec)])

u[0, Nx // 2, Ny // 2] = numpy.sin(0) # disturbance at t = 0
u[1, Nx // 2, Ny // 2] = numpy.sin(1/10) # disturbance at t = 1

print("Inputs set")

for t in range(1, Nt-1):
    #print(t/Nt)
    for x in range(1, Nx-1):
        for y in range(1, Ny-1):
            if (t < 100):
                u[t, Nx // 2, Ny // 2] = numpy.sin(t / 10)

            u[t+1, x, y] = c**2 * dt**2 * ( ((u[t, x+1, y] - 2*u[t, x, y] + u[t, x-1, y])/(dx**2)) + ((u[t, x, y+1] - 2*u[t, x, y] + u[t, x, y-1])/(dy**2)) ) + 2*u[t, x, y] - u[t-1, x, y]

print("Big for loop done...plotting")

# single timestep validation

cross_idx = int(Nt/4)
print("Cross section @ts: ", cross_idx)
print(u.shape)
print(u[cross_idx][int(Lx/2)])
print(numpy.max(u[cross_idx]))

# scale to a more reasonable range, like 0..32 (we're moving to a 64x64 image)
u_scaled = numpy.interp(u[cross_idx][int(Nx/2)], ((-1, 1)), (0, Lx))
print("Scaled")
print(u_scaled)

# convert to int
u_scaled_int = u_scaled.astype(int)
print(u_scaled_int)

# convert to wave profile slice
wave_profile = numpy.zeros((Lx, Nx))
for i in range(Lx):
    wave_profile[:u_scaled_int[i], i] = 1
wave_profile = numpy.flipud(wave_profile)
print("Wave profile")
print(wave_profile)

mX = Nx
mY = Ny

# i'm using a slice, not the entire u! TODO: fix this!
# ok now convert the entire run
final_wave = numpy.zeros((Nt, mX, mY))

for i in range(Nt):
    for j in range(Nx):
        u_scaled_int = numpy.interp(u[i][int(Nx/2)], ((-1, 1)), (0, Nx)).astype(int)
        final_wave[i,:u_scaled_int[j], j] = 1

numpy.save(np_save_file, final_wave)

final_wave = numpy.fliplr(final_wave)
for i in range(359, 469, 11):
    print("Timestep: ", i)
    print(final_wave[i])

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = numpy.meshgrid(x_vec, y_vec)
for t in range(0, Nt):
    surf = ax.plot_surface(X, Y, u[t], color='b', shade=True,
                           linewidth=0, antialiased=False)

    ax.view_init(elev=45)
    ax.set_zlim(-.0001, 2.4)
    pyplot.axis('off')

    pyplot.pause(.0001)
    pyplot.cla()

