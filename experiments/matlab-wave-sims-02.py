#!/usr/local/bin/python3

import numpy
from matplotlib import pyplot
from matplotlib import cm

Lx = 9 # total width of the pool
Nx = 9 # amount of points in the x direction, the more the better
Ly = 9 # total height of the pool
Ny = 9 # amount of points in the y direction, the more the better

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
Nt = 1000 # amount of iterations

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

cross_idx = int(Nt/2)
print("Cross section: ", cross_idx)
print(u.shape)
print(u[50][5])
print(numpy.max(u[cross_idx]))
print(numpy.min(u[cross_idx]))

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

mX = 9
mY = 9
# ok now convert the entire run
final_wave = numpy.zeros((Nt, mX, mY))
for i in range(Nt):
    for j in range(Lx):
        final_wave[i, :u_scaled_int[j], j] = 1

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

for i in range(0, Nt, 20):
    print(final_wave[i])