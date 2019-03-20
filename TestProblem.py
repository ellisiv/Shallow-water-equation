# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:23:53 2019

@author: Johan
"""
import numpy as np
import scipy.sparse
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
from matplotlib import animation


def h_nothing(x):
    return 1


def h_dupp(x):
    return 2 + 1.5*np.sin(np.pi * x*2)


def h_step(x, x0, xf):
    if x <= (x0 + xf) / 2:
        return 2
    else:
        return 0


def u_initial(x, x0, xf):
    if x>0.5:
        return -1
    else:
        return 1


def non_lin_Wendroff_mod(M, N, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0)/M
    dt = (tf - t0)/N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for m in range(M + 1):
        #h[m,0] = h_step(m*dx, x0, xf)
        #h[m, 0] = h_dupp(m * dx)
        h[m,0] = h_nothing(m*dx)
        v[m, 0] = u_initial(m * dx, x0, xf)
    print("Første gang \n",h[:, 0])
    va=1
    ha=1
    for n in range(N):
        for m in range(M + 1):
            #va = v[m, n] #disse kommer fra A-matrisen
            #ha = h[m, n] #Når de er inne i for-løkka er A ikke konstant. kan evt ta de ut og gjøre de konstante
            if m == 0:
                h[m, n + 1] = h[m, n] - 1 / 2 * dt / dx * (va * (h[m + 1, n] - h[m + 1, n]) + ha * (v[m + 1, n] + v[m + 1, n])) + 1 / 2 * (dt / dx) **2 * ((va ** 2 + ha * g) * (h[m + 1, n] - 2 * h[m, n] + h[m + 1, n]) + 2 * va * ha * (v[m + 1, n] - 2 * v[m, n] - v[m + 1, n]))
                v[m, n + 1] = v[m, n] - 1 / 2 * dt / dx * (g * (h[m + 1, n] - h[m + 1, n]) + va * (v[m + 1, n] + v[m + 1, n])) + 1 / 2 * (dt / dx) **2 * (2 * g * va * (h[m + 1, n] - 2 * h[m, n] + h[m + 1, n]) + (g * ha + va ** 2) * (v[m + 1, n] - 2 * v[m, n] - v[m + 1, n]))
            elif m == M:
                h[m, n + 1] = h[m, n] - 1 / 2 * dt / dx * (va * (h[m - 1, n] - h[m - 1, n]) + ha * (-v[m - 1, n] - v[m - 1, n])) + 1 / 2 * (dt / dx) **2 * ((va ** 2 + ha * g) * (h[m - 1, n] - 2 * h[m, n] + h[m - 1, n]) + 2 * va * ha * (- v[m - 1, n] - 2 * v[m, n] + v[m - 1, n]))
                v[m, n + 1] = v[m, n] - 1 / 2 * dt / dx * (g * (h[m - 1, n] - h[m - 1, n]) + va * (-v[m - 1, n] - v[m - 1, n])) + 1 / 2 * (dt / dx) **2 * (2 * g * va * (h[m - 1, n] - 2 * h[m, n] + h[m - 1, n]) + (g * ha + va ** 2) * (-v[m - 1, n] - 2 * v[m, n] + v[m - 1, n]))
            else:
                h[m, n + 1] = h[m, n] - 1 / 2 * dt / dx * (va * (h[m + 1, n] - h[m - 1, n]) + ha * (v[m + 1, n] - v[m - 1, n])) + 1 / 2 * (dt / dx) ** 2 * ((va ** 2 + ha * g) * (h[m + 1, n] - 2 * h[m, n] + h[m - 1, n]) + 2 * va * ha * (v[m + 1, n] - 2 * v[m, n] + v[m - 1, n]))
                v[m, n + 1] = v[m, n] - 1 / 2 * dt / dx * (g * (h[m + 1, n] - h[m-1, n]) + va * (v[m + 1, n] - v[m - 1, n])) + 1 / 2 * (dt / dx) ** 2 * (2 * g * va * (h[m + 1, n] - 2 * h[m, n] + h[m - 1, n]) + (g * ha + va ** 2) * (v[m + 1, n] - 2 * v[m, n] + v[m - 1, n]))
    
    return v, h

x_steg = 100
t_steg = 10000

v, h = non_lin_Wendroff_mod(x_steg, t_steg, tf=10)
U = np.copy(h)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-1, 5))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 1, x_steg + 1)
    line.set_data(x, U[:, i])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=15, blit=True)
plt.show()