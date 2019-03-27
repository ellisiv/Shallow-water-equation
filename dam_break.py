import numpy as np
# import scipy.sparse
# import numpy.linalg as la
import matplotlib.pyplot as plt
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
# from matplotlib import cm
from matplotlib import animation

def u_initial(x, x0, xf):
    return 2

def h_step(x, x0, xf):
    if x <= (x0 + xf) / 2:
        return 2
    else:
        return 0

def h_step_flip(x, x0, xf, h0, h1):
    if x <= (x0 + xf) / 2:
        return h0
    else:
        return h1

def h_sin(x, x0, xf):
    return 2 * np.sin(x * np.pi / 2 - np.pi / 2) + 3

def h_dupp(x, x0, xf):
    return 2 + 1.5 * np.sin(np.pi * x * 2)

def non_lin_LF_reflect(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_step_flip(k * dx, x0, xf, h0, h1)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m == 0:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m + 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m + 1, n] * v[m + 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] - v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx
            elif m == M:
                h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m - 1, n] * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (- v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h


def non_lin_LF_const_cond(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_step_flip(k * dx, x0, xf, h0, h1)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m == 0:
                #h[m, n + 1]  = h0
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h0) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h0 * v[m + 1, n])) * dt / dx
                # reflective h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m + 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m + 1, n] * v[m + 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h0 + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx
                # reflective v[m, n + 1] = 1 / 2 * (v[m + 1, n] - v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx

            elif m == M:
                #h[m, n + 1] = h1
                h[m, n + 1] = 1 / 2 * (h1 + h[m - 1, n]) - 1 / 2 * ((h1 * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                # reflective h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m - 1, n] * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h1 + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
                # reflective v[m, n + 1] = 1 / 2 * (- v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx

            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h


def non_lin_lf_reflective_rettet(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_step_flip(k * dx, x0, xf, h0, h1)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m == 0:
               #h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m + 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) + (h[m + 1, n] * v[m + 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] - v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx

            elif m == M:
                h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m - 1, n] * (-v[m - 1, n])) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                #h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * (-(h[m - 1, n] * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx

                v[m, n + 1] = 1 / 2 * (-v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * (-v[m - 1, n]) ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
                #v[m, n + 1] = 1 / 2 * (- v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx

            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h

def non_lin_LF_infinite(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_step_flip(k * dx, x0, xf, h0, h1)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m > M:
                print("uffda")
            if m == 0:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m + 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m + 1, n] * v[m + 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx
            elif m == M:
                h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m - 1, n] * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h

def non_lin_LF_reservoir(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_step_flip(k * dx, x0, xf, h0, h1)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m > M:
                print("uffda")
            if m == 0:
                h[m, n + 1] = h0
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m + 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h0 + 1 / 2 * v[m + 1, n] ** 2)) * dt / dx
            elif m == M:
                h[m, n + 1] = 1 / 2 * (h[m - 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m - 1, n] * v[m - 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m - 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h

def non_lin_LF_periodic(M, N, h0, h1, x0=0, xf=1, t0=0, tf=1):
    g = 0#9.81
    dx = (xf - x0) / M
    dt = (tf - t0) / N
    h = np.zeros((M + 1, N + 1))
    v = np.zeros((M + 1, N + 1))

    for k in range(M + 1):
        h[k, 0] = h_dupp(k * dx, x0, xf)
        v[k, 0] = u_initial(k * dx, x0, xf)

    for n in range(N):
        for m in range(M + 1):
            if m > M:
                print("uffda")
            if m == 0:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[M - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[M - 1, n] * v[M - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[M - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[M - 1, n] + 1 / 2 * v[M - 1, n] ** 2)) * dt / dx
            elif m == M:
                h[m, n + 1] = 1 / 2 * (h[1, n] + h[m - 1, n]) - 1 / 2 * ((h[1, n] * v[1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[1, n] + 1 / 2 * v[1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
            else:
                h[m, n + 1] = 1 / 2 * (h[m + 1, n] + h[m - 1, n]) - 1 / 2 * ((h[m + 1, n] * v[m + 1, n]) - (h[m - 1, n] * v[m - 1, n])) * dt / dx
                v[m, n + 1] = 1 / 2 * (v[m + 1, n] + v[m - 1, n]) - 1 / 2 * ((g * h[m + 1, n] + 1 / 2 * v[m + 1, n] ** 2) - (g * h[m - 1, n] + 1 / 2 * v[m - 1, n] ** 2)) * dt / dx
    return v, h



x_steg = 100
t_steg = 10000

#v, h = non_lin_LF_const_cond(x_steg, t_steg, 0, 1, tf=10)
#v, h = non_lin_LF_reflect(x_steg, t_steg, 1, 0, tf=10)
#v, h = non_lin_lf_reflective_rettet(x_steg, t_steg, 0, 1, tf=10)
#v, h = non_lin_LF_infinite(x_steg, t_steg, 0, 1, tf=10)
v, h = non_lin_LF_reservoir(x_steg, t_steg, 2, 1, tf=10)
#v, h = non_lin_LF_periodic(x_steg, t_steg, 1, 0, tf=10)
U = np.copy(h)


# plot hver 1000. rad i h-matrisen mot x:
def plot_h(h):
    i = 0
    # i=10
    x = np.linspace(0, 1, x_steg + 1)
    print('ant ganger gjennom for-løkka')
    plt.figure()
    while i <= 200:
        print(i)
        y = h[:, i]
        plt.plot(x, y, label=i)
        plt.legend()
        i = i + 20
    # plt.xlim(0.48,0.52)
    plt.title('Numerical height')
    plt.xlabel('x')
    plt.ylabel('h')
    plt.show()


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
                               frames=1000, interval=10, blit=True)
plt.show()

plot_h(h)
