import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def F_shallow(Q):
    F = np.zeros((len(Q), 2))
    g = 9.81
    for i in range(len(F)):
        F[i][0] = Q[i][1]
        if Q[i][0] == 0:
            F[i][1] = 0
        else:
            F[i][1] = Q[i][1] ** 2 / Q[i][0] + 1/2 * g * Q[i][0] ** 2
    return F


def macCormack(Q, dt, dx, N, M, boundary):
    for n in range(N):
        q_p = np.zeros((len(Q[n]) + 1, 2))
        q_n = np.zeros((len(Q[n]) + 2, 2))
        q_n[1:-1] = Q[n]
        if boundary == "periodic":
            q_n[0] = q_n[-3]
            q_n[-1] = q_n[2]
        if boundary == "inf":
            q_n[0][0] = q_n[2][0]
            q_n[0][1] = q_n[2][1]
            q_n[-1][0] = 2#q_n[-3][0]
            q_n[-1][1] = q_n[-3][1]
        if boundary == "reflective":
            q_n[0][0] = q_n[2][0]
            q_n[0][1] = - q_n[2][1]
            q_n[-1][0] = q_n[-3][0]
            q_n[-1][1] = - q_n[-3][1]
        q_p = q_n[:-1] - (dt / dx) * (F_shallow(q_n[1:]) - F_shallow(q_n[:-1]))
        Q[n + 1, :] = 1 / 2 * (Q[n, :] + q_p[1:]) + dt / (2 * dx) * (F_shallow(q_p[:-1]) - F_shallow(q_p[1:]))
    return Q[:, :, 0], Q[:, :, 1]


def Lax_Friedrich(Q, dt, dx, N, M, boundary):
    for n in range(N):
        q_n = np.zeros((len(Q[n]) + 2, 2))
        q_n[1:-1] = Q[n]
        if boundary == "periodic":
            q_n[0] = q_n[-3]
            q_n[-1] = q_n[2]
        if boundary == "inf":
            q_n[0][0] = q_n[2][0]
            q_n[0][1] = q_n[2][1]
            q_n[-1][0] = 2#q_n[-3][0]
            q_n[-1][1] = q_n[-3][1]
        if boundary == "reflective":
            q_n[0][0] = q_n[2][0]
            q_n[0][1] = - q_n[2][1]
            q_n[-1][0] = q_n[-3][0]
            q_n[-1][1] = - q_n[-3][1]
        Q[n + 1] = 1 / 2 * (q_n[2:] + q_n[:-2]) - dt / (2 * dx) * (F_shallow(q_n[2:]) - F_shallow(q_n[:-2]))
    return Q[:, :, 0], Q[:, :, 1]


def initialize_Q(M, N, init):
    Q = np.zeros((N + 1, M + 1, 2))
    if init == "dam-break":
        for m in range(M + 1):
            if m < 3 * M/6:
                Q[0][m][0] = 1
                Q[0][m][1] = 0
            else:
                Q[0][m][0] = 2
                Q[0][m][1] = 0
    if init == "sinus":
        for m in range(M + 1):
            Q[0][m][0] = 2 * np.sin(np.pi * 2 * m * 1/M) + 3
            Q[0][m][1] = Q[0][m][0] * 0.8
    if init == 'flood':
        for m in range(M + 1):
            if m < 4*M/10 or m > 6 * M/10:
                Q[0][m][0] = 2
            else:
                Q[0][m][0] = 10 * (m / M) - 4
    if init == 'stairs':
        for m in range(M + 1):
            if m < M/4 or m > 3 * M/4:
                Q[0][m][0] = 3
            elif m < m/2:
                Q[0][m][0] = 5
            else:
                Q[0][m][0] = 2
    return Q


def plot_h(h):
    i = 0
    # i=10
    x = np.linspace(0, 1, x_steg + 1)
    print('ant ganger gjennom for-løkka')
    plt.figure()
    while i <= 11:
        print(i)
        y = h[i * 1, :]
        plt.plot(x, y, label=i)
        plt.legend()
        i = i + 1
    # plt.xlim(0.48,0.52)
    plt.title('Numerical height')
    plt.xlabel('x')
    plt.ylabel('h')


x_steg = 100
t_steg = 10000

#v, h = non_lin_LF(x_steg, t_steg, tf=10)   #non_lin_LF(M, N, x0=0, xf=1, t0=0, tf=1):
#v, h = non_lin_Wendroff(x_steg, t_steg, tf=10)  #non_lin_Wendroff(M, N, x0=0, xf=1, t0=0, tf=1)

#Denne funker hittil best
#h, hv = macCormack(initialize_Q(x_steg, t_steg, init="dam-break"), 10 / t_steg, 1 / x_steg, t_steg, x_steg, boundary='inf')
#h, hv = Lax_Friedrich(initialize_Q(x_steg, t_steg, init='dam-break'), 10/t_steg, 1/ x_steg, t_steg, x_steg, boundary='inf')
#plot_h(h)

#v, h = non_lin_Wendroff_mod2(x_steg, t_steg, tf=5)
#U = np.copy(h)

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
    line.set_data(x, U[i, :])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=50, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()


