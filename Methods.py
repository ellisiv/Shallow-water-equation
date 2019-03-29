import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

rcParams.update({'font.size': 12})

title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}# Bottom vertical alignment for more space



def F_shallow(Q):
    F = np.zeros((len(Q), 2))
    g = 0
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


def plot_h_fin(h, method, problem):
    i = 0
    x_len = len(h[1])
    x = np.linspace(0, 1, x_len)
    plt.figure()
    mult = 100
    while i <= 10:
        print(i)
        y = h[i * mult, :]
        plt.plot(x, y, label='t = {}'.format('{0:.3f}'.format(i * mult * 10 / t_steg)))
        plt.legend()
        i = i + 1
    plt.title('Approximation to exact solution with Lax-Friedrich on Dam break problem', **title_font)
    plt.xlabel('x')
    plt.ylabel('h')
    plt.show()


def Convergence_one_h(h_exact, H, tidspunkt, tf=10):
    h_exact = h_exact[:, :-1]
    H = H[:, :-1]

    l_exact = len(h_exact)
    l_H = len(H)

    dx_fin = 1 / (len(h_exact[1]))
    dx = 1 / (len(H[1]))
    dt_fin = 1 / len(h_exact)
    dt = 1 / len(H)

    #h_t_exact = h_exact[np.int(l_exact / tf * tidspunkt)]
    #H_t = H[np.int(l_H / tf * tidspunkt)]

    h_t_exact = h_exact[-1]
    H_t = H[-1]

    mult_x = np.int(dx / dx_fin)

    error = np.subtract(h_t_exact[::(mult_x)], H_t)

    plottmot = np.linspace(0, 10, len(H[1]))
    plt.plot(plottmot, H_t, label='num')
    plt.plot(plottmot, h_t_exact[::(mult_x)], label='"exact"')
    plt.legend()
    plt.show()
    return np.linalg.norm(error, np.inf)


def convergence_plot(h_exact, K, metode):
    r = 0.1
    tall_som_går_opp_i_900 = [45, 90, 100, 180, 300, 450]
    antx = tall_som_går_opp_i_900[0]
    e = np.zeros(K)
    xer = np.zeros(K)

    for k in range(K):
        print(k)
        antt = 1 / r * antx
        xer[k] = 1 / antx
        if metode == 'LF':
            H, v = Lax_Friedrich(initialize_Q(int(antx), int(antt), init='dam-break'), 10/antt, 1 / antx, int(antt), int(antx), boundary='inf')
            h = H[:-1, :]
        error = Convergence_one_h(h_exact, h, 0.08, 10)
        print(error)
        e[k] = np.abs(error)
        antx = tall_som_går_opp_i_900[k + 1]
    plt.loglog(xer, e)
    plt.show()







#x_steg = 80
#t_steg = 8000

#h, hv = macCormack(initialize_Q(x_steg, t_steg, init="sinus"), 10 / t_steg, 1 / x_steg, t_steg, x_steg, boundary='periodic')
#h, hv = Lax_Friedrich(initialize_Q(x_steg, t_steg, init='sinus'), 10/t_steg, 1/ x_steg, t_steg, x_steg, boundary='periodic')
#h, hv = Lax_Friedrich(initialize_Q(int(90), int(9000), init='dam-break'), 10/9000, 1 / 90, int(9000), int(90), boundary='inf')
#plot_h_fin(h, method="LF", problem='dam-break')


#np.save('height_LF.npy', h)
#np.save('hv_LF.npy', hv)

U = np.load('height_LF.npy')
u = U[:-1, :]

convergence_plot(u, 5, 'LF')

"""
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
                               frames=10000, interval=1, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
"""


