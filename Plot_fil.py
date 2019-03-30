import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

rcParams.update({'font.size': 12})
title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}# Bottom vertical alignment for more space

from Methods import F_shallow
from Methods import macCormack
from Methods import Lax_Friedrich
from Methods import initialize_Q

x_steg = 45
t_steg = 4500

def plot_h_fin(h, method, problem):
    i = 0
    x = np.linspace(0, 1, x_steg + 1)
    plt.figure()
    mult = 10
    while i <= 10:
        print(i)
        y = h[i * mult, :]
        plt.plot(x, y, label='t = {}'.format('{0:.3f}'.format(i * mult * 10 / t_steg)))
        plt.legend()
        i = i + 1
    plt.title('{}'.format(method) + ' on {}'.format(problem) +' problem', **title_font)
    plt.xlabel('x')
    plt.ylabel('h')
    plt.show()

def plot_h_finefarger(h, method, problem):
    i = 0
    x = np.linspace(0, 1, x_steg + 1)
    plt.figure()
    mult = 10

    plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
    
    while i <= 10:
        print(i)
        y = h[i * mult, :]
        plt.plot(x, y, label='t = {}'.format('{0:.3f}'.format(i * mult * 10 / t_steg)))
        plt.legend()
        i = i + 1
    plt.title('{}'.format(method) + ' on {}'.format(problem) +' problem', **title_font)
    plt.xlabel('x')
    plt.ylabel('h')
    plt.show()



#h, hv = macCormack(initialize_Q(x_steg, t_steg, init="dam-break"), 10 / t_steg, 1 / x_steg, t_steg, x_steg, boundary='inf')
#plot_h_fin(h, 'Dam break', 'MacCormack')

#h, hv = Lax_Friedrich(initialize_Q(x_steg, t_steg, init='dam-break'), 10/t_steg, 1/ x_steg, t_steg, x_steg, boundary='inf')#plot_h(h)
#plot_h_fin(h, 'Dam break', 'Lax-Friedrich')

#h, hv = macCormack(initialize_Q(x_steg, t_steg, init="sinus"), 10 / t_steg, 1 / x_steg, t_steg, x_steg, boundary='periodic')
#plot_h_fin(h, 'Still sine', 'MacCormack')

#h, hv = Lax_Friedrich(initialize_Q(x_steg, t_steg, init='sinus'), 10/t_steg, 1/ x_steg, t_steg, x_steg, boundary='periodic')#plot_h(h)
#plot_h_fin(h, 'Still sine', 'Lax-Friedrich')

#h, hv = macCormack(initialize_Q(x_steg, t_steg, init="sinus"), 10 / t_steg, 1 / x_steg, t_steg, x_steg, boundary='periodic')
#plot_h_fin(h, 'Sine in space, initial v = 1', 'MacCormack')

#h, hv = Lax_Friedrich(initialize_Q(x_steg, t_steg, init='sinus'), 10/t_steg, 1/ x_steg, t_steg, x_steg, boundary='periodic')#plot_h(h)
#plot_h_fin(h, 'Sine in space', 'Lax-Friedrich')

h, hv = Lax_Friedrich(initialize_Q(45, int(1 / 0.1 * 45), init='dam-break'), 10 * 0.1 / 45, 1 / 45, int(45 * 0.1) ,45, boundary='inf')#plot_h(h)
plot_h_fin(h, 'Dam break', 'Lax-Friedrich')

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
    line.set_data(x, U[i, :])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=15, blit=True)
plt.show()

#plot_h(h)