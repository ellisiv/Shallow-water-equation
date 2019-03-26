# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:56:18 2019

@author: helen
"""

import numpy as np
#import scipy.sparse
#import numpy.linalg as la
import matplotlib.pyplot as plt
#from scipy import sparse
#from scipy.sparse.linalg import spsolve
#from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
#from matplotlib import cm
from matplotlib import animation


def db_h_analytical(x, t, h0):
    # analytical solution of h for the dam break problem 
    c0 = np.sqrt(9.81 * h0)
    
    y = np.zeros(len(x))
    
    for i in range(len(x)):
        if x[i] < -2 * c0 * t:
            y[i] = 0
        if ((x[i] >= - 2 * c0 * t) and (x[i] <= c0 * t)):
            y[i] = 1 / (9 * g) * (x[i] / t + 2 * c0) ** 2 #fjernet g
        if x[i] > c0 * t:
            y[i] = h0
            
    return y
 
    
def db_u_analytical(x, t, h0):
    #analytical solution of u for the dam break problem
    c0 = np.sqrt(9.81 * h0)
    
    if x < -2 * c0 * t:
        return 0
    if ((x >= - 2 * c0 * t) and (x <= c0 * t)):
        return 2 / 3 * (x / t - c0)
    if x > c0 * t:
        return 0
    
def f2(x):               
        y = 2*x
        y[x>0.5] = 2-2*x[x>0.5]
        return y

def plot_analytical(h0):
    xlist = np.linspace(-1,1,100)
    tlist  = np.linspace(0.01,0.1,10)
    
    plt.figure()
    for t in tlist:
        plt.plot(xlist, db_h_analytical(xlist, t, h0), label = "t = {}".format(t))
    plt.legend()
    plt.show()

plot_analytical(2)
    