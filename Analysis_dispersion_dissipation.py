# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:42:30 2019

@author: Johan
"""

import numpy as np
import matplotlib.pyplot as plt


#Lax-Friedrich

theta = np.linspace(-np.pi,np.pi,200)
x_steg=100
t_steg=10000
dx=1/(x_steg+1)
dt=1/(t_steg+1)
beta = (1/dx)*theta
lambd=np.sqrt(4.905)
xi=np.sqrt((dt**2-2*lambd*(dt/dx))*np.cos(theta) + (2*lambd/dx)**2)
#alpha = (1/(beta*dt)*np.arctan((((dt/2)-(2*lambd/dx))*np.cos(beta*dx))/((2*lambd/dx)*np.sin(beta*dx))))
plt.plot(theta,xi)
#plt.plot(theta, alpha)
plt.show()
