# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:42:30 2019

@author: Johan
"""

import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(-np.pi,np.pi,100)
x_steg=900
t_steg=9000
dx=1/(x_steg+1)
dt=1/(t_steg+1)
beta = (1/dx)*theta
lambd=np.sqrt(4.905)
#lambd=0
s = lambd*dt/dx
y = np.ones(len(theta))

#Lax_Friedrich
xi=np.sqrt((1-s**2)*(np.cos(theta))**2 + s**2)
alpha = (1/(beta*dt)*np.arctan2(s*np.sin(theta),np.cos(theta)))
plt.figure(1)
plt.plot(theta,xi, label=r'$\xi$, dispersion' )
plt.plot(theta, alpha, label =r'$\alpha$, dissipation')
plt.plot(theta,y, label = 'y = 1', linewidth=0.5)
plt.legend()
plt.title('Lax-Friedrich')
plt.show()

#Mac-Cormack
xi_MC = np.sqrt((1-s**2 + s**2*np.cos(theta))**2 + (s*np.sin(theta))**2)
alpha_MC = (1/(beta*dt))*np.arctan2(s*np.sin(theta),(1-s**2+s**2*np.cos(theta)))
plt.figure(2)
plt.plot(theta, xi_MC, label=r'$\xi$, dispersion')
plt.plot(theta,alpha_MC, label=r'$\alpha$, dissipation')
plt.plot(theta,y, label = 'y = 1', linewidth=0.5)
plt.legend()
plt.title('MacCormack')
plt.show()