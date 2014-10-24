"""
Created on Fri Oct 24 16:41:16 2014
@author: f.groestlinger
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


w0 = 800
dw = 20
w_oo = 100
N_w = 100
N_t = 1000
T = 1.0
w = np.linspace(w0 - w_oo, w0 + w_oo, N_w)
N_A = lambda w: 1.0 / (np.sqrt(2 * np.pi) * dw) * np.exp(- (w - w0) / (2 * dw ** 2))
t = np.linspace(0, T, N_t)
fw = lambda p: np.array([N_A(x) * np.exp(- 1j * x * (t + p)) for x in w])
I = lambda E: np.abs(E) ** 2 
I_p = np.array([np.max(I(np.sum(fw(0.0), 0) + np.sum(fw(p), 0))) for p in np.linspace(-10, 10, 1000)])
plt.clf()
plt.plot(I_p)