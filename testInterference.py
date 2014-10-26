"""
Created on Fri Oct 24 16:41:16 2014
@author: f.groestlinger
"""
import numpy as np
import matplotlib.pyplot as plt


c = 3E8 # m s-1
l0 = 8E-7 # m
dl = 2E-8 #m
num_v = 100
num_t = 10000
num_dt = 100
T = 1E-12 # s
dt = 1E-13 # s
k_1 = 0.5
k_2 = 0.5

v0 = c / l0 # s-1
print('v0 = {0} s-1'.format(v0))
dv = c / l0 ** 2 * dl
print('dv = {0} s-1'.format(dv))
v_oo = 5 * dv

plt.clf()

N_v = lambda v: 1.0 / (np.sqrt(2 * np.pi) * dv) * np.exp(- (v - v0) ** 2 / (2 * dv ** 2))
vs = np.linspace(v0 - v_oo, v0 + v_oo, num_v)
#plt.plot(vs, N_v(vs), 'k-')

V = lambda x, t, v, V0: V0 * np.exp(- 2 * np.pi * 1j * (v /c * x - v * t))
#x = np.linspace(0, 1E-5, 1000)
#V_test = V(x, 0.0, v0, 1.0)
#plt.plot(x, V_test.real, 'b-')
#plt.plot(x, V_test.imag, 'g-')

ts = np.linspace(0.0, T, num_t)
#plt.plot(ts, V(0.0, ts, v0, 1.0).real, 'b-')
#plt.plot(ts, V(0.0, ts, v0, 1.0).imag, 'g-')
V_v = lambda dt: np.array([V(0.0, ts - dt, v, N_v(v)) for v in vs])
#plt.plot(ts, V(0.0, ts, v0, N_v(v0 - v_oo)).real)
#for x in V_v(0.0) + V_v(dt):
#    plt.plot(x)

dts = np.linspace(0.0, dt, num_dt)

V_v_dt = lambda dt: np.sum(k_1 * V_v(0.0) + k_2 * V_v(dt), 0)
#for i, y in enumerate(dts):
#    z = V_v_dt(y)
#    plt.subplot(len(dts) + 1, 2, (2 * i) + 2)
#    plt.plot(np.sum(z, 0).conjugate() * np.sum(z, 0))
#    for x in z:
#        plt.subplot(len(dts) + 1, 2, (2 * i) + 1)
#        plt.plot(x)
#

I_dt = lambda dt: V_v_dt(dt).conjugate() * V_v_dt(dt)
I = np.array([np.sum(I_dt(t)) for t in dts])
plt.plot(I)

plt.show()