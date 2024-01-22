import pylab
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def rhs(t, y):
    a = 0.7
    c = 0.007
    b = 1
    p1 = y[0]
    p2 = y[1]

    dp1dt = a * p1 - c * p1 * p2
    dp2dt = c * p1 * p2 - b * p2

    return [dp1dt, dp2dt]


def rhs_odeint(y, t):
    a = 0.7
    c = 0.007
    b = 1
    # p1 = y[0]
    # p2 = y[1]
    p1, p2 = y

    dp1dt = a * p1 - c * p1 * p2
    dp2dt = c * p1 * p2 - b * p2

    return [dp1dt, dp2dt]


p0 = [70, 50]      # initial condition
t0 = 0
tfinal = 30
ts = np.linspace(t0, tfinal, 200)

# solve_ivp - Runge-Kutta method
sol = solve_ivp(rhs, [t0, tfinal], p0, t_eval=ts)
p1 = sol.y[0]
p2 = sol.y[1]

# older method, recommended to change for matplotlib.pyplot
# pylab.plot(sol.t, p1, label='rabbits')
# pylab.plot(sol.t, p2, '-og', label='foxes')
# pylab.legend()
# pylab.xlabel('t')
# pylab.savefig('predprey.pdf')
# pylab.savefig('predprey.png')


# equivalent, recommended to use matplotlib.pyplot instead of pylab
f1, ax1 = plt.subplots()
ax1.plot(sol.t, p1, label='rabbits')
ax1.plot(sol.t, p2, '-og', label='foxes')
ax1.legend()
ax1.set_xlabel('t')
ax1.set_title('Runge-Kutta method')
# plt.savefig('./export/predprey_runge.png')

# odeint - lsoda method
lsoda = odeint(rhs_odeint, p0, ts)
lsoda_1 = lsoda[:, 0]
lsoda_2 = lsoda[:, 1]

f2, ax2 = plt.subplots()
ax2.plot(ts, lsoda_1, label='rabbits')
ax2.plot(ts, lsoda_2, label='foxes')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_title('LSODA method')
# plt.savefig('./export/predprey_lsoda.png')

plt.show()
