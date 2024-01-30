import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pprint import pprint

# V4 - use dp/dt formula from SPB_Polytechnic, Donskoy

# Initial physical data
# Constants
P_atm = 101_325  # Pa
R = 286.69  # J/kg*K, for air, R = R_air / M_air
c_p = 1010  # J/kg*K, ???
c_v = c_p - R  # ???
x_t_valve = 0.5
gamma_air = 1.4
rho_0 = 1000  # kg/m3
delta_p_0 = 100_000  # Pa
pi = math.pi

# Variables
# General data
T_in = 20 + 273  # K
P_supply = 4.0 * 100_000  # Pa
P_in = P_supply + P_atm  # Pa
kv_total = 1 / 3600  # m^3/s
rho_in = P_in / (R * T_in)
# Calculated pneumatic valve values
delta_p_max = x_t_valve * gamma_air * P_in / 1.4  # Pa

# Actuator data
act_model = "VT240 S06"
torque_da = 1508  # N*m, @ P_in
torque_spring_start = 664.8  # N*m
torque_spring_end = 492.6  # N*m
total_volume = 11.40 / 1000  # m3
act_weight = 77.76  # kg

# Valve data
valve_torque_bto = 330  # N*m

# Advanced data - calculated actuator values
alfa = 2.89  # invariant of rack & pinion piston actuator, alfa = piston_diameter / lever_length
piston_diameter = math.pow((4 * alfa * torque_da / pi / P_supply), 1/3)  # m
piston_surface = pi * math.pow(piston_diameter, 2) / 4  # m2
lever_length = torque_da / P_supply / piston_surface
dead_volume = total_volume * 0.1
full_stroke = (total_volume - dead_volume) / piston_surface
spring_force_relaxed = torque_spring_end / full_stroke
spring_force_comp_factor = (torque_spring_start - torque_spring_end) / full_stroke / lever_length
friction_force = spring_force_relaxed * 0.2
M = act_weight * 0.3
torque_air_start = torque_da - torque_spring_end  # N*m
torque_air_end = torque_da - torque_spring_start  # N*m

print(f"""
    {piston_diameter=},
    {piston_surface=},
    {lever_length=},
    {dead_volume=},
    {full_stroke=},
    {spring_force_relaxed=},
    {spring_force_comp_factor=},
    {friction_force=},
    {M=},
    {torque_air_start=},
    {torque_air_end=};
""")

# Calculated values
# piston_diameter = piston_diameter
# V_extra = dead_volume # m3
x_01 = dead_volume / piston_surface
xf = full_stroke  # m
S = piston_surface
k = gamma_air


def get_relative_valve_torque(rel_travel):
    x_r = rel_travel
    f_x = 1 - 3.83*x_r + 13.65*x_r**2 - 28.39*x_r**3 + 33.51*x_r**4 - 19.53*x_r**5 + 4.34*x_r**6
    return f_x


def ds_dt(t, y):
    x, v, p, f = y

    p_bar = p / 100_000
    # print(f"{x=}, {v=}, {p_bar=}")

    relative_travel = x / xf
    valve_torque_factor = get_relative_valve_torque(relative_travel)
    valve_current_torque = valve_torque_bto * valve_torque_factor
    # print(f"{relative_travel=}, {valve_current_torque=}")
    F_valve = valve_current_torque / lever_length

    F_static_friction = friction_force + F_valve
    F_spring = spring_force_relaxed + spring_force_comp_factor * x
    # F_pressure = (p - P_atm) * S
    F_pressure = f
    F_pressure__spring = F_pressure - F_spring
    F_diff = abs(F_pressure__spring) - F_static_friction
    if F_diff > 0:
        F_result = math.copysign(1, F_pressure__spring) * F_diff
    else:
        F_result = 0
    # print(f"{F_result=}, {F_pressure=}, {F_spring=}, {F_static_friction=}")
    # forces[0].append(F_result)
    # forces[1].append(F_pressure)
    # forces[2].append(F_spring)
    # forces[3].append(F_static_friction)

    if x < 0:
        x = 0
        v = max(0, v)

    if x >= xf:
        x = xf
        dxdt = 0
        v = 0
        dvdt = 0
    else:
        dxdt = v
        dvdt = F_result / M

    dpdt = math.copysign(1, (P_in - p)) * kv_total * k * math.sqrt(
        R*T_in*0.005*(abs(P_in**2 - p**2))
    ) / (S * (x + x_01)) - k * p * v / (x + x_01)

    dfdt = dpdt * S

    return [dxdt, dvdt, dpdt, dfdt]


def work_hit_max(t, y, *args):
    # x, v, p = y
    # if (int(x) - xf) >= 0:
    #     return 0
    return (y[0] - xf) + (y[2] - P_in)


work_hit_max.terminal = True

# Initial ODE data
x0 = 0
v0 = 0
p0 = P_atm
f0 = 0

y0 = [x0, v0, p0, f0]
t0 = 0
t_final = 15
ts = np.linspace(t0, t_final, 10000)
events = [work_hit_max, ]
forces = [[], [], [], []]

sol = solve_ivp(ds_dt, [t0, t_final], y0, method='BDF', t_eval=ts, events=events)

# RK23 - not bad
# Radau - Runge-Kutta error controlled - good result
# BDF - seems to be a bit slower than Radau
# LSODA - good result

x1 = sol.y[0]
v1 = sol.y[1]
p1 = [x/100_000 for x in sol.y[2]]
f1 = sol.y[3]

time_to_fill = sol.t[-1]
print("Time to fill", time_to_fill)

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(sol.t, x1, label="Travel")
ax1.plot(sol.t, v1, label="Velocity")
ax1.legend()

ax2.plot(sol.t, p1, label="Pressure")
ax2.legend()


# f_len = len(forces[0])
# tf = np.linspace(t0, time_to_fill, f_len)
# ax3.plot(tf, forces[0], label="F_result")
# ax3.plot(tf, forces[1], label="F_pressure")
# ax3.plot(tf, forces[2], label="F_spring")
# ax3.plot(tf, forces[3], label="F_static_friction")
# ax3.legend()
#
# ax4.plot(sol.t, f1, label="Pressure Force")
#
# plt.show()

print(type(f1))

f_result = np.array(f1)
f_spring = np.array(f1)
f_static_friction = np.array(f1)

for i, f in enumerate(f1):
    x = x1[i]
    relative_travel = x / xf
    valve_torque_factor = get_relative_valve_torque(relative_travel)
    valve_current_torque = valve_torque_bto * valve_torque_factor
    F_valve = valve_current_torque / lever_length

    F_static_friction = friction_force + F_valve
    f_static_friction[i] = F_static_friction
    F_spring = spring_force_relaxed + spring_force_comp_factor * x
    f_spring[i] = F_spring
    F_pressure = f
    F_pressure__spring = F_pressure - F_spring
    F_diff = abs(F_pressure__spring) - F_static_friction
    if F_diff > 0:
        F_result = math.copysign(1, F_pressure__spring) * F_diff
    else:
        F_result = 0
    f_result[i] = F_result

ax3.plot(sol.t, f_result, label="F_result")
ax3.plot(sol.t, f1, label="F_pressure")
ax3.plot(sol.t, f_spring, label="F_spring")
ax3.plot(sol.t, f_static_friction, label="F_static_friction")
ax3.legend()

plt.show()
