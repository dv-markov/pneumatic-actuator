import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pprint import pprint
from flowcalc import kv_dzeta
from initial_data import *

# V6 - use dp/dt formula from SPB_Polytechnic, Donskoy
# add p2 and p2_force
# move initial data to separate module

# Kv_data and piping data
kv_in_1 = 0.32 / 3600  # m^3/s
d_pipe = 5 / 1000  # m
le_pipe = 100 / 1000  # m

# Kv_calculation
kv_pipe = kv_dzeta.kv_calculated_for_pipe(d_pipe, le_pipe)
print(f'Kv_pipe = {kv_pipe * 3600:.2f} m3/h')
kv_total_1 = kv_dzeta.kv_addition(kv_in_1, kv_pipe)  # m^3/s
kv_total_2 = kv_pipe  # m^3/s
print(f"""Kv_total_1 = {kv_total_1 * 3600:.2f} m3/h
Kv_total_2 = {kv_total_2 * 3600:.2f} m3/h""")


def get_relative_valve_torque(rel_travel):
    x_r = rel_travel
    f_x = 1 - 3.83*x_r + 13.65*x_r**2 - 28.39*x_r**3 + 33.51*x_r**4 - 19.53*x_r**5 + 4.34*x_r**6
    return f_x


def get_f_result(x, pc_1, pc_2, plot=False, force_dict=None, flow_dict=None, ind=None):
    relative_travel = x / xf
    valve_torque_factor = get_relative_valve_torque(relative_travel)
    valve_current_torque = valve_torque_bto * valve_torque_factor
    F_valve = valve_current_torque / lever_length

    F_static_friction = friction_force + F_valve
    F_spring = spring_force_relaxed + spring_force_comp_factor * x
    F_pressure_1 = (pc_1 - P_atm) * S
    F_pressure_2 = (pc_2 - P_atm) * S
    F_pressure__spring = F_pressure_1 - F_spring - F_pressure_2
    F_diff = abs(F_pressure__spring) - F_static_friction
    if F_diff > 0:
        F_result = math.copysign(1, F_pressure__spring) * F_diff
    else:
        F_result = 0

    if plot:
        force_dict['f_static_friction'][ind] = F_static_friction
        force_dict['f_spring'][ind] = F_spring
        force_dict['f_pressure'][ind] = F_pressure_1 - F_pressure_2
        force_dict['f_result'][ind] = F_result

        delta_p = abs(P_in - pc_1)
        x_factor_valve = delta_p / P_in
        expansion_factor = max(2 / 3, (1 - (1 / 3) * (1.4 / gamma_air) * (x_factor_valve / x_t_valve)))
        delta_p_calc = min(delta_p, delta_p_max)

        qv = kv_dzeta.qv_calculated(kv_total_1, rho_in, delta_p_calc) * expansion_factor
        flow_dict['qv'][i] = qv
        qm = kv_dzeta.qm_calculated(qv, rho_in)
        flow_dict['qm'][i] = qm
        gm = kv_dzeta.gm_calculated(kv_total_1, P_in, pc_1, T_in)
        gm_array[i] = gm

    return F_result


def ds_dt(t, y):
    x, v, pc_1, pc_2 = y

    F_result = get_f_result(x, pc_1, pc_2)

    if x < 0:
        x = 0
        v = max(0, v)

    if x >= xf:
        x = xf
        dxdt = 0
        if v > 0:
            dvdt = -100*v
        else:
            dvdt = 0
        v = 0
    else:
        dxdt = v
        dvdt = F_result / M

    dpc_1dt = math.copysign(1, (P_in - pc_1)) * kv_total_1 * k * math.sqrt(
        R*T_in*0.005*(abs(P_in**2 - pc_1**2))
    ) / (S * (x + x_01)) - k * pc_1 * v / (x + x_01)

    dpc_2dt = (-1) * math.copysign(1, (pc_2 - P_atm)) * kv_total_2 * k * math.sqrt(
        R*T_in*0.005*(abs(pc_2**2 - P_atm**2))
    ) * math.pow(
        abs(pc_2) / P_in,
        (k - 1) / (2 * k)
    ) / (S * (xf - x + x_02)) + k * pc_2 * v / (xf - x + x_02)

    return [dxdt, dvdt, dpc_1dt, dpc_2dt]


def work_hit_max(t, y, *args):
    return (y[0] - xf) + (y[2] - P_in)


def piston_reach_max_travel(t, y):
    return xf - y[0]


def p2_equals_atm(t, y, *args):
    return y[3] - P_in


work_hit_max.terminal = True

# Initial ODE data
x0 = 0
v0 = 0
pc1_0 = P_atm
pc2_0 = P_atm

y0 = [x0, v0, pc1_0, pc2_0]
t0 = 0
t_final = 15
ts = np.linspace(t0, t_final, 10000)
events = [work_hit_max,
          # p2_equals_atm,
          piston_reach_max_travel
          ]

sol = solve_ivp(ds_dt, [t0, t_final], y0, method='BDF', t_eval=ts, events=events)
# pprint(sol)

# RK23 - not bad
# Radau - Runge-Kutta error controlled - good result
# BDF - seems to be a bit slower than Radau
# LSODA - good result

x1 = sol.y[0]
v1 = sol.y[1]
p1 = sol.y[2]
p1_bar = [x/100_000 for x in p1]
p2 = sol.y[3]
p2_bar = [x/100_000 for x in p2]

time_to_fill = sol.t_events[0][0]
time_to_stroke = sol.t_events[1][0]
print(f"Stroking time = {time_to_stroke:.2f} sec")
print(f"Filling time = {time_to_fill:.2f} sec")

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(sol.t, x1, label="Travel")
ax1.plot(sol.t, v1, label="Velocity")
ax1.legend()

ax2.plot(sol.t, p1_bar, label="Pressure 1st cyl")
ax2.plot(sol.t, p2_bar, label="Pressure 2nd cyl")
ax2.legend()

f_pressure = np.array(p1)
f_result = np.array(p1)
f_spring = np.array(p1)
f_static_friction = np.array(p1)

qv_array = np.array(p1)
qm_array = np.array(p1)
gm_array = np.array(p1)

forces = {'f_pressure': f_pressure,
          'f_result': f_result,
          'f_spring': f_spring,
          'f_static_friction': f_static_friction}

flows = {'qv': qv_array,
         'qm': qm_array,
         'gm': gm_array}

for i, p in enumerate(p1):
    x = x1[i]
    get_f_result(
        x,
        p,
        p2[i],
        plot=True,
        force_dict=forces,
        flow_dict=flows,
        ind=i,
    )

ax3.plot(sol.t, f_pressure, label="F_pressure")
ax3.plot(sol.t, f_spring, label="F_spring")
ax3.plot(sol.t, f_static_friction, label="F_static_friction + valve load")
ax3.plot(sol.t, f_result, label="F_result")
ax3.legend()

t_result = sol.t

ax4.plot(t_result, qv_array, label="Qv")
ax4.plot(t_result, qm_array, label="Qm")
ax4.plot(t_result, gm_array, label="Gm")
ax4.legend()

plt.show()
