import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import kv_dzeta
from initial_data import *

# V6 - use dp/dt formula from SPB_Polytechnic, Donskoy
# add new force concept with viscous force
# add Euler method

# Governing flags
QV_IEC = True
EULER_SOLVER = False

# Kv_calculation
kv_pipe_1 = kv_dzeta.kv_calculated_for_pipe(d_pipe_1, le_pipe_1)
print(f'Kv_pipe_1 = {kv_pipe_1 * 3600:.2f} m3/h')
kv_pipe_2 = kv_dzeta.kv_calculated_for_pipe(d_pipe_2, le_pipe_2)
print(f'Kv_pipe_2 = {kv_pipe_2 * 3600:.2f} m3/h')
kv_pipe_in = kv_dzeta.kv_addition(kv_pipe_1, kv_pipe_2)
print(f'Kv_pipe_in = {kv_pipe_in * 3600:.2f} m3/h')
kv_pipe_3 = kv_dzeta.kv_calculated_for_pipe(d_pipe_3, le_pipe_3)
print(f'Kv_pipe_3 = {kv_pipe_3 * 3600:.2f} m3/h')
kv_pipe_out = kv_dzeta.kv_addition(kv_pipe_2, kv_pipe_3)
# kv_pipe_out = kv_pipe_3
print(f'Kv_pipe_out = {kv_pipe_out * 3600:.2f} m3/h')
print(f"""Kv_in_device_1 = {kv_in_1 * 3600:.2f} m3/h; Kv_in_device_2 = {kv_in_2 * 3600:.2f} m3/h""")
print(f"""Kv_out_device_1 = {kv_out_1 * 3600:.2f} m3/h""")
kv_in_devices_total = kv_dzeta.kv_addition(kv_in_1, kv_in_2)
print(f"""Kv_in_devices_total = {kv_in_devices_total * 3600:.2f} m3/h""")
kv_total_1 = kv_dzeta.kv_addition(kv_in_devices_total, kv_pipe_in)  # m^3/s
kv_total_2 = kv_dzeta.kv_addition(kv_out_1, kv_pipe_out)  # m^3/s
print(f"""Kv_in_total = {kv_total_1 * 3600:.2f} m3/h
Kv_out_total = {kv_total_2 * 3600:.2f} m3/h""")


def get_relative_valve_torque(rel_travel):
    x_r = rel_travel
    f_x = 1 - 3.83*x_r + 13.65*x_r**2 - 28.39*x_r**3 + 33.51*x_r**4 - 19.53*x_r**5 + 4.34*x_r**6
    return f_x


def get_f_result(x, v, pc_1, pc_2, plot=False, force_dict=None, flow_dict=None, ind=None):
    relative_travel = x / xf
    valve_torque_factor = get_relative_valve_torque(relative_travel)
    valve_current_torque = valve_torque_bto * valve_torque_factor
    F_valve = valve_current_torque / lever_length
    F_static_load = F_valve + friction_force_still
    F_dynamic_load = F_valve + friction_force_move
    F_viscous_friction = 17 * h * v
    # <= 7 * h * v - makes back stroke
    # 17 - not bad, with RK23 gives results similar to real, with Radau also decent
    # 50 - also smooth enough
    # 70 - very smooth

    F_spring = spring_force_relaxed + spring_force_comp_factor * x
    F_pressure_1 = (pc_1 - P_atm) * S
    F_pressure_2 = (pc_2 - P_atm) * S
    F_pressure__spring = F_pressure_1 - F_pressure_2 - F_spring

    if v == 0:
    # print(v)
    # if abs(v) < 0.2:
        if abs(F_pressure__spring) < F_static_load:
            F_result = 0
        else:
            F_result = math.copysign(1, F_pressure__spring) * (
                    abs(F_pressure__spring) - F_static_load) - F_viscous_friction
    else:
        if abs(F_pressure__spring) < F_dynamic_load:
            F_result = 0
        else:
            F_result = math.copysign(1, F_pressure__spring) * (
                    abs(F_pressure__spring) - F_dynamic_load) - F_viscous_friction

    if plot:
        force_dict['f_static_load'][ind] = F_static_load
        force_dict['f_dynamic_load'][ind] = F_dynamic_load
        force_dict['f_viscous_friction'][ind] = F_viscous_friction
        force_dict['f_spring'][ind] = F_spring
        force_dict['f_pressure'][ind] = F_pressure_1 - F_pressure_2
        force_dict['f_result'][ind] = F_result if x < xf else 0

    return F_result


def ds_dt(t, y):
    x, v, pc_1, pc_2 = y

    F_result = get_f_result(x, v, pc_1, pc_2)

    if x <= 0:
        x = 0
        v = max(0, v)

    if x >= xf:
        x = xf
        dxdt = 0
        if v > 0:
            dvdt = -100*v
        # elif v < 0:
        #     dvdt = -10*v
        else:
            dvdt = 0
        v = 0
    else:
        dxdt = v
        dvdt = F_result / M

    if QV_IEC:
        # Cylinder 1
        delta_p_1_calc: float = P_in - pc_1
        delta_p_max_1 = x_t_valve * gamma_air * P_in / 1.4
        delta_p_1 = min(delta_p_1_calc, delta_p_max_1)
        qm_1_calc = math.copysign(1, delta_p_1) * kv_dzeta.qm_calculated(kv_total_1, rho_in, abs(delta_p_1))
        delta_p_factor_1 = delta_p_1_calc / P_in
        expansion_factor_1 = max(2/3, 1-1.4*delta_p_factor_1/(3*gamma_air*x_t_valve))
        qm_1 = qm_1_calc * expansion_factor_1
        dpc_1dt = (k * R * T_in * qm_1) / (S * (x + x_01)) - k * pc_1 * v / (x + x_01)
        # Cylinder 2
        delta_p_2_calc: float = pc_2 - P_atm
        T_2 = T_in * math.pow(abs(pc_2) / P_atm, (k - 1) / k)
        # print(f"{pc_2=}, T_2={T_2 - 273}")
        rho_2 = abs(pc_2) / (R * T_2)
        delta_p_max_2 = x_t_valve * gamma_air * pc_2 / 1.4
        delta_p_2 = min(delta_p_2_calc, delta_p_max_2)
        qm_2_calc = math.copysign(1, delta_p_2) * kv_dzeta.qm_calculated(kv_total_2, rho_2, abs(delta_p_2))
        delta_p_factor_2 = delta_p_2_calc / pc_2
        expansion_factor_2 = max(2/3, 1-1.4*delta_p_factor_2/(3*gamma_air*x_t_valve))
        qm_2 = qm_2_calc * expansion_factor_2
        dpc_2dt = (-1) * (k * R * T_2 * qm_2) / (S * (xf - x + x_02)) + k * pc_2 * v / (xf - x + x_02)
    else:
        # Cylinder 1
        dpc_1dt = math.copysign(1, (P_in - pc_1)) * kv_total_1 * k * math.sqrt(
            R*T_in*0.005*(abs(P_in**2 - pc_1**2))
        ) / (S * (x + x_01)) - k * pc_1 * v / (x + x_01)
        # Cylinder 2
        dpc_2dt = (-1) * math.copysign(1, (pc_2 - P_atm)) * kv_total_2 * k * math.sqrt(
            R * T_in * 0.005 * (abs(pc_2 ** 2 - P_atm ** 2))
        ) * math.pow(
            abs(pc_2) / P_in,
            (k - 1) / (2 * k)
        ) / (S * (xf - x + x_02)) + k * pc_2 * v / (xf - x + x_02)

    return [dxdt, dvdt, dpc_1dt, dpc_2dt]


def euler(time, y0):
    x = np.zeros_like(time)
    v = np.zeros_like(time)
    pc_1 = np.zeros_like(time)
    pc_2 = np.zeros_like(time)

    x[0], v[0], pc_1[0], pc_2[0] = y0

    dt = time[1] - time[0]
    t_res = time
    time_stroke = math.inf

    for i, t in enumerate(time[:-1]):
        dxdt, dvdt, dpc_1dt, dpc_2dt = ds_dt(None, [x[i], v[i], pc_1[i], pc_2[i]])
        x[i + 1] = x[i] + dxdt * dt
        v[i + 1] = v[i] + dvdt * dt
        pc_1[i + 1] = pc_1[i] + dpc_1dt * dt
        pc_2[i + 1] = pc_2[i] + dpc_2dt * dt

        if x[i + 1] >= xf and t < time_stroke:
            time_stroke = t

        if x[i + 1] >= xf and pc_1[i + 1] >= P_in:
            t_res = time[:i+1]
            x = x[:i + 1]
            v = v[:i + 1]
            pc_1 = pc_1[:i + 1]
            pc_2 = pc_2[:i + 1]
            break

    y = [x, v, pc_1, pc_2]

    return t_res, y, time_stroke


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
t_final = 30
ts = np.linspace(t0, t_final, 10_000)
events = [work_hit_max,
          # p2_equals_atm,
          piston_reach_max_travel
          ]

if EULER_SOLVER:
    # Euler's method
    eul = euler(ts, y0)

    t_result = eul[0]
    x1 = eul[1][0]
    x1_mm = x1 * 1000
    v1 = eul[1][1]
    v1_mm_s = v1 * 1000
    p1 = eul[1][2]
    p1_bar = [(x - P_atm) / 100_000 for x in p1]
    p2 = eul[1][3]
    p2_bar = [(x - P_atm) / 100_000 for x in p2]

    time_to_stroke = eul[2]
    time_to_fill = t_result[-1]
else:
    sol = solve_ivp(ds_dt, [t0, t_final], y0, method='LSODA', t_eval=ts, events=events)
    # pprint(sol)

    # RK23 - not bad
    # Radau - Runge-Kutta error controlled - good result
    # BDF - seems to be a bit slower than Radau
    # LSODA - good result, even better results with viscous friction force

    # Results for SciPy
    t_result = sol.t
    x1 = sol.y[0]
    x1_mm = x1 * 1000
    v1 = sol.y[1]

    # Сглаживание
    # win = 300
    # filt = np.ones(win)/win
    # mov = win // 2
    # res = np.convolve(v1, filt, mode='same')
    # res = v1

    v1_mm_s = v1 * 1000
    p1 = sol.y[2]
    p1_bar = [(x - P_atm) / 100_000 for x in p1]
    p2 = sol.y[3]
    p2_bar = [(x - P_atm) / 100_000 for x in p2]

    time_to_stroke = time_to_fill = math.inf
    if sol.t_events:
        if sol.t_events[1].any():
            time_to_stroke = sol.t_events[1][0]
        if sol.t_events[0].any():
            time_to_fill = sol.t_events[0][0]

print(f"Stroking time = {time_to_stroke:.2f} sec")
print(f"Filling time = {time_to_fill:.2f} sec")

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(t_result, x1_mm, label="Travel, mm")
ax1.plot(t_result, v1_mm_s, label="Velocity, mm/s")
ax1.legend()

ax2.plot(t_result, p1_bar, label="Pressure 1st cyl, bar(g)")
ax2.plot(t_result, p2_bar, label="Pressure 2nd cyl, bar(g)")
ax2.legend()

f_pressure = np.zeros_like(p1)
f_result = np.zeros_like(p1)
f_spring = np.zeros_like(p1)
f_static_load = np.zeros_like(p1)
f_dynamic_load = np.zeros_like(p1)
f_viscous_friction = np.zeros_like(p1)

# qv_array = np.array(p1)
# qm_array = np.array(p1)
# gm_array = np.array(p1)

forces = {'f_pressure': f_pressure,
          'f_result': f_result,
          'f_spring': f_spring,
          'f_static_load': f_static_load,
          'f_dynamic_load': f_dynamic_load,
          'f_viscous_friction': f_viscous_friction,
          }

# flows = {'qv': qv_array,
#          'qm': qm_array,
#          'gm': gm_array}

for i, p in enumerate(p1):
    x = x1[i]
    get_f_result(
        x,
        v1[i],
        p,
        p2[i],
        plot=True,
        force_dict=forces,
        # flow_dict=flows,
        ind=i,
    )

ax3.plot(t_result, f_pressure, label="F_pressure, N")
ax3.plot(t_result, f_spring, label="F_spring, N")
ax3.plot(t_result, f_static_load, label="F_static_load, N")
ax3.plot(t_result, f_dynamic_load, label="F_dynamic_load, N")
ax3.plot(t_result, f_viscous_friction, label='Viscous friction, N')
ax3.legend()

# ax4.plot(t_result, qv_array, label="Qv")
# ax4.plot(t_result, qm_array, label="Qm")
# ax4.plot(t_result, gm_array, label="Gm")
ax4.plot(t_result, f_result, label="F_result, N")
# ax4.plot(t_result, v1_mm_s, label="Velocity, mm/s")
ax4.legend()

plt.show()
