import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# V2 - Use adiabatic process formula to calculate current temp in the cylinder:
# pT^(gamma/gamma-1) = const

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
kv_total = 0.5 / 3600  # m^3/s
rho_in = P_in / (R * T_in)
# Calculated pneumatic valve values
delta_p_max = x_t_valve * gamma_air * P_in / 1.4  # Pa

# Actuator data
act_model = "VT240 S06"
torque_da = 1508  # N*m, @ P_in
torque_spring_start = 664.8  # N*m
torque_spring_end = 492.6  # N*m
total_volume = 11.40 / 1000  # m3
act_weight = 77.76

# Valve data
valve_torque_bto = 330  # N*m

# Advanced data - calculated actuator values
alfa = 2.89  # invariant of rack & pinion piston actuator, alfa = piston_diameter / lever_length
piston_diameter = math.pow((4 * alfa * torque_da / pi / P_supply), 1/3)  # m
piston_surface = pi * math.pow(piston_diameter, 2) / 4  # m2
lever_length = torque_da / P_supply / piston_surface
dead_volume = total_volume * 0.2
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
    {torque_air_end=};""",
    # sep="\n",
)

# Calculated values
# piston_diameter = piston_diameter
# V_extra = dead_volume # m3
xf = full_stroke  # m

# Initial ODE data
x0 = 0
v0 = 0
T0 = T_in
m0 = P_atm * dead_volume / R / T0
# P0 = m0 / dead_volume * R * T0

print(f"""
{m0=},
{rho_in=};
""")


def get_relative_valve_torque(relative_travel):
    x_r = relative_travel
    f_x = 1 - 3.83*x_r + 13.65*x_r**2 - 28.39*x_r**3 + 33.51*x_r**4 - 19.53*x_r**5 + 4.34*x_r**6
    return f_x


def ds_dt(t, y):
    x, v, T_s, m_s = y

    P_s = m_s * R * T_s / (x * piston_surface + dead_volume)

    # P_s = math.pow((m_s / (x * piston_surface + dead_volume) * R * T_in * P_in**(1/3.5)), 1.75)
    # P_s = m_s / (x * piston_surface + dead_volume) * R * T_in
    operating_pressure.append(P_s)
    print(P_s / 100_000)

    delta_p_valve = abs(P_in - P_s)
    x_factor_valve = delta_p_valve / P_in
    expansion_factor = max(2/3, (1 - (1/3) * (1.4/gamma_air) * (x_factor_valve/x_t_valve)))

    Q_V_in = kv_total * expansion_factor * math.pow(
        (rho_0 * min(delta_p_valve, delta_p_max) / rho_in / delta_p_0),
        1/2)
    Q_m_in = rho_in * Q_V_in

    relative_travel = x / xf
    valve_torque_factor = get_relative_valve_torque(relative_travel)
    valve_current_torque = valve_torque_bto * valve_torque_factor
    F_valve = valve_current_torque / lever_length

    # add - DM - friction force
    F_static_friction = friction_force + F_valve
    F_spring = spring_force_relaxed + spring_force_comp_factor * x
    F_pressure_spring = (P_s - P_atm) * piston_surface - F_spring
    F_diff = abs(F_pressure_spring) - F_static_friction
    if F_diff > 0:
        F_result = math.copysign(1, F_pressure_spring) * F_diff
    else:
        F_result = 0

    dxdt = v if x < xf else 0
    dvdt = F_result / M if x < xf else 0 if v == 0 else -abs(v)
    dTdt = (-c_p * T_s * Q_m_in - P_s * piston_surface * v + Q_m_in * c_v * T_in) / c_p / m_s
    dmdt = Q_m_in

    # operating_pressure.append(P_s)
    # operating_time.append(t)

    return [dxdt, dvdt, dTdt, dmdt]


def work_hit_max(t, y, *args):
    return y[0] - xf


work_hit_max.terminal = True

y0 = [x0, v0, T0, m0]
t0 = 0
t_final = 15
ts = np.linspace(t0, t_final, 1000)
events = [work_hit_max, ]
operating_pressure = []
# operating_time = []

sol = solve_ivp(ds_dt, [t0, t_final], y0, t_eval=ts, events=events)

x1 = sol.y[0]
v1 = sol.y[1]
T1 = [x - 273 for x in sol.y[2]]
m1 = sol.y[3]
# p1 = [x/1000_000 for x in sol.y[4]]

time_to_fill = sol.t[-1]
print("Time to fill", time_to_fill)
psize = len(operating_pressure)
tp = np.linspace(t0, time_to_fill, psize)
operating_pressure = [x/100_000 for x in operating_pressure]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(sol.t, x1, label="Travel")
ax1.plot(sol.t, v1, label="Velocity")
ax1.plot(sol.t, m1, label="Mass")
# ax1.plot(sol.t, p1, label="Pressure")
ax1.legend()
# ax2.plot(sol.t, T1, label="Temperature")
ax2.plot(tp, operating_pressure, label="Pressure")
ax2.legend()

ax3.plot(sol.t, T1, label="Temperature")
ax3.legend()

plt.show()


# print(operating_pressure)
# print(operating_time)

