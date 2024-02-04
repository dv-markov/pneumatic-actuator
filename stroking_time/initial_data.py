from constants import *

# Variables
# General data
T_in = 20 + 273  # K
P_supply = 4.0 * 100_000  # Pa
P_in = P_supply + P_atm  # Pa
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

# Calculated values
# piston_diameter = piston_diameter
# V_extra = dead_volume # m3
x_01 = dead_volume / piston_surface
x_02 = x_01 * 2
xf = full_stroke  # m
S = piston_surface
k = gamma_air


if __name__ == "__main__":
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
