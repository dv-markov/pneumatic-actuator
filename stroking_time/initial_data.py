from constants import *
from valve_and_actuator_data import *

# Variables
# General data
T_in = 20 + 273  # K
P_supply = 4.0 * 100_000  # Pa
P_in = P_supply + P_atm  # Pa
P_master_force = 4.0 * 100_000 + P_atm
rho_in = P_in / (R * T_in)
# Calculated pneumatic valve values
delta_p_max = x_t_valve * gamma_air * P_in / 1.4  # Pa

# Advanced data - calculated actuator values
h = 300
k_friction = 0.12
alfa = 2.89  # invariant of rack & pinion piston actuator, alfa = piston_diameter / lever_length
piston_diameter = math.pow((4 * alfa * torque_da / pi / P_supply), 1/3)  # m
piston_surface = pi * math.pow(piston_diameter, 2) / 4  # m2
lever_length = torque_da / P_supply / piston_surface
dead_volume = total_volume * 0.1
full_stroke = (total_volume - dead_volume) / piston_surface
spring_force_relaxed = torque_spring_end / full_stroke
spring_force_comp_factor = (torque_spring_start - torque_spring_end) / full_stroke / lever_length
friction_force_move = P_master_force * k_friction * piston_surface
friction_force_still = friction_force_move * 3
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
