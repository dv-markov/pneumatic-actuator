# Kv_data and piping data
kv_in_1 = 0.32 / 3600  # m^3/s
kv_out_1 = 2 / 3600  # m^3/s
# kv_in_1 = 1.1 / 3600  # m^3/s
# kv_out_1 = 2.0 / 3600  # m^3/s
d_pipe = 5 / 1000  # m
le_pipe = 100 / 1000  # m

# Actuator data
act_model = "VT240 S06"
torque_da = 1508  # N*m, @ P_in
torque_spring_start = 664.8  # N*m
torque_spring_end = 492.6  # N*m
total_volume = 11.40 / 1000  # m3
act_weight = 77.76  # kg

# act_model = "A651U S08"
# torque_da = 1430  # N*m, @ P_in
# torque_spring_start = 834  # N*m
# torque_spring_end = 577  # N*m
# total_volume = 10 / 1000  # m3
# act_weight = 69  # kg

# Valve data
valve_torque_bto = 330  # N*m
