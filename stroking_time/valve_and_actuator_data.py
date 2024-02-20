# Kv_data and piping data
kv_in_1 = 0.32 / 3600  # m^3/s, 3963
kv_in_2 = 1.5 / 3600  # m^3/s, 4708-45
kv_out_1 = 2 / 3600  # m^3/s
# kv_in_1 = 1.1 / 3600  # m^3/s
# kv_out_1 = 2.0 / 3600  # m^3/s
d_pipe_1 = 6 / 1000  # m, internal diameter
le_pipe_1 = 1000 / 1000  # m
d_pipe_2 = 8 / 1000  # m, internal diameter
le_pipe_2 = 3  # m
d_pipe_3 = 5 / 1000  # m
le_pipe_3 = 500 / 1000  # m

# Actuator data
# act_model = "VT240 S06"
# torque_da = 1508  # N*m, @ P_in
# torque_spring_start = 664.8  # N*m
# torque_spring_end = 492.6  # N*m
# total_volume = 11.40 / 1000  # m3
# act_weight = 77.76  # kg

# act_model = "AT651U S08 / SC 2000"
# torque_da = 1430  # N*m, @ P_in
# torque_spring_start = 834  # N*m
# torque_spring_end = 577  # N*m
# total_volume = 10 / 1000  # m3
# act_weight = 69  # kg

act_model = "AT651U S07 / SC 2000"  # @ 3.5 bar(g)
torque_da = 1251  # N*m, @ P_in
torque_spring_start = 730  # N*m
torque_spring_end = 505  # N*m
total_volume = 10 / 1000  # m3
act_weight = 69  # kg

# act_model = "AT651U S12 / SC 2000"  # @ 6 bar(g)
# torque_da = 2144  # N*m, @ P_in
# torque_spring_start = 1251  # N*m
# torque_spring_end = 865  # N*m
# total_volume = 10 / 1000  # m3
# act_weight = 69  # kg

# act_model = "AT301U S07 / SC 150"  # @ 3.5 bar(g)
# torque_da = 93.1  # N*m, @ P_in
# torque_spring_start = 55.1  # N*m
# torque_spring_end = 35.5  # N*m
# total_volume = 0.71 / 1000  # m3
# act_weight = 6  # kg

# act_model = "AT301U S12 / SC 150"  # @ 6.0 bar(g)
# torque_da = 160  # N*m, @ P_in
# torque_spring_start = 94.5  # N*m
# torque_spring_end = 60.8  # N*m
# total_volume = 0.71 / 1000  # m3
# act_weight = 6  # kg

# act_model = "AT1001U S07 / SC 10000"  # @ 3.5 bar(g)
# torque_da = 5837  # N*m, @ P_in
# torque_spring_start = 5939  # N*m
# torque_spring_end = 4068  # N*m
# total_volume = 49 / 1000  # m3
# act_weight = 238  # kg

# act_model = "AT1001U S12 / SC 10000"  # @ 6.0 bar(g)
# torque_da = 10007  # N*m, @ P_in
# torque_spring_start = 5939  # N*m
# torque_spring_end = 4068  # N*m
# total_volume = 49 / 1000  # m3
# act_weight = 238  # kg

# Valve data
valve_torque_bto = 0  # N*m
