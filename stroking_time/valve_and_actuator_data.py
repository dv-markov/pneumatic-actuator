from device_db import (at_AT651U_S08_4bar,
                       airset_4708_45,
                       sov_3963_namur_4_3,
                       sov_3963_pipe_4_3,
                       sov_3963_pipe_1_4,
                       sov_3963_pipe_0_32,
                       )

# Pneumatic device Kv_data
pd1 = sov_3963_pipe_0_32
pd2 = airset_4708_45
kv_in_1 = pd1.kv_in
kv_in_2 = pd2.kv_in

# Actuator data
act = at_AT651U_S08_4bar
act_model = act.model
torque_da = act.torque_da
torque_spring_start = act.torque_spring_start
torque_spring_end = act.torque_spring_end
total_volume = act.total_volume
act_weight = act.weight

# Piping data
kv_out_1 = 2 / 3600  # m^3/s  # supposes silencer / bug filter on the outlet
d_pipe_1 = 8 / 1000  # m, supply pressure pipe
le_pipe_1 = 3  # m
d_pipe_2 = act.internal_channel_size_1  # m, internal channels in
le_pipe_2 = act.channel_length_1  # m
d_pipe_3 = act.internal_channel_size_2  # m , internal channels out
le_pipe_3 = act.channel_length_2  # m

# Valve data
# valve_torque_bto = 330  # N*m
valve_torque_bto = 0  # N*m
