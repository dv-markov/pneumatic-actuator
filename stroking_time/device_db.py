from kv_dzeta import kv_calculated_for_pipe, kv_addition


class PneumaticActuator:
    def __init__(self, model: str, failsafe: str, pressure_bar: float,
                 torque_da_Nm: float, torque_spring_start_Nm: float, torque_spring_end_Nm: float,
                 total_volume_liter: float, weight_kg: float,
                 internal_channel_size_1_mm: float, internal_channel_size_2_mm: float,
                 channel_length_1_mm: float, channel_length_2_mm: float):
        self.model = model
        self.failsafe = failsafe
        self.pressure = pressure_bar * 100_000
        self.torque_da = torque_da_Nm
        self.torque_spring_start = torque_spring_start_Nm
        self.torque_spring_end = torque_spring_end_Nm
        self.total_volume = total_volume_liter / 1000
        self.weight = weight_kg
        self.internal_channel_size_1 = internal_channel_size_1_mm / 1000
        self.internal_channel_size_2 = internal_channel_size_2_mm / 1000
        self.channel_length_1 = channel_length_1_mm / 1000
        self.channel_length_2 = channel_length_2_mm / 1000

    def __str__(self):
        return (f"Pneumatic actuator type {self.model}, "
                f"Failsafe: {self.failsafe}, "
                f"Pressure: {self.pressure / 100_000} bar")


class PneumaticDevice:
    def __init__(self, name: str, kv_in: float, kv_out: float, connection_size_inch: float):
        self.name = name
        self.kv_in = kv_in / 3600
        self.kv_out = kv_out / 3600
        self.connection_size = connection_size_inch

    def __str__(self):
        return (f"Pneumatic device type {self.name}, "
                f"Kv_in: {round(self.kv_in*3600, 1)} m3/h, Kv_out: {round(self.kv_out*3600, 1)} m3/h, "
                f"Connection size: {self.connection_size} inch")


class PneumaticDeviceWithPipe(PneumaticDevice):
    def __init__(self, name: str, kv_in: float, kv_out: float, connection_size_inch: float,
                 pipe_inner_diameter_mm: float, pipe_length_mm: float):
        super().__init__(name, kv_in, kv_out, connection_size_inch)
        pipe_inner_diameter = pipe_inner_diameter_mm / 1000
        pipe_length = pipe_length_mm / 1000
        kv_pipe = kv_calculated_for_pipe(pipe_inner_diameter, pipe_length)
        self.kv_in = kv_addition(self.kv_in, kv_pipe)
        self.kv_out = kv_addition(self.kv_out, kv_pipe)


at_AT651U_S08_4bar = PneumaticActuator(
    "AT651U-S08",
    "FC",
    4,
    1430,
    834,
    577,
    10,
    69,
    6.6,
    6.6,
    250,
    1000,
)

airset_4708_45 = PneumaticDevice("4708-45", 1.5, 1.5, 1/2)
sov_3963_namur_4_3 = PneumaticDevice("3963 NAMUR 4,3", 1.9, 4.3, 1/2)
sov_3963_pipe_4_3 = PneumaticDeviceWithPipe("3963 Pipe 4,3", 1.9, 4.3, 1/2,
                                            10, 350)
sov_3963_pipe_1_4 = PneumaticDeviceWithPipe("3963 Pipe 1,4", 1.4, 1.4, 1/4,
                                            10, 350)
sov_3963_pipe_0_32 = PneumaticDeviceWithPipe("3963 Pipe 0,32", 0.32, 0.32, 1/4,
                                             10, 350)


if __name__ == "__main__":
    print(at_AT651U_S08_4bar)
    print(airset_4708_45)
    print(sov_3963_namur_4_3)
    print(sov_3963_pipe_4_3)
    print(sov_3963_pipe_1_4)
