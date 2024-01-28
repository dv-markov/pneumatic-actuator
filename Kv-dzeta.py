import math

pi = math.pi
delta_p_0 = 100_000  # Pa
rho_0 = 1000  # kg/m3
R = 286.69  # J/kg*K, for air
p_atm = 101_325  # Pa


def dzeta_calculated_from_kv(d, kv):
    return (delta_p_0 * pi ** 2 * d ** 4) / (8 * rho_0 * kv ** 2)


def rho_calculated(pressure, temperature):
    return pressure / (R * temperature)


def kv_calculated_from_qv(qv, rho_op, delta_p_1):
    return qv * math.pow(rho_op * delta_p_0, 1 / 2) / math.pow(rho_0 * delta_p_1, 1 / 2)


def qv_calculated(kv, rho_op, delta_p_1):
    return kv * math.pow(rho_0 * delta_p_1, 1 / 2) / math.pow(rho_op * delta_p_0, 1 / 2)


def qm_calculated(qv, rho_op):
    return qv * rho_op


def gm_calculated(kv, p_1, p_2, T_op):
    sigma = p_2 / p_1
    return kv * p_1 * math.pow(rho_0 * (1 - sigma ** 2), 1 / 2) / math.pow(2 * delta_p_0 * R * T_op, 1 / 2)


def kv_calculated_from_dzeta(d, dzeta):
    f = pi * d ** 2 / 4
    return f * math.pow(2 * delta_p_0 / (dzeta * rho_0), 1 / 2)


# Input data
D = 8  # mm
Kv = 0.5  # m3/h
T_M = 20  # deg. C
T_1 = T_M + 273  # K
p_M = 4  # bar
p_1 = p_M * 100_000 + p_atm  # Pa
rho_1 = rho_calculated(p_1, T_1)  # kg/m3
p_2 = (p_M - 1) * 100_000 + p_atm  # Pa
delta_p = p_1 - p_2

dia = D / 1000  # m
kvs = Kv / 3600  # m3/s
res = dzeta_calculated_from_kv(dia, kvs)
print('Dzeta = ', res, end="\n\n")


qv1 = qv_calculated(kvs, rho_1, delta_p)
print('Qv =', qv1, "m3/s =", qv1 * 3600, "m3/h;")
qm1 = qm_calculated(qv1, rho_1)
print('Qm =', qm1, "kg/s =", qm1 * 3600, "kg/h;")
gm1 = gm_calculated(kvs, p_1, p_2, T_1)
print('Gm = ', gm1, "kg/s =", gm1 * 3600, "kg/h;")
print()

accessories_p_17 = [
    [1, 3, 19.2],
    [2, 4, 16.6],
    [3, 4, 12.4],
    [4, 8, 16.7],
    [5, 4, 12.4],
    [6, 8, 18.1],
    [7, 4, 12.4],
    [9, 8, 18.1],
    [10.1, 10, 17.1],
    [10.2, 15, 17.1],
    [10.1, 20, 17.1],
    [12, 3, 19.2],
    [13, 4, 16.6],
    [25, 10, 41],
    [26, 15, 33.2],
    [27, 10, 10.2],
    [28, 15, 12.8],
    [29.1, 10, 21.1],
    [29.1, 15, 21.1],
    [29.1, 25, 21.1],
]

print("Расчет Kvs для советской пневмоаппаратуры")
for acc in accessories_p_17:
    d_mm = acc[1]
    d_meter = d_mm / 1000
    dzeta = acc[2]
    print(f"N {acc[0]}, d = {d_mm} mm, dzeta = {dzeta}, "
          f"Kvs = {kv_calculated_from_dzeta(d_meter, dzeta) * 3600:.2f} m3/h")
