import filterpy.kalman
import numpy as np
import pandas as pd
from filterpy.common import Q_discrete_white_noise, Saver
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import least_squares
import numpy.polynomial.polynomial as poly

Anc_2D = [
    [6160189.068799788, 235476.2946930449],
    [6160105.440872673, 235527.55777198478],
    [6160007.2800598815, 235462.62533516675],
]
# Anc_2D = [
#     [6160189.068799788, 235476.2946930449],
#     [6160162.550425654, 235499.53862634592],
#     [6160158.070704494, 235459.50499518885]
# ]

BASE_STATION_COORDS = (6160188.9135, 235476.0090)
# LR_POLY3 = poly.Polynomial([1.70667308,  0.05036929,  0.91299066,  0.25358677, -1.04010403], [2.816559, 187.039706])
# LR_POLY = poly.Polynomial([0.1], [100, 150.881626])
# short range poly # LR_POLY = poly.Polynomial([ 0.51075268, -0.00117379, -0.43347423,  0.54042526,  0.38338424, -0.33350059, -0.14380512], [0, 100])
# LR_POLY = poly.Polynomial([0])
LR_POLY = poly.Polynomial([ 1.54826851,  0.05238909,  0.28801415,  0.24172409, -0.54328581, -0.15440796,  0.32235139], domain=[ 8.841323, 60])
# LR_POLY2 = poly.Polynomial([ 1.9180532 , -0.50012581,  0.84565419,  1.46381503, -1.71216534, -1.00912226,  0.93729064], domain=[ 60.001, 187.039706])
LR_POLY = poly.Polynomial([1])

Anc_2D = [
    [coord[0] - BASE_STATION_COORDS[0], coord[1] - BASE_STATION_COORDS[1]] for coord in Anc_2D
]

# x, y = sym.symbols("x,y")
initial = (-60, 15)
# time_filter = ("04:05:40", "04:06:40")
time_filter = ("05:08:00", "05:09:00")
# data_source = "2024-07-05 14:03:00.171006"
data_source = "2024-07-05 15:02:00.308097"

# dir_fp = "../results/v2/2024-07-05 15:02:00.308097/"
dir_fp = f"../results/v2/{data_source}/"
headers = lambda x: ["dt", f"dist{x}"]
dtypes = lambda x: {"dt": float, f"dist{x}": float}
# parse_dates = ["anc1_dt", "anc0_dt", "anc2_dt", "gnss_dt"]
anc0 = pd.read_csv(dir_fp + "uwb-0.csv", delimiter=',', names=headers(0), dtype=dtypes(0))
anc1 = pd.read_csv(dir_fp + "uwb-1.csv", delimiter=',', names=headers(1), dtype=dtypes(1))
anc2 = pd.read_csv(dir_fp + "uwb-2.csv", delimiter=',', names=headers(2), dtype=dtypes(2))

anc0.set_index('dt', inplace=True)
anc1.set_index('dt', inplace=True)
anc2.set_index('dt', inplace=True)

anc0.index = pd.to_datetime(anc0.index, unit='s').round('50ms')
anc1.index = pd.to_datetime(anc1.index, unit='s').round('50ms')
anc2.index = pd.to_datetime(anc2.index, unit='s').round('50ms')

# anc0.resample(rule='50ms').mean()
# anc1.resample(rule='50ms').mean()
# anc2.resample(rule='50ms').mean()


gnss_data = pd.read_csv(dir_fp + "gnss.csv", delimiter=',', names=["dt", "northing", "easting"], dtype={"dt": float, "northing": float, "easting": float})
gnss_data.set_index('dt', inplace=True)
gnss_data.index = pd.to_datetime(gnss_data.index, unit='s').round('50ms')
gnss_data = gnss_data.between_time(*time_filter)
gnss_data["northing"] = gnss_data["northing"] - BASE_STATION_COORDS[0]
gnss_data["easting"] = gnss_data["easting"] - BASE_STATION_COORDS[1]


loaded_df = anc0.join(anc1, how='outer').join(anc2, how='outer').join(gnss_data, how='outer')
loaded_df = loaded_df.between_time(*time_filter)
loaded_df = loaded_df.resample(rule='50ms').mean()

print(len(loaded_df), len(gnss_data))

def perform_kf(Xk, A, Pk, Q, H, R, Z):
    # 1. Project the state ahead
    Xk_prev = A * Xk           # There is no control input
    # 2. Project the error covariance ahead
    Pk_prev = np.matmul(A, Pk, np.transpose(A)) + Q  # Initial value for Pk shoud be guessed.

    S = np.matmul(H, Pk_prev, np.transpose(H), casting='unsafe') + R   # prepare for the inverse

    # 1. compute the Kalman gain
    K = (np.matmul(Pk_prev, np.transpose(H)))/S   # K = Pk_prev * H' * inv(S);
    # 2. update the estimate with measurement Zk
    Xk = Xk_prev + np.matmul(K, (Z - np.matmul(H, Xk_prev)))
    # 3. Update the error Covariance
    Pk = Pk_prev - np.matmul(K, H, Pk_prev)
    return Xk, Pk


results = []

prev_vals = [0]*len(Anc_2D)

dt = 0.05

R_std = 5
Q_std = 0.01

kf_results = []

kf = filterpy.kalman.KalmanFilter(dim_x=4, dim_z=2)
# kf.x = np.array([2.5, 2.5, 1.5, 2]).reshape(4, 1)  # initial guess
kf.x = np.array([[*initial, 0, 0]]).T
kf.P = np.eye(4) * 500
# q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
# print(q, block_diag(q, q))
# kf.Q = block_diag(q, q)
kf.Q = np.diag([dt**2/2, dt**2, 1, 1])
# kf.Q = np.diag([dt**4/4, dt**4/4, dt**2/2, dt**2/2])
# kf.R = np.eye(2) * R_std**2
# kf.R = np.eye(2) * R_std**2
kf.R = np.array([
    [0.1, 0],
    [0, 0.1],
])

kf.H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
kf.F = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
])


for row in loaded_df.iterrows():
    vals = row[1].values
    # is_initial = initial[0] == 0 and initial[1] == 0
    # if is_initial and np.isnan(vals).any():
    #     continue
    # print(vals)
    eqns = []
    for i, known in enumerate(Anc_2D):
        distance = vals[i]
        # print(distance)
        if np.isnan(distance):
            # print(i, distance)
            # distance = prev_vals[i]
            continue

        # if distance > 60:
        #     distance = distance - LR_POLY2(distance)
        # else:
        distance = distance - LR_POLY(distance)

        prev_vals[i] = distance
        def wrapped(k, d):
            def func(x, y):
                r = (x - k[0]) ** 2 + (y - k[1]) ** 2 - d ** 2
                return r
            return func
        # eqns.append(lambda x, y: (x - known[0]) ** 2 + (y - known[1]) ** 2 - distance ** 2)
        eqns.append(wrapped(known, distance))

    def _guess(guess):
        x, y = guess
        return [eqn(x, y) for eqn in eqns]

    result = least_squares(_guess, initial)
    # print(result)
    res = result.x
    initial = res
    results.append(res)

    # kf.predict()
    # kf.update(res)
    #
    # kf_results.append(kf.x)
    # print(kf.x)

s = Saver(kf)
kf.batch_filter(results, saver=s)
s.to_array()

print(len(s.x), len(loaded_df.easting))

if __name__ == "__main__":
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("UWB-KF v GNSS", fontsize=14, fontweight='bold')

    plt.subplot(321)
    plt.scatter(s.x[:, 2], s.x[:, 0], c=list(range(len(s.x))))
    plt.scatter(gnss_data["easting"], gnss_data["northing"], c=list(range(gnss_data.shape[0])), cmap="Wistia")
    plt.plot([coord[1] for coord in Anc_2D], [coord[0] for coord in Anc_2D], 'xr')
    plt.title("Tri + KF v GNSS")
    plt.legend(["KF", "GNSS"])
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    ax = plt.gca()

    plt.subplot(322)
    plt.scatter([r[1] for r in results], [r[0] for r in results], c=list(range(len(results))), cmap='cool')
    plt.scatter(gnss_data["easting"], gnss_data["northing"], c=list(range(gnss_data.shape[0])), cmap="Wistia")
    plt.plot([coord[1] for coord in Anc_2D], [coord[0] for coord in Anc_2D], 'xr')
    plt.title("Tri v GNSS")
    plt.legend(["Tri", "GNSS"])
    plt.xlabel("Easting (m)")
    # plt.ylabel("Northing (m)")

    plt.subplot(312)

    print(s.x[:, 0].reshape((-1,)))
    northing_error = loaded_df.northing.to_numpy() - s.x[:, 0].reshape((-1,))
    easting_error = loaded_df.easting.to_numpy() - s.x[:, 2].reshape((-1,))
    plt.plot(loaded_df.index, northing_error)
    plt.plot(loaded_df.index, easting_error, color='r')
    plt.legend(["Northing", "Easting"])
    plt.xlabel("time")
    plt.ylabel("metres")
    plt.title("GNSS-KF Error")

    plt.subplot(313)

    df2 = loaded_df.resample(rule='s').count()
    plt.plot(df2.index, df2.dist0, color='r')
    plt.plot(df2.index, df2.dist1, color='g')
    plt.plot(df2.index, df2.dist2, color='b')
    plt.plot(df2.index, df2.northing, color='orange')
    plt.legend(["Anc 0", "Anc 1", "Anc 2", "GNSS"])
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("Frequency")

    # plt.subplot(414)

    ax = plt.gca()
    # ax.tick_params(labelsize=14)

    # Add a footnote below and to the right side of the chart
    plt.figtext(0.99, 0.01, f"Data: {data_source}, Start: {time_filter[0]}, End: {time_filter[1]} (UTC)", ha="right", fontsize=4,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    # print(s.x[:, 0])
    # res = [[r[0]] for r in results] - s.x[:, 0]
    # # for i, row in enumerate(res):
    # #     print(f"{i}: {row}, {s.x[i, 0]}, {results[i][0]}")
    # plt.plot(res)
    #
    # std = np.sqrt(s.P[:, 0, 0]) * 1
    # plt.plot(-std, color='k', ls=':', lw=2)
    # plt.plot(std, color='k', ls=':', lw=2)
    # plt.fill_between(range(len(std)), -std, std,
    #              facecolor='#ffff00', alpha=0.3)
    #
    # plt.xlabel("time (sec)")
    # plt.ylabel("metres")
    # plt.title("KF Residuals")


    # ax.set_aspect('equal')
    plt.show()

