import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter


from trilat_kf import results
# %%%%%%%%%%%%%%%%%%%% REAL MEASUREMENT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Load the logged Data
# getRangeUWB = importfile_Ranges('exp_data/UWB_data_Ranges/output_range_uwb_m2r.txt');
# [rowR, colR] = size(getRangeUWB);
# ts_R = getRangeUWB.ts;
# tid  = getRangeUWB.tagID;       % tag ID no.
# r_t2A0 = getRangeUWB.T2A0;      % tag to Anc0 measured values
# r_t2A1 = getRangeUWB.T2A1;      % tag to Anc1 measured values
# r_t2A2 = getRangeUWB.T2A2;      % tag to Anc2 measured values
# r_t2A3 = getRangeUWB.T2A3;      % tag to Anc3 measured values
# %}

# % Rescale the measured ranges into the original values in meter
# r_t2A0 = r_t2A0 ./ 1000;        % the data are scaled with 1000 in the log file
# r_t2A1 = r_t2A1 ./ 1000;
# r_t2A2 = r_t2A2 ./ 1000;
# r_t2A3 = r_t2A3 ./ 1000;
#
# % Range values matrix. Each ranges from tag to each anchors is stored in
# % the columns of the matrix
# t2A_4R = [r_t2A0 r_t2A1 r_t2A2 r_t2A3]; % use 4 ranges
#
# dimKF = 2;

Anc_2D = [
    [6160189.068799788, 235476.2946930449],
    [6160105.440872673, 235527.55777198478],
    [6160007.2800598815, 235462.62533516675],
]

BASE_STATION_COORDS = (6160188.9135, 235476.0090)

Anc_2D = [
    [coord[0] - BASE_STATION_COORDS[0], coord[1] - BASE_STATION_COORDS[1]] for coord in Anc_2D
]

# time_filter = ("05:08:45", "05:09:45")
time_filter = ("04:06:00", "04:07:30")
data_source = "2024-07-05 14:03:00.171006"

initial = (-15, -30)
# A0_2d = [0, 0]
# A1_2d = [5.77, 0]
# A2_2d = [5.55, 5.69]
# A3_2d = [0, 5.65]
#
# Anc_2D = [A0_2d, A1_2d, A2_2d, A3_2d]

dt = 0.05

# with open("demo.txt") as f:
#     data = f.readlines()
#
# ts_R = []
# r_t2A0 = []
# r_t2A1 = []
# r_t2A2 = []
# r_t2A3 = []
#
# for row in data:
#     ts, tag, a0, a1, a2, a3 = row.split()
#     ts_R.append(int(ts.strip()))
#     r_t2A0.append(int(a0.strip())/1000)
#     r_t2A1.append(int(a1.strip())/1000)
#     r_t2A2.append(int(a2.strip())/1000)
#     r_t2A3.append(int(a3.strip())/1000)
#
# t2A_4R = [r_t2A0, r_t2A1, r_t2A2, r_t2A3]


# def get_const_acceleration_kf():
#     # function [Xk, A, Pk, Q, H, R] = initConstAcceleration_KF(dim)
#     # dt = 0.2    # update rate of the system (10 Hz)
#     #
#     # % Process noise and measurement noise setup
#     # % Note: These values can be tuned to get the most out of it for each
#     # % specific application since KF uses these noises across all evaluations.
#     # % This constant noise may not reflect well in every conditions.
#     #
#     # % Process noise based on the Decawave datasheet excluding (Z -direction)
#     v_x = 0.01  # the precision (m) of DW1000 in var (10 cm)
#     v_y = 0.01
#     v_z = 0.01855
#     #
#     # % Measurement noise
#     v_xm = 0.0137  # Based on our prior data evaluation
#     v_ym = .0153
#     v_zm = 0.02855
#     # % Initial guess of State vector: Xk = [ x, y, vx, vy, ax, ay]
#     Xk = [2.5, 2.5, 1.5, 2, 1.0, 1.2]    # a posteriori
#     A = [
#         [1,  0,   dt,   0,   dt**2/2,     0,     ],
#         [0,  1,   0,    dt,  0,           dt**2/2],
#         [0,  0,   1,    0,   dt,          0      ],
#         [0,  0,   0,    1,   0,           dt     ],
#         [0,  0,   0,    0,   1,           0      ],
#         [0,  0,   0,    0,   0,           1      ],
#     ]
#     # % Error Covariance Matrix P (Posterior)
#     # % Initial guess just non zero entry.
#     Pk = [
#         [2,    0,     0,    0,    0,   0],
#         [0,    2,     0,    0,    0,   0],
#         [0,    0,     2,    0,    0,   0],
#         [0,    0,     0,    2,    0,   0],
#         [0,    0,     0,    0,    2,   0],
#         [0,    0,     0,    0,    0,   2],
#         ]
#
#     # % Process Noise
#     # % accoording to the book "estimation with applications to tracking'', Page
#     # % 273, Chapter 6
#     q_t4 = (dt**4/4)
#     q_t3 = (dt**3/2)
#     q_t2 = (dt**2/2)
#
#     # % Approximation of process noise excluding the off-diagonal values
#     # % For the purpose of numerical stability/efficiency
#     Q = [[q_t4, 0,    0,     0,     0, 0],
#          [0,    q_t4, 0,     0,     0, 0],
#          [0,    0,    dt**2, 0,     0, 0],
#          [0,    0,    0,     dt**2, 0, 0],
#          [0,    0,    0,     0,     1, 0],
#          [0,    0,    0,     0,     0, 1]]
#
#
#     # % The measurement matrix or Observation model matrix H.
#     # % The relation b/w the measurement vector and the state vector
#     H = [[1,  0,  0,  0,  0,  0],
#          [0,  1,  0,  0,  0,  0]]
#
#     # % The measurement noise covariance R
#     R = [[v_xm,   0],
#          [0,   v_ym]]
#
#     return Xk, A, Pk, Q, H, R

def measurement_jacob(input_vec, n_fields):
    # % Inputs:
    # %    x - States x[k]
    # %
    # % Outputs:
    # %    dhdx - dh/dx, the Jacobian of citrackMeasurementFcn evaluated at x[k]
    # %
    # % Known anchors Positions in 2D at TWB
    [nAnc, nDim] = (len(Anc_2D), len(Anc_2D[0]))
    state_vec_len = len(input_vec)

    # print(input_vec)
    dhdx = [[0] * state_vec_len]*nAnc
    ri_0 = [0] * nAnc


    for jj in range(nAnc):
        #     % This is the Jacobian Matrix for measurement
        # %     dhdx(jj) = sqrt((Anc_2D(jj, 1) - xk(1)).^2 + (Anc_2D(jj, 2) - xk(2)).^2);
        # res = math.sqrt((Anc_2D[jj][0] - input_vec[0])**2 + (Anc_2D[jj][1] - input_vec[1])**2)
        if n_fields[0][jj]:
            res = 0
        else:
            res = math.sqrt((Anc_2D[jj][0] - input_vec[0][0])**2 + (Anc_2D[jj][1] - input_vec[1][0])**2)
        # ri_0[jj] = res

        dhdx[jj] = [
            (input_vec[0][0] - Anc_2D[jj][0]) / res if res != 0 else res,
            (input_vec[1][0] - Anc_2D[jj][1]) / res if res != 0 else res,
            # (input_vec[0] - Anc_2D[jj][0]) / res,
            # (input_vec[1] - Anc_2D[jj][1]) / res,
            0,
            0,
            0,
            0
        ]
    # print(dhdx)
    return np.array(dhdx)

def measurement_func(input_vec, n_fields):
    to_return = []
    # print(input_vec)
    # for anc_pos in Anc_2D:
    #     if input_vec[0]
    for known, is_nan_field in zip(Anc_2D, n_fields[0]):
        if is_nan_field:
            to_return.append(0)
            continue

        to_return.append(math.sqrt((known[0] - input_vec[0][0])**2 + (known[1] - input_vec[1][0])**2))

    # res = np.array([math.sqrt((Anc_2D[i][0] - input_vec[0][0])**2 + (Anc_2D[i][1] - input_vec[1][0])**2) for i in range(len(Anc_2D))])
    # res = np.array([math.sqrt((Anc_2D[i][0] - input_vec[0])**2 + (Anc_2D[i][1] - input_vec[1])**2) for i in range(len(Anc_2D))])
    res = np.array(to_return).reshape((len(Anc_2D), 1))
    # print(res.shape)
    return res


def state_func(input_vec):
    # print(input_vec)
    A = np.array(
        [[1,   0,   dt,  0,    dt**2/2,    0],
         [0,   1,   0,   dt,   0,          dt**2./2],
         [0,   0,   1,   0,    dt,         0],
         [0,   0,   0,   1,    0,          dt],
         [0,   0,   0,   0,    1,          0],
         [0,   0,   0,   0,    0,          1]])
    # A = [[1,  0,   dt,  0],
    #      [0,  1,   0,   dt],
    #      [0,  0,   1,   0],
    #      [0,  0,   0,   1]]
    # print(A, input_vec)
    return np.matmul(A, input_vec)

def state_func_jacob(input_vec):
    A = np.array(
        [[1,   0,   dt,  0,    dt**2/2,    0],
         [0,   1,   0,   dt,   0,          dt**2./2],
         [0,   0,   1,   0,    dt,         0],
         [0,   0,   0,   1,    0,          dt],
         [0,   0,   0,   0,    1,          0],
         [0,   0,   0,   0,    0,          1]])
    # A = np.array(
    #     [[1,  0,   dt,  0],
    #      [0,  1,   0,   dt],
    #      [0,  0,   1,   0],
    #      [0,  0,   0,   1]])
    return A
dir_fp = f"../results/v2/{data_source}/"
# dir_fp = "../results/v2/2024-07-05 15:02:00.308097/"
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

result = anc0.join(anc1, how='outer').join(anc2, how='outer')
result = result.between_time(*time_filter)

gnss_data = pd.read_csv(dir_fp + "gnss.csv", delimiter=',', names=["dt", "northing", "easting"], dtype={"dt": float, "northing": float, "easting": float})
gnss_data.set_index('dt', inplace=True)
gnss_data.index = pd.to_datetime(gnss_data.index, unit='s').round('50ms')
gnss_data = gnss_data.between_time(*time_filter)
gnss_data["northing"] = gnss_data["northing"] - BASE_STATION_COORDS[0]
gnss_data["easting"] = gnss_data["easting"] - BASE_STATION_COORDS[1]

# print(result)
# %%%%%%%%%%% Initialization of state parameters %%%%%%%%%%%%%
# % For Constant Acceleration Dynamic/Motion Model
# [xk, A, Pk, Q, Hkf, R] = get_const_acceleration_kf()
#
# % Specify an initial guess for the two states
# % initialStateGuess = [2; 1.5; 0; 0];   % the state vector [x, y, vx, vy];#
# % Q_ekf = diag([0.01 0.01 0.01 0.01 0.01 0.01]); % process noise regarding ranges is different from pose data
ekf = ExtendedKalmanFilter(6, 3)

R_ekf = np.diag([0.123, 0.123, 0.123])   # spread of the data directly
ekf.R = R_ekf

F = np.array(
    [[1, 0, dt, 0, dt ** 2 / 2, 0],
     [0, 1, 0, dt, 0, dt ** 2. / 2],
     [0, 0, 1, 0, dt, 0],
     [0, 0, 0, 1, 0, dt],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]])
ekf.F = F

Q_ekf = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # use precision from datasheet directly
ekf.Q = Q_ekf
# print(np.linalg.eig(Q_ekf))
Xk = [[6], [-6], [1.5], [2], [1.0], [1.2]]    # a posteriori
# ekf.x = np.array([2.5, 2.5, 1.5, 2, 1, 1.2])
ekf.x = np.array(Xk)
# print(ekf.x.shape)

# ekf.update(np.array([2, 2, 2]), measurement_jacob, measurement_func)
# ekf.update(np.array([2, 1, 1]), measurement_jacob, measurement_func)

res = []

is_initial = True
prev_data  = np.zeros((1, len(Anc_2D)))

for row in result.iterrows():
# for a0, a1, a2, a3 in zip(r_t2A0, r_t2A1, r_t2A2, r_t2A3):
    # print(row[1:3])
    # print(list(row[1:3]))
    # continue
    d = np.array(row[1:3])
    if np.isnan(d).any() and is_initial:
        continue

    nan_fields = np.isnan(d)
    d[nan_fields] = 0

    prev_data = d
    is_initial = False
    # print(a0, a1, a2, a3)
    ekf.predict_update(d.reshape((3, 1)), measurement_jacob, measurement_func, nan_fields, hx_args=nan_fields)
    # ekf.predict_update(np.array([a0, a1, a2, a3]).reshape((4, 1)), measurement_jacob, measurement_func)
    print(ekf.x, "hi")
    res.append(ekf.x)
    # print(ekf.x)
    # ekf.predict()
    # ekf.update(np.array([2, 2, 2]), measurement_jacob, measurement_func)

# plt.scatter([r[1] for r in results], [r[0] for r in results], c=list(range(len(results))), cmap='Greens')  # copper
plt.figure(figsize=(8, 8))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("UWB-EKF v GNSS", fontsize=14, fontweight='bold')

plt.scatter([r[2] for r in res], [r[0] for r in res], c=list(range(len(res))), cmap='cool')  # cool
plt.scatter(gnss_data["easting"], gnss_data["northing"], c=list(range(gnss_data.shape[0])), cmap="Reds")  # Wistia
# ax.set_aspect('equal')
plt.legend(["UWB-EKF", "GNSS"])
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")

ax = plt.gca()
# ax.tick_params(labelsize=14)

# Add a footnote below and to the right side of the chart
plt.figtext(0.99, 0.01, f"Data: {data_source}, Start: {time_filter[0]}, End: {time_filter[1]} (UTC)", ha="right",
            fontsize=4,
            bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

# plt.plot([r[1] for r in res], [r[0] for r in res], 'o')
plt.show()
#
# %%% Create the extended Kalman filter ekfObject
# %%% Use function handles to provide the state transition and measurement functions to the ekfObject.
# ekfObj = extendedKalmanFilter(@citrackStateFcn,@citrackMeasurementFcn,initialStateGuess);
#
# % Jacobians of the state transition and measurement functions
# ekfObj.StateTransitionJacobianFcn = @citrackStateJacobianFcn;
# ekfObj.MeasurementJacobianFcn = @citrackMeasurementJacobianFcn;
#
# % Measurement noise v[k] and process noise w[k]
# % R_ekf = diag([0.0151 0.0151 0.0151 0.0151]);     % based on the exp data by finding var of the spread
# R_ekf = diag([0.123 0.123 0.123 0.123]);   % spread of the data directly
# ekfObj.MeasurementNoise = R_ekf;
#
# % Q_ekf = diag([0.01 0.01 0.01 0.01 0.01 0.01]); % process noise regarding ranges is different from pose data
# Q_ekf = diag([0.1 0.1 0.1 0.1 0.1 0.1]);  % use precision from datasheet directly
# % Q_ekf = diag(Q);
# ekfObj.ProcessNoise = Q_ekf ;
#
# [Nsteps, n] = size(t2A_4R);
# xCorrectedEKFObj = zeros(Nsteps, length(xk)); % Corrected state estimates
# PCorrectedEKF = zeros(Nsteps, length(xk), length(xk)); % Corrected state estimation error covariances
#
# for k=1 : Nsteps
#
#     % Incorporate the measurements at time k into the state estimates by
#     % using the "correct" command. This updates the State and StateCovariance
#     % properties of the filter to contain x[k|k] and P[k|k]. These values
#     % are also produced as the output of the "correct" command.
# %     [xCorrectedekfObj(k,:), PCorrected(k,:,:)] = correct(ekfObj,yMeas(:, k));
#     [xCorrectedEKFObj(k,:), PCorrectedEKF(k,:,:)] = correct(ekfObj,t2A_4R(k, :));
#
#     % Predict the states at next time step, k+1. This updates the State and
#     % StateCovariance properties of the filter to contain x[k+1|k] and
#     % P[k+1|k]. These will be utilized by the filter at the next time step.
#     predict(ekfObj);
# end