import typing

from filterpy.common import Saver
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import least_squares

if typing.TYPE_CHECKING:
    from .collectors import DataCollector, RECENT_DATA_COUNT


class LocalisationAlgorithm:
    least_squares = "Least Squares"
    least_squares_kf = "Least Squares + Kalman Filter"
    extended_kf = "Extended Kalman Filter"

    @staticmethod
    def get_names():
        return [
            LocalisationAlgorithm.least_squares,
            LocalisationAlgorithm.least_squares_kf,
            LocalisationAlgorithm.extended_kf
        ]

class LeastSquaresLocaliser:
    def __init__(self, use_kf=False, initial_pos=(0, 0), data_collector: "DataCollector" = None):
        self.use_kf = use_kf
        self.data_collector = data_collector

        dt = self.data_collector.get_config("dt", 0.05)

        kf = KalmanFilter(dim_x=4, dim_z=2)
        # kf.x = np.array([2.5, 2.5, 1.5, 2]).reshape(4, 1)  # initial guess
        kf.x = np.array([[*initial_pos, 0, 0]]).T
        kf.P = np.eye(4) * 500
        # q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
        # print(q, block_diag(q, q))
        # kf.Q = block_diag(q, q)
        kf.Q = np.diag([dt ** 2 / 2, dt ** 2, 1, 1])
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
        self.kf = kf
        self.last_guess = initial_pos
        self.trilat_results = []
        self.kf_results = []

    def mass_update(self, data):
        anchors = self.data_collector.get_local_anchor_pos()
        feed_to_kf = []
        for row in data:
            res = self.update(row, use_kf=False, anchors=anchors)
            feed_to_kf.append(res)

        if not self.use_kf:
            return

        s = Saver(self.kf)
        self.kf.batch_filter(feed_to_kf, saver=s)
        s.to_array()
        northing, easting = s.x[:, 0].reshape((-1, )), s.x[:, 2].reshape((-1,))
        for i, (n, e) in enumerate(zip(northing, easting)):
            self.kf_results.append((data[i][0], n, e))

    def get_results(self, count=5000, fetch_all=False, between=None, exclude_ts=False):
        if self.use_kf:
            # data = self.kf_results
            data = self.kf_results
        else:
            data = self.trilat_results

        if between is not None:
            return [r[1:3] if exclude_ts else r for r in data if between[0] <= r[0] <= between[1]]

        if fetch_all is False:
            return [r[1:3] if exclude_ts else r for r in data[:-count:-1]] or []
        else:
            return [r[1:3] if exclude_ts else r for r in data]

    @staticmethod
    def _get_equation(known, distance):
        def func(x, y):
            r = (x - known[0]) ** 2 + (y - known[1]) ** 2 - distance ** 2
            return r
        return func

    def update(self, data, use_kf=True, anchors=None):
        # vals = row[1].values
        # is_initial = initial[0] == 0 and initial[1] == 0
        # if is_initial and np.isnan(vals).any():
        #     continue
        # print(vals)
        eqns = []
        if anchors is None:
            anchors = self.data_collector.get_local_anchor_pos()

        for name, anchor_position in anchors.items():
            try:
                name = int(name)
            except ValueError:
                continue

            distance = data[name+1]
            if distance is None or np.isnan(distance):
                continue

            # assume heuristic already applied
            # print(f"distance is {distance}, pos: {anchor_position}")
            eqns.append(self._get_equation(anchor_position, distance))

        def _guess(guess):
            x, y = guess
            return [eqn(x, y) for eqn in eqns]

        result = least_squares(_guess, self.last_guess)
        # print(f"result is {result}")
        # print(result)
        res = result.x
        self.last_guess = res

        self.trilat_results.append((data[0], res[0], res[1]))

        if use_kf is True and self.use_kf:
            self.kf.predict()
            self.kf.update(res)
            to_ret = [self.kf.x[0].reshape((-1,))[0], self.kf.x[2].reshape((-1,))[0]]

            self.kf_results.append((data[0], *to_ret))
            return to_ret
        else:
            return res

    def get_prediction(self):
        return self.kf.x

    def clear(self):
        self.kf_results.clear()
        self.trilat_results.clear()
