import copy
import math

from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import pyqtgraph as pg


class BaseRoverDiffPlot(pg.PlotItem):
    def __init__(self):
        super().__init__(
            name="Base-Rover Distance Error",
            # title="Base-Rover Distance",
            labels={"left": "GNSS-UWB Error (m)"},
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.showGrid(x=True, y=True, alpha=0.3)
        self.addLegend(offset=(-30, 10))
        self.gps_curve = self.plot(name="GNSS", pen="cyan", )
        self.uwb_curve = self.plot(name="UWB", pen="pink")

        self.uwb_plot = None  # bit hacky, for now...

    def truncate_ts(self, ts_ms, precision=50_000):
        ts = datetime.fromtimestamp(ts_ms)
        increase = ts.microsecond % precision
        nearest = math.floor(ts.microsecond / precision)
        if increase < precision / 2 or nearest >= (1_000_000 / precision - 1):
            ts = ts.replace(microsecond=nearest * precision)
        else:
            ts = ts.replace(microsecond=(nearest + 1) * precision)
        return ts.timestamp()

    def update_plots(self, gps_data, uwb_data, location_reference):
        gnss = {}
        gnss_times = []
        gnss_dists = []

        uwb_times = []
        uwb_dists = []

        gnss_by_time = {}
        gnss_uwb_diff = {}
        for r in gps_data:
            dist = math.dist(location_reference, r[1:3])
            ts = self.truncate_ts(r[0])

            gnss_dists.append(dist)
            gnss_times.append(ts)
            gnss_by_time[ts] = dist

        for row in uwb_data:
            ts = self.truncate_ts(row[0])
            uwb_times.append(ts)
            uwb_dists.append(row[1])

            try:
                gnss_uwb_diff[ts] = row[1] - gnss_by_time[ts]
            except KeyError:
                pass  # if we don't have a GNSS record then just skip it

        x = pd.DataFrame(gnss_uwb_diff.values())
        print(x)
        y_rolling_avg5 = x.rolling(window=10).mean()
        print(y_rolling_avg5)

        # self.gps_curve.setData(gnss_times, gnss_dists)
        # self.uwb_curve.setData(uwb_times, uwb_dists)
        if gnss_uwb_diff:
            self.uwb_curve.setData(list(gnss_uwb_diff.keys()), y_rolling_avg5[0])

        if gnss_times:
            self.uwb_plot.update_plots({"GNSS": list(zip(gnss_times, gnss_dists))})



class FrequencyPlot(pg.PlotItem):
    def __init__(self):
        super().__init__(
            name="Frequency",
            # title="Frequency",
            labels={"left": "Frequency (Hz)"},
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.addLegend(offset=(-30, 10))
        self.plot_curves: dict[str, pg.PlotDataItem] = {"GNSS": self.plot(name="GNSS", pen="cyan")}

        self.frequency_data = defaultdict(int)
        self.plot_data = defaultdict(list)

    def record_event(self, anchor_id):
        # print(f"recording, {anchor_id}")
        self.frequency_data[anchor_id] += 1

    def clear_and_track_events(self, now=None):
        freq_data = copy.deepcopy(self.frequency_data)
        self.frequency_data.clear()

        now = now or datetime.now().timestamp()
        for anchor, freq in freq_data.items():
            anchor = f"Anchor {anchor}" if isinstance(anchor, int) else anchor
            self.plot_data[anchor].append((now, freq))

    def update_plots(self):
        for anchor, data in self.plot_data.items():
            try:
                curve = self.plot_curves[anchor]
                curve.setData(*zip(*data))
            except KeyError:
                self.plot_curves[anchor] = self.plot(
                    *zip(*data),
                    pen=pg.intColor(len(self.plot_curves)),
                    name=anchor,

                )

    def load_from_static(self, data):
        for anchor, data in data.items():
            freq = defaultdict(int)
            for row in data:
                freq[datetime.fromtimestamp(row[0]).replace(microsecond=0).timestamp()] += 1

            self.plot_data[anchor] = [(dt, occ) for dt, occ in freq.items()]

    def clear_data(self):
        self.plot_data.clear()
        self.frequency_data.clear()


class UWBPlot(pg.PlotItem):
    def __init__(self):
        super().__init__(
            name="UWB Distance",
            # title="UWB Distance",
            labels={"left": "Distance (m)"},
            axisItems={"bottom": pg.DateAxisItem()},
        )
        self.addLegend(offset=(-30, 10))
        self.plot_curves: dict[str, pg.PlotDataItem] = {}

    def update_plots(self, uwb_data):
        for anchor, data in uwb_data.items():
            try:
                curve = self.plot_curves[anchor]
                curve.setData(*zip(*data))
            except KeyError:
                self.plot_curves[anchor] = self.plot(
                    *zip(*data),
                    pen=pg.intColor(len(self.plot_curves)),
                    name=anchor,
                    # symbol="o"
                )


class LocationPlot(pg.PlotItem):
    def __init__(self, base_northing, base_easting):
        super().__init__(
            name="Location",
            title="Location",
            labels={"left": "Northing", "bottom": "Eastings"},
        )
        self.addLegend(offset=(-30, 10))
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setXRange(-10, 10, padding=0)
        self.setYRange(-10, 10, padding=0)
        self.setAspectLocked()

        # self.gnss_scatter = self.plot(pen="cyan", brush=pg.mkBrush(255, 0, 0, 120), symbol="x", name="GNSS")
        # self.uwb_scatter = self.plot(pen="pink", brush=pg.mkBrush(255, 0, 0, 120), symbol="x", name="UWB")

        self.base_northing = base_northing
        self.base_easting = base_easting
        # self.data_collector = data_collector
        # self.scatter = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(255, 0, 0, 120))
        # self.addItem(self.scatter)
        self.plot_curves: dict[str, pg.PlotDataItem] = {}

    def update_location(self, device, northing, easting, relative: bool = True):
        if isinstance(easting, float):
            easting = [easting - self.base_easting if relative else 0]
            northing = [northing - self.base_northing if relative else 0]
        elif relative:
            easting = [e - self.base_easting for e in easting]
            northing = [n - self.base_northing for n in northing]

        # if relative:
        #     easting = easting - self.base_easting
        #     northing = northing - self.base_northing

        try:
            curve = self.plot_curves[device]
            curve.setData(easting, northing)
        except KeyError:
            self.plot_curves[device] = self.plot(
                easting, northing,
                pen=None,
                symbolBrush=pg.intColor(len(self.plot_curves)),
                symbolSize=2 if device == "GNSS" else 10,
                name=device,
                symbol="x"
            )
        # self.set_location(device, easting - self.base_easting, northing - self.base_northing)
