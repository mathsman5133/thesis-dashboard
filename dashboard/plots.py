import copy
import math
import statistics

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg


# //            6160189.068799788,
# //            235476.2946930449

colours = {
    "GNSS": "cyan",
    "0": "red",
    "1": "orange",
    "2": "green",
}

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
        self.localised_error = self.plot(name="Localised", pen="cyan", )

        self.plot_curves = {}

        self.uwb_plot = None  # bit hacky, for now...

        self.active_data = None
        self.active_range = None

        self.max_text = None
        self.min_text = None
        self.med_text = None
        self.std_text = None

        self.latest_error = 0

        self.set_text()
        self.sigXRangeChanged.connect(self.on_x_axis_changed)

    def truncate_ts(self, ts_ms, precision=50_000):
        ts = datetime.fromtimestamp(ts_ms)
        increase = ts.microsecond % precision
        nearest = math.floor(ts.microsecond / precision)
        if increase < precision / 2 or nearest >= (1_000_000 / precision - 1):
            ts = ts.replace(microsecond=nearest * precision)
        else:
            ts = ts.replace(microsecond=(nearest + 1) * precision)
        return ts.timestamp()

    def update_plots(self, gps_data, uwb_data, anchor_locations, localised_uwb, base_station_anchor: int):
        gnss = {}
        gnss_times = []
        gnss_dists = []

        # uwb_times = []
        uwb_dists = {}

        # gnss_by_time = {}
        # gnss_uwb_diff = {}
        gnss_loc_by_time = {}

        loc_diff = []
        loc_diff_ts = []

        gnss_by_time = defaultdict(dict)

        for r in gps_data:
            for anchor, loc in anchor_locations.items():
                dist = math.dist(loc, r[1:3])
                ts = self.truncate_ts(r[0])
                gnss[ts] = dist
                gnss_by_time[anchor][ts] = dist

                if str(anchor) == str(base_station_anchor):
                    gnss_dists.append(dist)
                    gnss_times.append(ts)
                    gnss_loc_by_time[ts] = r[1:3]

            # dist = math.dist(location_reference, r[1:3])
            # ts = self.truncate_ts(r[0])

            # gnss_by_time[ts] = dist

        for anchor, data in uwb_data.items():
            anchor = str(anchor)
            res = {}
            # dists = []
            for timestamp, distance in data:
                ts = self.truncate_ts(timestamp)
                # times.append(ts)

                try:
                    res[ts] = distance - gnss_by_time[anchor][ts]
                    # dists.append(row[1] - gnss_by_time[ts])
                    # gnss_uwb_diff[ts] = row[1] - gnss_by_time[ts]
                except KeyError:
                    pass  # if we don't have a GNSS record then just skip it

            # uwb_times[anchor] = res
            uwb_dists[anchor] = res

        base_station_pos = anchor_locations[str(base_station_anchor)]
        for row in localised_uwb:
            ts = self.truncate_ts(row[0])
            try:
                gnss_loc = gnss_loc_by_time[ts]
                gnss_loc = [gnss_loc[0] - base_station_pos[0], gnss_loc[1] - base_station_pos[1]]

                loc_diff.append(math.dist(row[1:3], gnss_loc))
                loc_diff_ts.append(ts)
            except KeyError:
                pass

        self.latest_error = loc_diff[0] if loc_diff else 0

        # if not loc_diff:
        #     x = pd.DataFrame(gnss_uwb_diff.values())
        #     ind = pd.to_datetime(list(gnss_uwb_diff.keys()), unit='s', utc=True)
        # else:
        x = pd.DataFrame(loc_diff)
        ind = pd.to_datetime(loc_diff_ts, unit='s', utc=True)
        # if uwb_dists:
        #     x = pd.DataFrame(list(uwb_dists["0"].values()))
        #     ind = pd.to_datetime(list(uwb_dists["0"].keys()), unit='s', utc=True)
        #     print(x)
        # print(x)
        y_rolling_avg5 = x.rolling(window=5).mean()
        # print(y_rolling_avg5, type(y_rolling_avg5))
        self.active_data = x
        if not self.active_data.empty:
            # print(len(self.active_data), len(gnss_uwb_diff.keys()))
            self.active_data.index = ind

        self.update_text()

        # self.gps_curve.setData(gnss_times, gnss_dists)
        # self.uwb_curve.setData(uwb_times, uwb_dists)
        # print(uwb_dists)
        for anchor, data in uwb_dists.items():
            self.maybe_plot(anchor, list(data.keys()), list(data.values()))

        # if gnss_uwb_diff:
        #     self.uwb_curve.setData(list(gnss_uwb_diff.keys()), list(gnss_uwb_diff.values()))

        if gnss_times:
            self.uwb_plot.update_plots({"GNSS": list(zip(gnss_times, gnss_dists))})

        if loc_diff:
            self.localised_error.setData(loc_diff_ts, loc_diff)
            # self.getViewBox().setXRange(0, 5)
        self.update_text()

    def set_text(self):
        self.min_text = pg.TextItem("Min: {:.2f}m\nMax: 0.00m\nAvg: 0.00m\nStd: 0.00m".format(0), anchor=(-1, -0.1))
        # self.addItem(self.min_text)
        # self.max_text = pg.LabelItem("Max: {:.2f}m".format(0))
        # self.med_text = pg.LabelItem("Avg: {:.2f}m".format(0))
        # self.std_text = pg.LabelItem("Std: {:.2f}m".format(0))

        self.min_text.setParentItem(self.graphicsItem())
        # self.max_text.setParentItem(self.graphicsItem())
        # self.med_text.setParentItem(self.graphicsItem())
        # self.std_text.setParentItem(self.graphicsItem())

        # self.min_text.anchor((1, 0))
        # self.max_text.anchor((0, 0), (0.8, 0.15))
        # self.med_text.anchor((0, 0), (0.8, 0.3))
        # self.std_text.anchor((0, 0), (0.8, 0.45))

    def update_text(self):
        if self.active_range is None or self.active_data is None or self.active_data.empty:
            print("no active data" + str(self.active_data))
            return

        # print(self.active_data)
        data = self.active_data.between_time(*self.active_range)
        if data.empty:
            print("no data")
            return

        min_val = float(data.min().iloc[0])
        max_val = float(data.max().iloc[0])
        med = float(data.mean().iloc[0])
        std = float(data.std().iloc[0])

        # print("updating text: ", min_val, max_val, med, std)

        self.min_text.setText("Min: {:.2f}m\n"
                               "Max: {:.3f}m\n"
                               "Avg: {:.3f}m\n"
                               "Std: {:.3f}m".format(min_val, max_val, med, std))
        # self.max_text.setText("Max: {:.2f}m".format(max_val))
        # self.med_text.setText("Avg: {:.2f}m".format(med))
        # self.std_text.setText("Std: {:.2f}m".format(std))

    def on_x_axis_changed(self):
        new_x = self.viewRange()[0]
        to_set = []
        for dt in new_x:
            minus_10 = datetime.fromtimestamp(dt) - timedelta(hours=10)
            to_set.append(str(minus_10.time()))

        self.active_range = to_set
        # print(self.active_range)
        self.update_text()

    def maybe_plot(self, anchor, x, y):
        try:
            curve = self.plot_curves[anchor]
            curve.setData(x, y)
        except KeyError:
            self.plot_curves[anchor] = self.plot(
                x, y,
                # symbol="o",
                # symbolBrush=colours.get(anchor, pg.intColor(len(self.plot_curves))),
                pen=colours.get(anchor, pg.intColor(len(self.plot_curves))),
                # pen=pg.intColor(len(self.plot_curves)),
                name=f"Anchor {anchor}",
            )


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
        for anchor in self.plot_curves.keys():
            self.plot_data[anchor].append((now, freq_data.get(anchor, 0)))

        # print("freq data: ", freq_data.keys(), self.plot_curves.keys())
        for anchor in set(freq_data.keys()) - set(self.plot_curves.keys()):
            self.plot_data[anchor].append((now, freq_data[anchor]))

    def update_plots(self):
        for anchor, data in self.plot_data.items():
            try:
                curve = self.plot_curves[anchor]
                curve.setData(*zip(*data))
            except KeyError:
                self.plot_curves[anchor] = self.plot(
                    *zip(*data),
                    pen=colours.get(str(anchor), pg.intColor(len(self.plot_curves))),
                    # pen=pg.intColor(len(self.plot_curves) - 1),
                    name=f"Anchor {anchor}",

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
            print(f"updating uwb: {anchor}, {type(anchor)}")
            # print(f"updating uwb: {anchor}, {len(data)}, {data[0]}")
            try:
                curve = self.plot_curves[anchor]
                curve.setData(*zip(*data))
            except KeyError:
                self.plot_curves[anchor] = self.plot(
                    *zip(*data),
                    pen=colours.get(str(anchor), pg.intColor(len(self.plot_curves))),
                    # pen=pg.intColor(len(self.plot_curves)),
                    name=f"Anchor {anchor}",
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
                symbolBrush=pg.intColor(len(self.plot_curves)) if device != "UWB" else "red",
                symbolSize=2 if device in ("GNSS", "UWB") else 10,
                name=str(device),
                symbol="x"
            )
        # self.set_location(device, easting - self.base_easting, northing - self.base_northing)
