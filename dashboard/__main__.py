import copy
import time
import traceback

import serial

from scipy.optimize import least_squares

import pyqtgraph as pg
# from pyqtgraph.Qt import QtWidgets

from .collectors import DataCollector
from .config import GNSS_SERIAL_PORT, GNSS_SERIAL_BAUD, UWB_SERIAL_PORT, UWB_SERIAL_BAUD
from .main_window import GraphicsLayout
from .parsers import decode_gnss, decode_uwb
from .plots import BaseRoverDiffPlot, FrequencyPlot, LocationPlot, UWBPlot
from .replay import ReplayDriver
from .utils import set_xaxis_timestamp_range

# BASE_STATION_COORDS = (0, 0)
# BASE_STATION_COORDS = (6244942.8409, 333364.3062)  # northing, easting (centre of plot), sydney
BASE_STATION_COORDS = (6160188.9135, 235476.0090)  # northing, easting (centre of plot), wingello
BASE_STATION_ANCHOR = 0  # anchor ID of the UWB anchor at GNSS base station
GNSS_BODY_LENGTH = 84  # 84 byte payload

data_collector = DataCollector()
data_collector.record_config("base_station_coords", BASE_STATION_COORDS)
data_collector.record_config("base_station_anchor", BASE_STATION_ANCHOR)
data_collector.record_config("anchor_locations", {
    "0": (6160189.068799788, 235476.2946930449),
    # "1": (6160162.550425654, 235499.53862634592),  # short range
    "1": (6160105.440872673, 235527.55777198478),
    # "2": (6160162.550425654, 235499.53862634592),  # short range
    "2": (6160007.2800598815, 235462.62533516675)
})
replay_driver = ReplayDriver(data_collector)

window = GraphicsLayout(data_collector, replay_driver)

uwb_plot = UWBPlot()
base_rover_diff_plot = BaseRoverDiffPlot()
base_rover_diff_plot.uwb_plot = uwb_plot

freq_plot = FrequencyPlot()
location_plot = LocationPlot(*data_collector.get_config("base_station_coords"))
window.setup(uwb_plot, base_rover_diff_plot, freq_plot, location_plot)
replay_driver.freq_plot = freq_plot


def _do_update_location(device, data, relative=True):
    if not data:
        return

    if isinstance(data, list):
        location_plot.update_location(device, *zip(*data), relative)
    else:
        northing, easting = data
        location_plot.update_location(device, northing, easting, relative)


def update_location(only_last: bool = True):
    if only_last:
        _do_update_location("GNSS", data_collector.get_last_gnss())
        _do_update_location("UWB", data_collector.get_last_localised())
    else:
        d = data_collector.get_recent_gnss(between=uwb_plot.getViewBox().viewRange()[0])
        _do_update_location("GNSS", d)


def serial_gnss():
    while True:
        try:
            s = serial.Serial(GNSS_SERIAL_PORT, GNSS_SERIAL_BAUD)
            print("GNSS Connected!")
            break
        except Exception as e:
            print("Can't connect to GNSS... cycling", str(e))
            time.sleep(5)

    while True:
        try:
            if window.is_static:
                time.sleep(1)
                continue

            s.read_until(0xAA4412.to_bytes(3, "big"))
            header_length = int(s.read().hex(), 16)
            header = s.read(header_length - 4)
            body = s.read(GNSS_BODY_LENGTH)
            # print(body)
            data_collector.record_gnss(*decode_gnss(header, body))
            freq_plot.record_event("GNSS")
        except IOError:
            pass
        except Exception as e:
            print(f"exec: {e}")


def serial_uwb():
    while True:
        try:
            s = serial.Serial(UWB_SERIAL_PORT, UWB_SERIAL_BAUD)
            print("UWB Connected!")
            break
        except Exception as e:
            print("Can't connect to UWB... cycling", str(e))
            time.sleep(5)

    while True:
        try:
            if window.is_static:
                time.sleep(1)
                continue

            data = s.read_until().decode("utf-8")
            decoded = decode_uwb(data)
            # print(data)
            if not decoded:
                print(f"decoding failed, {data}")
                continue

            anchor, *to_record = decoded
            data_collector.record_uwb(anchor, *to_record)
            freq_plot.record_event(anchor)
        except Exception as e:
            pass
            # print(f"exc: {e}")


def check_frequency():
    while True:
        if window.is_static:
            continue

        time.sleep(1)
        if replay_driver.is_live and replay_driver.latest_time:
            freq_plot.clear_and_track_events(now=replay_driver.latest_time.timestamp())
        else:
            freq_plot.clear_and_track_events()


def uwb_localise():
    initial = (0, 0)
    while True:
        if window.is_static:
            continue

        time.sleep(0.025)
        with data_collector.last_uwb_lock:
            last_uwb = copy.deepcopy(data_collector.last_uwb)
            data_collector.last_uwb.clear()

        if not last_uwb:
            continue

        known_points = data_collector.get_config("anchor_locations", {})
        # x, y = sym.symbols("x,y")
        eqns = []
        for anchor, data in last_uwb.items():
            known = known_points.get(str(anchor))
            if not known:
                continue

            distance = data[0]
            eqns.append(lambda x, y: (x-known[0])**2 + (y-known[1])**2 - distance**2)

        def _guess(guess):
            x, y = guess
            return [eqn(x, y) for eqn in eqns]

        result = least_squares(_guess, initial)
        res = result.x
        data_collector.record_localised_result(*res)
        initial = res


def update_plots(from_static=False):
    try:
        if window.is_static and not from_static:
            return

        base_station_anchor = data_collector.get_config("base_station_anchor")
        base_station_coords = data_collector.get_config("base_station_coords")
        recent_uwb = data_collector.get_recent_uwb(fetch_all=from_static)
        uwb_base = recent_uwb.get(base_station_anchor, [])

        uwb_plot.update_plots(recent_uwb)

        recent_gnss = data_collector.get_recent_gnss(fetch_all=from_static)

        base_rover_diff_plot.update_plots(
            recent_gnss,
            uwb_base,
            base_station_coords
        )

        if from_static:
            freq_plot.autoRange()
            uwb_plot.autoRange()
            base_rover_diff_plot.autoRange()
            freq_plot.load_from_static(recent_uwb)
            freq_plot.load_from_static({"GNSS": recent_gnss})

        freq_plot.update_plots()

        if data_collector.get_config("scrolling_timestamp", True) is True and not from_static:
            if replay_driver.is_live and replay_driver.latest_time:
                set_xaxis_timestamp_range(freq_plot, now=replay_driver.latest_time)
            else:
                set_xaxis_timestamp_range(freq_plot)

        update_location(only_last=not from_static)

    except Exception as e:
        print(f"exc: {e}")
        traceback.print_exc()


window.plot_update_func = update_plots
synced_plots = [uwb_plot, base_rover_diff_plot, freq_plot]


def on_new_range_set(box: pg.ViewBox):
    new_range = box.viewRange()
    for p in synced_plots:
        if p.getViewBox() != box:
            p.blockSignals(True)
            p.setXRange(*new_range[0])
            p.blockSignals(False)

    # update values in main location plot to reflect new time range
    update_location(only_last=False)


for plot in synced_plots:
    plot.sigRangeChanged.connect(on_new_range_set)


timer = pg.QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(50)

pg.exec()

data_collector.save()
