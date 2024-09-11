import csv
import json
import os.path
import threading
import time
from cgi import parse

from datetime import datetime, timedelta
from pathlib import Path

from .collectors import DataCollector
from .parsers import decode_combined_uwb_row
from .plots import FrequencyPlot

DEFAULT_DIRECTORY = Path("results/v2")


class ReplayDriver:
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.is_live = False
        self.stop_threads = False
        self.pause_threads = False

        self.fp: Path = None

        self.tasks: list[threading.Thread] = []
        self.start_time = None

        self.freq_plot: FrequencyPlot = None
        self.latest_time = None

    def _sync_time(self):
        while True:
            time.sleep(0.05)

            if not self.start_time:
                continue
            if not self.latest_time:
                self.latest_time = self.start_time
                continue
            while self.pause_threads is True:
                time.sleep(0.05)

            self.latest_time += timedelta(seconds=0.05)

    def _replay_uwb(self, anchor, data):
        prev_time = None
        for row in data:
            if self.stop_threads:
                return

            ts = float(row.pop(0))
            timestamp = datetime.fromtimestamp(ts)
            if not prev_time:
                prev_time = timestamp
            if self.start_time is None:
                self.start_time = timestamp

            sec = (timestamp - prev_time).total_seconds()
            # print(f"sleeping for {sec} seconds")
            time.sleep(sec)
            if self.stop_threads:
                return

            self.data_collector.record_uwb(anchor, *(float(r) for r in row), now=ts)
            if self.freq_plot:
                self.freq_plot.record_event(anchor)

            prev_time = timestamp

    def _replay_uwb_combined(self, data):
        prev_time = None
        for row in data:
            if self.stop_threads:
                return
            while self.pause_threads is True:
                time.sleep(0.05)

            parsed = decode_combined_uwb_row(row, ts_as_datetime=True, comp_type=self.data_collector.comp_type)
            timestamp = parsed[0]

            if not prev_time:
                prev_time = timestamp
            if self.start_time is None:
                self.start_time = timestamp

            time.sleep((timestamp - prev_time).total_seconds())
            if self.stop_threads:
                return

            # print(f"parsing, {parsed}")
            self.data_collector.record_uwb_combined(*parsed[1:], now=timestamp.timestamp())
            for i, anchor_distance in enumerate(parsed[1:]):
                if anchor_distance is not None:
                    # print(f"recording, {i}, {anchor_distance}")
                    self.data_collector.record_uwb(str(i), anchor_distance, now=timestamp.timestamp())
                    self.freq_plot.record_event(str(i))
            prev_time = timestamp

    def _replay_gnss(self, data):
        prev_time = None
        for row in data:
            if self.stop_threads:
                return
            while self.pause_threads is True:
                time.sleep(0.05)

            ts = float(row.pop(0))
            timestamp = datetime.fromtimestamp(ts)
            if not prev_time:
                prev_time = timestamp
            if self.start_time is None:
                self.start_time = timestamp

            time.sleep((timestamp - prev_time).total_seconds())

            if self.stop_threads:
                return

            self.data_collector.record_gnss(*(float(r) for r in row), now=ts)
            if self.freq_plot:
                self.freq_plot.record_event("GNSS")

            prev_time = timestamp

    def start(self) -> bool:
        if not self.fp:
            print("No file set!")
            return False

        self.data_collector.clear()
        self.freq_plot.clear_data()
        self.tasks.clear()
        self.stop_threads = False

        if os.path.exists(self.fp / "config.json"):
            with open(self.fp / "config.json") as f:
                self.data_collector.config = json.load(f)

        for file in self.fp.iterdir():
            if file.name.endswith(".csv"):
                with open(file) as f:
                    reader = csv.reader(f, delimiter=",")
                    if file.name == "gnss.csv":
                        self.tasks.append(threading.Thread(target=self._replay_gnss, args=(list(reader), )))
                    elif file.name == "localised.csv":
                        pass  # TODO
                    elif file.name == "uwb-combined.csv":
                        self.tasks.append(threading.Thread(target=self._replay_uwb_combined, args=(list(reader), )))
                    # else:
                    #     anchor = int(file.name.split("-")[1].split(".")[0])
                    #     self.tasks.append(threading.Thread(target=self._replay_uwb, args=(anchor, list(reader))))

        self.tasks.append(threading.Thread(target=self._sync_time))

        for task in self.tasks:
            task.start()

        self.data_collector.can_save = False
        self.is_live = True
        return True

    def stop(self) -> bool:
        if not self.is_live:
            return False

        self.stop_threads = True

        self.freq_plot.clear_data()
        self.data_collector.clear()
        self.data_collector.load()
        self.data_collector.can_save = True
        self.is_live = False

        return True

    def load_file(self, folder_name):
        self.fp = DEFAULT_DIRECTORY / folder_name
