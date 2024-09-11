import csv
import json
import os
import threading

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from .localisation import LocalisationAlgorithm, LeastSquaresLocaliser
from .parsers import decode_gnss_row, decode_uwb_row, decode_combined_uwb_row
from .utils import northing_easting_to_local

DEFAULT_DIRECTORY = Path("results/v2")
RECENT_DATA_COUNT = 5000


class DataCollector:
    def __init__(self):
        self.fp = DEFAULT_DIRECTORY / str(datetime.now().replace(second=0))

        self.uwb_data = defaultdict(list)
        self.gnss_data = []
        self.localised_data = []
        self.combined_uwb = []

        self.config = {}
        self.can_save = True
        self.comp_type = None
        self.localiser: "TrilaterationLocaliser" = None
        self.local_algo = LocalisationAlgorithm.least_squares

        self.last_uwb = {}
        self.last_uwb_lock = threading.Lock()

        self.setup_localisation()

    def record_anchor_location(self, anchor_id, x, y):
        known_points = self.config.get("anchor_locations", {})
        known_points[str(anchor_id)] = (x, y)
        self.config["anchor_locations"] = known_points

    def record_uwb(self, anchor_id, *data, now=None):
        with self.last_uwb_lock:
            self.last_uwb[anchor_id] = data

        now = now or datetime.now().timestamp()
        # print(f"recording uwb {anchor_id} {data}, {now}")
        self.uwb_data[anchor_id].append((now, *data))

    def record_uwb_combined(self, *data, now=None):
        now = now or datetime.now().timestamp()
        self.combined_uwb.append((now, *data))

    def record_gnss(self, *data, now=None):
        now = now or datetime.now().timestamp()
        self.gnss_data.append((now, *data))

    def record_localised_result(self, northing, easting, now=None):
        self.localised_data.append((now or datetime.now().timestamp(), northing, easting))

    def record_config(self, key, value):
        self.config[key] = value

    def get_config(self, key, default: Any = None):
        try:
            return self.config[key]
        except KeyError:
            return default

    def get_local_anchor_pos(self):
        known = self.get_config("anchor_locations", {})
        return {
            anchor: northing_easting_to_local(coord[0], coord[1]) for anchor, coord in known.items()
        }

    def load_config(self, directory):
        fp = directory / "config.json"
        if not fp.exists():
            return

        with open(fp) as f:
            self.config = json.load(f)

    def get_recent_uwb(self, count=RECENT_DATA_COUNT, fetch_all=False) -> Optional[dict[int, list[tuple[float, float]]]]:
        if fetch_all is False:
            return {anchor: data[:-count:-1] for anchor, data in self.uwb_data.items()}
        else:
            return self.uwb_data

    def get_recent_combined_uwb(self, count=RECENT_DATA_COUNT, fetch_all=False, between=None) -> Optional[list[tuple[float, float]]]:
        if between is not None:
            return [r for r in self.combined_uwb if between[0] <= r[0] <= between[1]]

        if fetch_all is False:
            return self.combined_uwb and self.combined_uwb[:-count:-1] or []
        else:
            return self.combined_uwb


    def get_recent_gnss(self, count=RECENT_DATA_COUNT, fetch_all=False, between=None, exclude_ts=False) -> Optional[list[tuple[float, float]]]:
        if between is not None:
            return [r[1:3] if exclude_ts else r for r in self.gnss_data if between[0] <= r[0] <= between[1]]

        if fetch_all is False:
            return [r[1:3] if exclude_ts else r for r in self.gnss_data[:-count:-1]]
        else:
            return [r[1:3] if exclude_ts else r for r in self.gnss_data]

    def get_last_gnss(self) -> Optional[tuple[float, float]]:
        return self.gnss_data and self.gnss_data[-1][1:3] or None

    def get_last_localised(self):
        return self.localised_data and self.localised_data[-1][1:3] or None

    def get_last_uwb_combined(self):
        return self.combined_uwb and self.combined_uwb[-1] or None

    def get_recent_uwb_localised(self, count=RECENT_DATA_COUNT, fetch_all=False, between=None, exclude_ts=False) -> Optional[list[tuple[float, float]]]:
        if not self.localiser:
            return None

        return self.localiser.get_results(count, fetch_all, between, exclude_ts=exclude_ts)

    def load(self):
        if not self.fp.exists():
            return

        self.gnss_data.clear()
        self.uwb_data.clear()
        self.localised_data.clear()

        config_fp = self.fp / "config.json"
        if config_fp.exists():
            with open(config_fp) as f:
                self.config = json.load(f)

        for file in os.listdir(self.fp):
            if file.endswith(".csv"):
                with open(self.fp / file) as f:
                    reader = csv.reader(f, delimiter=",")
                    if file == "gnss.csv":
                        self.gnss_data = [decode_gnss_row(row) for row in reader]
                    elif file == "localised.csv":
                        self.localised_data = [decode_gnss_row(row) for row in reader]
                    elif file == "uwb-combined.csv":
                        self.combined_uwb = [decode_combined_uwb_row(row, comp_type=self.comp_type) for row in reader]
                    else:
                        anchor = int(file.split("-")[1].split(".")[0])
                        self.uwb_data[anchor] = [decode_uwb_row(row, comp_type=self.comp_type) for row in reader]

        # self.mass_update_localisation()

    def on_comp_type_change(self, comp_type):
        print(f"loading... {comp_type}")
        self.comp_type = comp_type

    def on_localisation_algo_change(self, algo):
        if algo not in LocalisationAlgorithm.get_names():
            return

        if algo != self.local_algo:
            self.local_algo = algo
            self.setup_localisation()

        self.mass_update_localisation()

        from __main__ import update_base_rover_diff, update_location, update_text
        update_location(only_last=False)
        update_base_rover_diff()
        update_text()


    def setup_localisation(self):
        if self.local_algo == LocalisationAlgorithm.least_squares:
            localiser = LeastSquaresLocaliser(data_collector=self, use_kf=False)
        elif self.local_algo == LocalisationAlgorithm.least_squares_kf:
            localiser = LeastSquaresLocaliser(data_collector=self, use_kf=True)
        else:
            raise ValueError(f"Unknown localisation algorithm: {self.local_algo}")

        self.localiser = localiser


    def mass_update_localisation(self):
        self.setup_localisation()

        self.localiser.clear()
        self.localiser.mass_update(self.combined_uwb)

    def update_localiser(self, data):
        # self.setup_localisation()
        self.localiser.update(data)

    def clear(self, save_first: bool = True):
        if save_first:
            self.save()

        self.gnss_data.clear()
        self.uwb_data.clear()

    def _write(self, filename, data):
        with open(self.fp / filename, "w", newline="\n") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(data)

    def save(self):
        print("Saving Data!")
        if not (self.gnss_data or self.uwb_data) or not self.can_save:
            return

        if not self.fp.exists():
            self.fp.mkdir(parents=True, exist_ok=True)

        if self.gnss_data:
            self._write("gnss.csv", self.gnss_data)

        if self.localised_data:
            self._write("localised.csv", self.localised_data)

        for anchor in self.uwb_data.keys():
            self._write(f"uwb-{anchor}.csv", self.uwb_data[anchor])

        with open(self.fp / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)
