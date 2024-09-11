import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtWidgets import QLabel

from .collectors import DataCollector
from .localisation import LocalisationAlgorithm
from .plots import LocationPlot, UWBPlot
from .replay import ReplayDriver
from .utils import POLY_LOOKUP, northing_easting_to_local


class LoadDataPopup(pg.GraphicsLayoutWidget):
    def __init__(self, data_collector: DataCollector):
        super().__init__(
            show=True,
            title="Load Data",
            size=(500, 300),
        )
        self.data_collector = data_collector

    def setup(self):
        self.load_button = self.create_button("Load Data", self.on_load)
        self.cancel_button = self.create_button("Cancel", self.on_cancel)

        self.addItem(self.load_button, 0, 0)
        self.addItem(self.cancel_button, 1, 0)

    def add_select_menu(self):
        combobox1 = pg.QtWidgets.QComboBox()
        for directory in os.listdir("results/v2"):
            combobox1.addItem(directory)

    def create_button(self, name, callback):
        btn = QtWidgets.Q(name)
        btn.clicked.connect(callback)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(btn)
        return proxy

    def on_load(self):
        ...

    def on_cancel(self):
        self.close()

class GraphicsLayout(pg.GraphicsLayoutWidget):
    def __init__(self, data_collector: DataCollector, replay_driver: ReplayDriver):
        super().__init__(
            show=True,
            title="UWB-LoRa Tracking Dashboard",
            size=(1000, 600),
        )
        self.data_collector = data_collector
        self.replay_driver = replay_driver
        self.location_plot: LocationPlot = None
        self.uwb_plot: UWBPlot = None
        self.is_static = False

    def setup(self, uwb_plot, base_rover_diff_plot, freq_plot, location_plot):
        self.location_plot = location_plot
        self.uwb_plot = uwb_plot

        # uwb_plot.getAxis('bottom').setTickSpacing(levels=[(3600, 0), (60, 0), (5, 0)])
        # base_rover_diff_plot.getAxis('bottom').setTickSpacing(levels=[(3600, 0), (60, 0), (5, 0)])
        # freq_plot.getAxis('bottom').setTickSpacing(levels=[(3600, 0), (60, 0), (5, 0)])

        self.addItem(uwb_plot, 0, 0)
        self.addItem(base_rover_diff_plot, 1, 0)
        self.addItem(freq_plot, 2, 0)
        self.addItem(location_plot, 1, 1, rowspan=2)

        info_layout: pg.GraphicsLayout = self.addLayout(0, 1, colspan=1)

        # font = QtGui.QFont("Times", 10)
        # font = QFont("Helvetica [Cronyx]", 12)


        buttons = info_layout.addLayout(0, 2, colspan=1)
        self.save_button = self.create_button("Save Data", self.on_save)
        self.play_pause_button = self.create_button("Pause", self.on_play_pause)
        self.sync_anchor_button = self.create_button("Sync Anchor", self.on_anchor_sync)
        self.algorithm_select = self.create_select_menu("Localisation Algorithm", LocalisationAlgorithm.get_names(), self.on_algorithm_select)

        self.load_data_button = self.create_button("Load Data", self.on_load_data)
        self.replay_data_button = self.create_button("Replay Data", self.on_replay_data)
        self.load_config_button = self.create_button("Load Config", self.on_load_config)
        self.load_data_menu = self.create_select_menu("Data File", sorted(os.listdir("results/v2")), self.on_file_select)

        self.exit_button = self.create_button("Exit", self.on_exit)
        self.comp_type = self.create_select_menu("Compensation Type", POLY_LOOKUP.keys(), self.on_comp_type)
        self.relocal_button = self.create_button("Relocalise", self.on_relocalise)

        for i, button in enumerate((self.save_button, self.play_pause_button, self.exit_button)):
            buttons.addItem(button, i, 0)

        # buttons.addItem(self.create_line_entry(), 4, 0)
        buttons.addItem(self.load_data_button, 8, 0)
        buttons.addItem(self.replay_data_button, 9, 0)
        buttons.addItem(self.relocal_button, 10, 0)

        text_layout = buttons.addLayout(11, 0)
        anc_text = text_layout.addLabel("Anchors", 0, 0, size="8pt")
        error_text = text_layout.addLabel("Error", 0, 1, size="8pt")
        speed_text = text_layout.addLabel("Speed", 0, 2, size="8pt")
        self.anc_text = text_layout.addLabel("0", 1, 0, size="16pt")
        self.error_text = text_layout.addLabel("0m", 1, 1, size="16pt")
        self.speed_text = text_layout.addLabel("0km/h", 1, 2, size="16pt")


        buttons.addItem(self.load_data_menu, 7, 0)
        buttons.addItem(self.comp_type, 6, 0)
        buttons.addItem(self.algorithm_select, 5, 0)

        # buttons.addItem(pg.FeedbackButton("Test", self.on_load_data))

        # uwb_count = text_layout.addLabel("0", 1, 0)
        # # uwb_count.setFont(font)
        # fix_perc = text_layout.addLabel("0%", 1, 1)
        #
        # uwb_err = text_layout.addLabel("+/- 0cm", 2, 0)
        # fix_err = text_layout.addLabel("+/- 0cm", 2, 1)

        for anchor, location in self.data_collector.get_config("anchor_locations", {}).items():
            self.location_plot.update_location(f"Anchor {anchor}", *location)


    def create_line_entry(self):
        e1 = QtWidgets.QLineEdit()
        e1.setFont(pg.QtGui.QFont("Arial", 20))
        e1.textChanged.connect(self.on_metadata_change)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(e1)
        return proxy

    def create_select_menu(self, name, items, callback):
        menu = QtWidgets.QComboBox()
        # menu.addItem("")
        menu.addItems(items)
        menu.currentTextChanged.connect(callback)
        menu.setEditable(True)
        menu.setPlaceholderText(name)
        menu.setCurrentIndex(-1)
        menu.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        # change completion mode of the default completer from InlineCompletion to PopupCompletion
        menu.completer().setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(menu)
        return proxy

    def create_button(self, name, callback, cls: QtWidgets.QAbstractButton = QtWidgets.QPushButton):
        btn = cls(name)
        btn.clicked.connect(callback)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(btn)
        return proxy

    def on_save(self):
        self.data_collector.save()

    def on_play_pause(self):
        curr = self.data_collector.get_config("scrolling_timestamp", True)
        if curr is True:
            self.data_collector.record_config("scrolling_timestamp", False)
            self.play_pause_button.widget().setText("Play")
            self.replay_driver.pause_threads = True
        else:
            self.data_collector.record_config("scrolling_timestamp", True)
            self.play_pause_button.widget().setText("Pause")
            self.replay_driver.pause_threads = False

    def on_load_data(self):
        # if self.is_static:
        #     self.is_static = False
        #     self.load_data_button.widget().setText("Load Data")
        # else:
        self.is_static = True
        self.data_collector.can_save = False
        # self.load_data_button.widget().setText("Exit Static")
        self.data_collector.fp = self.replay_driver.fp
        self.data_collector.load()

        for anchor, location in self.data_collector.get_config("anchor_locations", {}).items():
            self.location_plot.update_location(f"Anchor {anchor}", *location)

        self.plot_update_func(from_static=True)

    def on_replay_data(self):
        if self.replay_driver.is_live:
            res = self.replay_driver.stop()
            if res is True:
                self.load_data_button.widget().setText("Load Data")
                self.save_button.setEnabled(True)
        else:
            res = self.replay_driver.start()
            if res is True:
                self.load_data_button.widget().setText("Load Live")
                self.save_button.setEnabled(False)

    def on_relocalise(self):
        print("Doing re-localisation", len(self.data_collector.localiser.trilat_results))
        self.data_collector.localiser.last_guess = (0, 0)
        self.data_collector.localiser.trilat_results.clear()
        self.data_collector.localiser.kf_results.clear()

        gnss_data = self.data_collector.get_recent_gnss(between=self.uwb_plot.getViewBox().viewRange()[0], exclude_ts=True)
        d = self.data_collector.localiser.last_guess = northing_easting_to_local(gnss_data[0][0], gnss_data[0][1])
        print(f"Initial guess: {d}, {d[0] - 6160188.9135}, {d[1] - 235476.0090}")

        data = self.data_collector.get_recent_combined_uwb(between=self.uwb_plot.getViewBox().viewRange()[0])
        self.data_collector.localiser.mass_update(data)

        from __main__ import update_plots, update_base_rover_diff, update_location, update_text
        update_location(only_last=False)
        update_plots()
        update_base_rover_diff()
        update_text()
        print("Done re-localisation", len(self.data_collector.localiser.trilat_results))

    def on_file_select(self, file_name):
        print(f"loading... {file_name}")
        self.replay_driver.load_file(file_name)

    def on_load_config(self):
        print(f"loading... {self.replay_driver.fp}")
        self.data_collector.load_config(self.replay_driver.fp)

        for anchor, location in self.data_collector.get_config("anchor_locations", {}).items():
            self.location_plot.update_location(f"Anchor {anchor}", *location)

        base_station_coords = self.data_collector.get_config("base_station_coords")
        self.location_plot.base_northing = base_station_coords[0]
        self.location_plot.base_easting = base_station_coords[1]

    def on_algorithm_select(self, algorithm):
        print(f"Selecting {algorithm}...")
        self.data_collector.on_localisation_algo_change(algorithm)

    def on_comp_type(self, selected):
        self.data_collector.on_comp_type_change(selected)
        if self.is_static:
            self.on_load_data()

    def on_anchor_sync(self):
        anchor_id = None
        anchor_dist = None
        for anchor, data in self.data_collector.get_recent_uwb(5).items():
            if anchor_id is None or data[0][1] < anchor_dist:
                anchor_id = anchor
                # print(data)
                anchor_dist = data[0][1]

        if anchor_id is None:
            return

        gnss_loc = self.data_collector.get_last_gnss()
        self.data_collector.record_anchor_location(anchor_id, *gnss_loc)
        self.location_plot.update_location(f"Anchor {anchor_id}", *gnss_loc)
        print("Syncing anchor...")

    def on_metadata_change(self, new_value: str):
        print(f"Setting metadata: {new_value}")
        self.data_collector.record_config("metadata", new_value)

    def on_exit(self):
        self.data_collector.save()
        self.close()
        pg.exit()
        exit(0)
