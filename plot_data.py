import csv
import random
import serial
import struct
import time
import os

from datetime import datetime
from collections import defaultdict
from itertools import count

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

while True:
    try:
        s = serial.Serial('/dev/cu.PL2303G-USBtoUART210', 115200)
        break
    except Exception as e:
        print("Can't connect... cycling", str(e))
        time.sleep(5)

t = time.time()


x_vals = []
y_vals = []

def get_default():
    # return {}
    return {"distance": [], "time": []}

data = defaultdict(get_default)
# df = pd.DataFrame(columns=['Distance', 'Anchor'], index=["Time"])

def animate(i):
    # print(data)
    plt.cla()

    for anchor, vals in data.items():
        plt.plot('time', 'distance', data=vals, label=f"Anchor {anchor}")
    # plt.plot(x, y1, label='Channel 1')
    # plt.plot(x, y2, label='Channel 2')

    plt.legend(loc='upper left')
    plt.tight_layout()
    # df = pd.DataFrame(data.values(), columns=[], index=data.keys())
    # s = pd.Series(df.a, index=df.date)
    # df.plot()


# ani = FuncAnimation(plt.gcf(), animate, interval=100)

# print("here2")

# plt.tight_layout()
# plt.ion()
# plt.show(block=False)

def decode_new(msg):
    split = msg.strip().split(",")
    # if x[0] == "mr":
    #     continue
    if len(split) != 3:
        return

    # print(split)
    # print(x[2])
    try:
        anchor = int(split[1][2:])
        dist = float(split[2][2:])
    except Exception:
        return None

    return anchor, dist

def decode_old(msg):
    split = msg.strip().split(" ")

    if len(split) != 8:
        return

    try:
        anchor = int(split[0][2:], 16)
        dist = int(split[2], 16)/1000
    except Exception:
        return

    return anchor, dist


def read():
    x = s.read_until().decode("utf-8")
    decoded = decode_new(x)
    if not decoded:
        return
    
    anchor, dist = decoded
    # if dist > 1000:
    #     return
    # print(f"{anchor}m, {dist}m, dt: {diff}sec")
    data[anchor]["time"].append(datetime.now())
    data[anchor]["distance"].append(dist)
    # data[datetime.now()] = {"Distance": dist, "Anchor": anchor}
    # df.loc[datetime.now(), df.columns] = dist, anchor
    # data.append({"Time": datetime.now(), "Distance": dist, "Anchor": anchor})
    plt.pause(0.001)


def handle_gps_msg(data):
    # read header in entirety and discard
    header_length = int(s.read().hex(), 16)
    s.read(header_length - 4)

    print("New Reading\n------------")
    # s.read(header_length - 3)

    # print(s.read(25))  # header
    print(f"SOL: {struct.unpack('<i', s.read(4))}")  # sol status
    print(f"PosType: {struct.unpack('<i', s.read(4))}")  # sol status
    print(f"LongZone: {struct.unpack('<i', s.read(4))}")  # sol status
    print(f"LatZone: {str(s.read(4))[2]}")  # sol status
    print(f"N: {struct.unpack('<d', s.read(8))[0]}")  # northing
    print(f"E: {struct.unpack('<d', s.read(8))[0]}")  # easting
    print(f"H: {struct.unpack('<d', s.read(8))[0]}")  # height
    s.read(8)  # discard
    print(f"N_stdev: {struct.unpack('<f', s.read(4))[0]}")
    print(f"E_stdev: {struct.unpack('<f', s.read(4))[0]}")
    print(f"H_stdev: {struct.unpack('<f', s.read(4))[0]}")

    s.read(12)  # discard
    print(f"NumSat: {struct.unpack('<B', s.read(1))[0]}")
    print(f"NumSatUsed: {struct.unpack('<B', s.read(1))[0]}")
    s.read(10)
    print("\n")


def read_gps():
    try:
        prev_msg = []
        while True:
            x = s.read(3).hex()
            if x.lower() == "aa4412":
                handle_gps_msg(prev_msg)

        # d = bytes.fromhex(x.decode("utf-8"))
    except Exception as e:
        print(f"read failed, {e}")
        return


while True:
    try:
        read_gps()
    except KeyboardInterrupt:
        exit()
        with open(f"results/results-{datetime.now().replace(second=0)}.csv", "w", newline="\n") as fp:
            writer = csv.writer(fp, delimiter=",")
            for anchor, vals in data.items():
                for time, dist in zip(vals["time"], vals["distance"]):
                    writer.writerow([anchor, time.timestamp(), dist])
        exit()
