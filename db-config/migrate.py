import csv
import json

from datetime import datetime
from pathlib import Path

import utm

UWB_RESULTS = Path("../results/v3/uwb_results.csv")
GNSS_RESULTS = Path("../results/v3/gnss_results.csv")
dir_name = Path("../results/v2/")

WINGELLO_ZONE = (56, "H")

# with open(dir_name / "config.json") as f:
#     config = json.load(f)


with open(GNSS_RESULTS, "w") as res:
    writer = csv.writer(res, delimiter=",")
    writer.writerow(["round_id", "tag_id", "lat", "long", "height", "record_ts"])


def transform_dt(dt):
    return datetime.fromtimestamp(float(dt)).strftime("%Y-%m-%d %H:%M:%S.%f")


def handle_uwb(round_id, tag_id, fp):
    anchor_id = int(fp.name.split("-")[1][0])
    with open(fp) as uwb_f:
        reader = csv.reader(uwb_f, delimiter=",")
        rows = list(reader)
        print(rows)

    with open(UWB_RESULTS, "a") as res:
        writer = csv.writer(res, delimiter=",")
        writer.writerows([round_id, anchor_id, tag_id, row[1], transform_dt(row[0])] for row in rows)


def transform_en(easting, northing):
    try:
        return utm.to_latlon(float(easting), float(northing), *WINGELLO_ZONE)
    except Exception as e:
        print(f"failed, {easting}, {northing}")
        return 0, 0


def handle_gnss(round_id, tag_id, fp):
    with open(fp) as gnss_f:
        reader = csv.reader(gnss_f, delimiter=",")
        rows = list(reader)

    if len(rows[0]) != 3:
        print(f"skipping {round_id}")
        return

    with open(GNSS_RESULTS, "a") as res:
        writer = csv.writer(res, delimiter=",")
        writer.writerows([round_id, tag_id, *transform_en(row[2], row[1]), transform_dt(row[0])] for row in rows)


for file in dir_name.rglob("*.csv"):
    # if "uwb" in file.name:
    #     handle_uwb(0, 0, file)

    if "gnss" in file.name:
        handle_gnss(0, 0, file)
