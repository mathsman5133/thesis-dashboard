import struct


from datetime import datetime
from .utils import apply_heuristic

WINGELLO_UTM_ZONE = (56, "H")


def decode_uwb(msg):
    split = msg.strip().split(",")
    if len(split) != 4:
        return

    try:
        anchor = int(split[1][2:])
        dist = float(split[2][2:])
        # tof = float(split[3][2:])
    except Exception:
        return None

    return anchor, dist


def unpack_bytes(fmt, data: list[bytes]):
    return struct.unpack(fmt, b"".join(data))[0]


def decode_gnss(header, body):
    """Decode GNSS data from header and body bytes payload.

    In reality, all this does is return nothing and easting. However, the other fields are listed too in case
    we ever want to quickly use them.
    """
    body_data = list(map(int.to_bytes, body))
    # print("body_data", body_data)
    # print("New Reading\n------------")
    # s.read(header_length - 3)

    # print(s.read(25))  # header

    sol = unpack_bytes("<i", body_data[0:4])
    pos_type = unpack_bytes("<i", body_data[4:8])
    long_zone = unpack_bytes("<i", body_data[8:12])
    lat_zone = body_data[12:16]

    northing = unpack_bytes("<d", body_data[16:24])
    easting = unpack_bytes("<d", body_data[24:32])
    height = unpack_bytes("<d", body_data[32:40])

    northing_stdev = unpack_bytes("<f", body_data[48:52])
    easting_stdev = unpack_bytes("<f", body_data[52:56])
    height_stdev = unpack_bytes("<f", body_data[56:60])

    num_sat = unpack_bytes("<B", body_data[72:73])
    num_sat_used = unpack_bytes("<B", body_data[73:74])

    # print(f"PosType: {struct.unpack('<i', s.read(4))}")  # sol status
    # print(f"LongZone: {struct.unpack('<i', s.read(4))}")  # sol status
    # print(f"LatZone: {str(s.read(4))[2]}")  # sol status
    # northing = struct.unpack('<d', s.read(8))[0]
    # easting = struct.unpack('<d', s.read(8))[0]
    # print(f"N: {northing}")  # northing
    # print(f"E: {easting}")  # easting
    # print(f"H: {struct.unpack('<d', s.read(8))[0]}")  # height
    # s.read(8)  # discard
    # print(f"N_stdev: {struct.unpack('<f', s.read(4))[0]}")
    # print(f"E_stdev: {struct.unpack('<f', s.read(4))[0]}")
    # print(f"H_stdev: {struct.unpack('<f', s.read(4))[0]}")
    #
    # s.read(12)  # discard
    # print(f"NumSat: {struct.unpack('<B', s.read(1))[0]}")
    # print(f"NumSatUsed: {struct.unpack('<B', s.read(1))[0]}")
    # s.read(10)
    # print("\n")
    # print(northing, easting, num_sat, num_sat_used)
    return northing, easting


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


def decode_gnss_row(row):
    # ts = datetime.fromtimestamp(row[0]).timestamp()
    try:
        # lat, long = utm.to_latlon(float(row[2]), float(row[1]), *WINGELLO_UTM_ZONE)
        # return float(row[0]), lat, long
        return float(row[0]), float(row[1]), float(row[2])
    except Exception:
        return float(row[0]), 0, 0


def decode_uwb_row(row, comp_type=None):
    return float(row[0]), apply_heuristic(float(row[1]), comp_type)
    # return datetime.fromtimestamp(float(row[0])), float(row[1])
