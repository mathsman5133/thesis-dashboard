from datetime import datetime, timedelta

import numpy.polynomial.polynomial as poly

SR_POLY = poly.Polynomial([0.5282694 ,  0.56712296,  0.28608351, -1.67388201, -0.44511844, 1.07195597], [1.686183, 61.166944])
SR_POLY2 = poly.Polynomial([ 0.51075268, -0.00117379, -0.43347423,  0.54042526,  0.38338424, -0.33350059, -0.14380512], [3.543565, 33.289808])
LR_POLY = poly.Polynomial([1.995755, 0.245262, -0.003054, 0.188594, -0.512554], [3.062803, 128.881626])
LR_POLY2 = poly.Polynomial([1.706673, 0.050369, 0.912991, 0.253587, -1.040104], [2.816559, 187.039706])
LR_POLY3 = poly.Polynomial([1.70667308,  0.05036929,  0.91299066,  0.25358677, -1.04010403], [2.816559, 187.039706])
LR_POLY4_80CM = poly.Polynomial([0.5])

POLY_LOOKUP = {
    "None": lambda x: 0,
    "Short-Range": SR_POLY,
    "Short-Range2": SR_POLY2,
    "Long-Range": LR_POLY,
    "Long-Range2": LR_POLY2,
    "Long-Range3": LR_POLY3,
    "Long-Range-50cm": LR_POLY4_80CM,
}

BASE_STATION_COORDS = (6160188.9135, 235476.0090)


def set_xaxis_timestamp_range(*plots, now=None):
    now = now or datetime.now()
    t_minus_30 = now - timedelta(seconds=30)
    for plot in plots:
        plot.setXRange(t_minus_30.timestamp(), now.timestamp())

def apply_heuristic(value, comp_type):
    if comp_type not in POLY_LOOKUP:
        return value

    return value - POLY_LOOKUP[comp_type](value)

def northing_easting_to_local(northing, easting, base_station_coords=BASE_STATION_COORDS):
    return northing - base_station_coords[0], easting - base_station_coords[1]