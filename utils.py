from datetime import datetime, timedelta


def set_xaxis_timestamp_range(*plots, now=None):
    now = now or datetime.now()
    t_minus_30 = now - timedelta(seconds=30)
    for plot in plots:
        plot.setXRange(t_minus_30.timestamp(), now.timestamp())
