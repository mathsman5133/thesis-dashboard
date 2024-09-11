import pandas as pd

data_source = "2024-07-05 15:20:00.725011"

# dir_fp = "../results/v2/2024-07-05 15:02:00.308097/"
dir_fp = f"../results/v2/{data_source}/"
headers = lambda x: ["dt", f"dist{x}"]
dtypes = lambda x: {"dt": float, f"dist{x}": float}
# parse_dates = ["anc1_dt", "anc0_dt", "anc2_dt", "gnss_dt"]
anc0 = pd.read_csv(dir_fp + "uwb-0.csv", delimiter=',', names=headers(0), dtype=dtypes(0))
anc1 = pd.read_csv(dir_fp + "uwb-1.csv", delimiter=',', names=headers(1), dtype=dtypes(1))
anc2 = pd.read_csv(dir_fp + "uwb-2.csv", delimiter=',', names=headers(2), dtype=dtypes(2))

anc0.set_index('dt', inplace=True)
anc1.set_index('dt', inplace=True)
anc2.set_index('dt', inplace=True)

anc0.index = pd.to_datetime(anc0.index, unit='s').round('50ms')
anc1.index = pd.to_datetime(anc1.index, unit='s').round('50ms')
anc2.index = pd.to_datetime(anc2.index, unit='s').round('50ms')

loaded_df = anc0.join(anc1, how='outer').join(anc2, how='outer')
# loaded_df = loaded_df.between_time(*time_filter)
loaded_df = loaded_df.resample(rule='50ms').mean()
loaded_df.set_index(loaded_df.index.values + pd.Timedelta(hours=10), inplace=True)

loaded_df.to_csv(f"{dir_fp}/uwb-combined.csv", header=False)
# loaded_df.to_csv(f"uwb-combined.csv")
