import csv

from collections import defaultdict
from pathlib import Path

fp = Path("results/")
output_base = Path("results/v2")

for file in fp.iterdir():
    if file.is_file():
        output = output_base / file.name.replace(".csv", "").replace("results-", "")
        with open(file) as f:
            reader = csv.reader(f)
            data = list(reader)
            by_anchor = defaultdict(list)
            for row in data:
                anchor = row.pop(0)
                by_anchor[anchor].append(row)

            for anchor, anchor_data in by_anchor.items():
                output.mkdir(parents=True, exist_ok=True)
                with open(output / f"uwb-{anchor}.csv", "w") as output_file:
                    writer = csv.writer(output_file)
                    writer.writerows(anchor_data)
