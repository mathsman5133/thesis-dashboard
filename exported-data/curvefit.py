import numpy
import matplotlib.pyplot as plt
import csv
import sys


arg = len(sys.argv) == 2 and sys.argv[1] or "lr-straight"
filename = f"distance-{arg}.csv"
# with open(filename) as f:
#     reader = csv.reader(f)
#     data = list(reader)


from numpy import genfromtxt
df = genfromtxt(filename, delimiter=',')
gnss_data = df[6:7]
# print(my_data)
# print(data)

# numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)

