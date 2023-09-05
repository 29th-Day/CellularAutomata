import argparse

import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser(
    prog="Plotter",
    description="Plot google benchmark"
)

parser.add_argument("-f", "--filename", required=True, help="Path to google benchmark csv file")
parser.add_argument("-s", "--size", type=int, default=5, help="Matplotlib scaling factor")
parser.add_argument("-m", "--measures", nargs="+", default=["mean", "median", "stddev", "cv"], help="Available are: mean, median, stddev, cv")

args = parser.parse_args()

with  open(args.filename) as f:
    data = json.load(f)

data["machine"] = {"cpu": data["context"]}

try:
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits", "-i", "0"]).decode().replace("\r", "").replace("\n", "").split(", ")

    data["machine"]["cuda"] = {
        "name": gpu[0],
        "driver": gpu[1],
        "memory": int(gpu[2])
    }
except:
    print("nividia-smi not found")


df = pd.DataFrame(columns=["device", "world_size", "iterations", "measure", "duration", "time_unit"])

for entry in data["benchmarks"]:
    splits = entry["name"].split("/")
    df.loc[len(df)] = [
        splits[0],
        splits[1],
        entry["repetitions"],
        entry["aggregate_name"],
        entry["real_time"],
        entry["time_unit"]
    ]

# print(df)

color = {
    "CPU": "royalblue",
    "CUDA": "forestgreen"
}

def plot_measures(df: pd.DataFrame, measures: list[str]):
    devices = df['device'].unique()

    nrows = len(devices)
    ncols = len(measures)

    time_unit = df["time_unit"].values
    if not all(x==time_unit[0] for x in time_unit):
        raise ValueError("time unit is not always the same!")
    time_unit = time_unit[0]

    _, axs = plt.subplots(nrows, ncols, figsize=(ncols*args.size, nrows*args.size))

    for row, device in enumerate(devices):
        for col, measure in enumerate(measures):
            m = df.loc[(df['measure'] == measure) & (df['device'] == device)]

            world_sizes = m["world_size"].values
            duration = m["duration"].values

            ax: plt.Axes = axs[row, col]

            ax.bar(world_sizes, duration, label=f"{device} time", color=color[device])
            ax.set_title(measure.capitalize())
            
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_xlabel("World size^2")
            ax.set_ylabel(time_unit)
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()

plot_measures(df, args.measures)
