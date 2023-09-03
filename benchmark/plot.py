import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="Plotter",
    description="Plot google benchmark"
)

parser.add_argument("-f", "--filename", required=True, help="Path to google benchmark csv file")
parser.add_argument("-s", "--size", type=int, default=5, help="Matplotlib scaling factor")
parser.add_argument("-m", "--measures", nargs="+", default=["mean", "median", "stddev", "cv"], help="Available are: mean, median, stddev, cv")

args = parser.parse_args()

df = pd.read_csv(args.filename)

# print(df)

pattern = r'Epoch/(\d+)/repeats:(\d+)_([a-zA-Z]+)'

world_sizes = []
measures = []

for _, row, in df.iterrows():
    match = re.search(pattern, row['name'])

    if match:
        world_sizes.append(match.group(1))
        measures.append(match.group(3))
    else:
        raise ValueError("Regex pattern different than expected")

df["world_size"] = world_sizes
df["measure"] = measures

df.drop(columns=['name'], inplace=True)



def plot_measures(measures: list[str]):

    axs: list[plt.Axes]
    plots = len(measures)
    _, axs = plt.subplots(1, plots, figsize=(plots*args.size, args.size))

    for index, measure in enumerate(measures):
        m = df.loc[df['measure'] == measure]

        world_sizes = m["world_size"].values
        cpu_times = m["cpu_time"].values
        real_times = m["real_time"].values

        time_unit = m["time_unit"].values
        if not all(x==time_unit[0] for x in time_unit):
            raise ValueError("time unit is not always the same!")
        time_unit = time_unit[0]

        axs[index].plot(world_sizes, cpu_times, label="CPU")
        axs[index].plot(world_sizes, real_times, label="Real")
        axs[index].set_title(measure.capitalize())
        axs[index].tick_params(axis='x', labelrotation=45)
        axs[index].set_xlabel("World size^2")
        axs[index].set_ylabel(time_unit)
        axs[index].legend()
        axs[index].grid()

    plt.tight_layout()
    plt.show()

plot_measures(args.measures)
