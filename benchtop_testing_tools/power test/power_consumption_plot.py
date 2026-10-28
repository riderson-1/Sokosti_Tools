#!/usr/bin/env python3
"""Plot battery voltage over time from consumption.csv -> PDF + PNG."""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "consumption.csv")
OUT_PDF = os.path.join(HERE, "voltage_over_time.pdf")
OUT_PNG = os.path.join(HERE, "voltage_over_time.png")

T0 = datetime(2000, 1, 1)  # absolute origin; ticks land on minute boundaries
T0_DAYS = mdates.date2num(T0)


def parse_time(t: str) -> pd.Timedelta:
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return pd.Timedelta(hours=h, minutes=m, seconds=s)


def h_label(x_days, _pos=None) -> str:
    """Format an x position (days since epoch) as whole hours since start."""
    total_min = int(round((x_days - T0_DAYS) * 1440.0))
    return f"{total_min // 60}"


def main() -> None:
    df = pd.read_csv(CSV)
    t = df["Time since start"].map(parse_time)
    x = T0 + t  # datetime axis: ticks land exactly on minute boundaries
    v = df["Voltage"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, v, marker="o", markersize=3.5, linewidth=1.2,
            color="#1f77b4", label="Battery voltage")
    ax.set_xlabel("Time since start (h)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Battery drain test — voltage over time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    # tick + gridline every 30 min: major (labelled) at whole hours,
    # minor (unlabelled) at :30
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(h_label))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    # no rotation — labels stay horizontal
    for label in ax.get_xticklabels():
        label.set_ha("center")

    # y tick every 0.1 V
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    fig.tight_layout()

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
