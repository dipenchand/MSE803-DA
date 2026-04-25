"""Air-quality analysis across Beijing monitoring stations.

Generates summary statistics and chart images used in the slide deck.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from beijing_aqi import load_dataset

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def station_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("station")[POLLUTANTS].mean().round(2)


def composite_rank(means: pd.DataFrame) -> pd.Series:
    # Normalise each pollutant 0-1 then average → composite pollution index.
    norm = (means - means.min()) / (means.max() - means.min())
    return norm.mean(axis=1).sort_values(ascending=False).round(3)


def monthly_pm25(df: pd.DataFrame, station: str) -> pd.Series:
    sub = df[df["station"] == station].copy()
    sub["date"] = pd.to_datetime(sub[["year", "month", "day"]])
    return sub.set_index("date")["PM2.5"].resample("ME").mean()


def main() -> None:
    df = load_dataset()
    means = station_means(df)
    rank = composite_rank(means)
    worst = rank.index[0]
    best = rank.index[-1]

    print("=== Mean pollutant levels by station ===")
    print(means.to_string())
    print("\n=== Composite pollution index (higher = worse) ===")
    print(rank.to_string())
    print(f"\nWORST station: {worst}")
    print(f"BEST station : {best}")

    # --- Chart 1: PM2.5 ranking bar chart ---
    pm = means["PM2.5"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if s == worst else "#2ca02c" if s == best else "#4c72b0"
              for s in pm.index]
    ax.barh(pm.index, pm.values, color=colors)
    ax.set_xlabel("Mean PM2.5 (µg/m³)")
    ax.set_title("Average PM2.5 by Beijing Monitoring Station (2013-2017)")
    for i, v in enumerate(pm.values):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pm25_by_station.png", dpi=150)
    plt.close(fig)

    # --- Chart 2: Composite index ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rank_asc = rank.sort_values()
    colors = ["#d62728" if s == worst else "#2ca02c" if s == best else "#4c72b0"
              for s in rank_asc.index]
    ax.barh(rank_asc.index, rank_asc.values, color=colors)
    ax.set_xlabel("Composite pollution index (0-1)")
    ax.set_title("Overall Air-Quality Ranking — Higher = Worse")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "composite_index.png", dpi=150)
    plt.close(fig)

    # --- Chart 3: Monthly PM2.5 trend for worst vs best ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    monthly_pm25(df, worst).plot(ax=ax, label=f"{worst} (worst)", color="#d62728")
    monthly_pm25(df, best).plot(ax=ax, label=f"{best} (best)", color="#2ca02c")
    ax.set_ylabel("Monthly mean PM2.5 (µg/m³)")
    ax.set_title("PM2.5 Trend: Worst vs Best Station")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trend_worst_vs_best.png", dpi=150)
    plt.close(fig)

    # --- Chart 4: Pollutant profile of worst station ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    worst_vals = means.loc[worst]
    avg_vals = means.mean()
    x = range(len(POLLUTANTS))
    width = 0.38
    ax.bar([i - width / 2 for i in x], worst_vals.values, width,
           label=worst, color="#d62728")
    ax.bar([i + width / 2 for i in x], avg_vals.values, width,
           label="Beijing avg", color="#888888")
    ax.set_xticks(list(x))
    ax.set_xticklabels(POLLUTANTS)
    ax.set_ylabel("Mean concentration")
    ax.set_title(f"Pollutant Profile: {worst} vs Beijing Average")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "worst_profile.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
