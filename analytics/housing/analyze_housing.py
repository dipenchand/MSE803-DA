from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Housing.xlsx"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "Housing.xlsx"


def money_millions(value: float) -> float:
    return round(value / 1_000_000, 2)


def build_initial_story(df: pd.DataFrame) -> dict:
    amenity_premiums = {}
    for feature in ["mainroad", "airconditioning", "prefarea"]:
        grouped = df.groupby(feature)["price"].mean()
        premium = (grouped["yes"] / grouped.drop("yes").iloc[0] - 1) * 100
        amenity_premiums[feature] = round(premium, 1)

    area_groups = pd.qcut(
        df["area"],
        4,
        labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"],
        duplicates="drop",
    )
    area_means = df.groupby(area_groups, observed=False)["price"].mean()

    return {
        "rows": int(len(df)),
        "median_price_m": money_millions(df["price"].median()),
        "top_quartile_starts_m": money_millions(df["price"].quantile(0.75)),
        "area_q1_mean_m": money_millions(area_means.iloc[0]),
        "area_q4_mean_m": money_millions(area_means.iloc[-1]),
        "amenity_premiums": amenity_premiums,
    }


def main() -> None:
    df = pd.read_excel(DATA_PATH)
    summary = build_initial_story(df)
    print("Simple Housing Price Story")
    print(f"- Total homes: {summary['rows']}")
    print(f"- Typical price: {summary['median_price_m']:.2f}M")
    print(f"- Higher-price homes start around: {summary['top_quartile_starts_m']:.2f}M")
    print(
        f"- Average price (smaller vs larger homes): "
        f"{summary['area_q1_mean_m']:.2f}M vs {summary['area_q4_mean_m']:.2f}M"
    )
    print(
        f"- Price lift with main-road access: +{summary['amenity_premiums']['mainroad']:.1f}%"
    )
    print(
        f"- Price lift with air conditioning: +{summary['amenity_premiums']['airconditioning']:.1f}%"
    )
    print("- Overall story: size and convenience features drive higher prices")


if __name__ == "__main__":
    main()
