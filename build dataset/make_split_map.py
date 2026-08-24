#!/usr/bin/env python3
"""
Render an interactive HTML map of the GSV-Cities train/test split.

One marker per place_id (its first row's lat/lon), coloured by split, with a
layer control so each split can be toggled independently — useful to see the
actual shape of the geographic block split (see build dataset/lib/export.py)
rather than just trusting the row counts.

Usage:
    python "build dataset/make_split_map.py" \
        --dataset-root "/media/rayan/usb/VPR Dataset/paris/gsv_cities" \
        --output train_test_split_map.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd


def load_places(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"panoid": str})
    # One row per place_id (its first view is enough to plot the place's location).
    return df.groupby("place_id", as_index=False).first()[["place_id", "lat", "lon"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map the GSV-Cities train/test split")
    parser.add_argument("--dataset-root", type=str,
                        default="/media/rayan/usb/VPR Dataset/paris/gsv_cities",
                        help="GSV-Cities dataset root (contains Dataframes/*.csv)")
    parser.add_argument("--city", type=str, default="Paris",
                        help="City CSV stem, e.g. Paris -> Paris_train.csv / Paris_test.csv")
    parser.add_argument("--output", type=str, default="train_test_split_map.html",
                        help="Output HTML path")
    args = parser.parse_args()

    df_dir = Path(args.dataset_root) / "Dataframes"
    train_csv = df_dir / f"{args.city}_train.csv"
    test_csv = df_dir / f"{args.city}_test.csv"
    for p in (train_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    train = load_places(train_csv)
    test = load_places(test_csv)
    print(f"train: {len(train)} places | test: {len(test)} places "
          f"| overlap: {len(set(train.place_id) & set(test.place_id))}")

    all_lat = pd.concat([train["lat"], test["lat"]])
    all_lon = pd.concat([train["lon"], test["lon"]])
    center = (float(all_lat.mean()), float(all_lon.mean()))

    fmap = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    def add_layer(df: pd.DataFrame, name: str, color: str) -> None:
        group = folium.FeatureGroup(name=f"{name} ({len(df)} places)")
        for row in df.itertuples():
            folium.CircleMarker(
                location=(row.lat, row.lon),
                radius=2,
                color=color,
                weight=0,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
            ).add_to(group)
        group.add_to(fmap)

    # Train first so test (fewer, more interesting to spot) draws on top.
    add_layer(train, "Train", "#3388ff")
    add_layer(test, "Test", "#e6194b")

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[all_lat.min(), all_lon.min()], [all_lat.max(), all_lon.max()]],
                    padding=(20, 20))

    fmap.save(args.output)
    print(f"Saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
