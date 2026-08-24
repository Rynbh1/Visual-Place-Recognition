#!/usr/bin/env python3
"""Utilitaires partages pour charger un dataset au format GSV-Cities
(Dataframes/<Ville>.csv [+ _train/_test] + Images/<Ville>/<...>.jpg),
utilises par MegaLoc_finetuning.py, infer_megaloc.py et test_megaloc.py.
"""
import glob
import os


def discover_csvs(dataset_root, split=None):
    """Repere les CSV d'un dossier au format GSV-Cities.

    split="train" -> prefere <Ville>_train.csv, sinon <Ville>.csv
    split="test"  -> prefere <Ville>_test.csv, sinon <Ville>.csv
    split=None    -> <Ville>.csv uniquement (dataset non splitte)

    Retourne une liste de (csv_path, img_dir).
    """
    dataframes_dir = os.path.join(dataset_root, "Dataframes")
    img_dir = os.path.join(dataset_root, "Images")
    all_csvs = sorted(glob.glob(os.path.join(dataframes_dir, "*.csv")))

    cities = set()
    for path in all_csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_train"):
            cities.add(name[:-6])
        elif name.endswith("_test"):
            cities.add(name[:-5])
        else:
            cities.add(name)

    selected = []
    for city in sorted(cities):
        split_specific = (
            os.path.join(dataframes_dir, f"{city}_{split}.csv") if split else None
        )
        full = os.path.join(dataframes_dir, f"{city}.csv")
        if split_specific and os.path.exists(split_specific):
            selected.append(split_specific)
        elif os.path.exists(full):
            selected.append(full)
    return [(csv_path, img_dir) for csv_path in selected]


def resolve_image_path(img_dir, row):
    """Reconstruit le chemin d'une image depuis sa ligne de CSV.

    Deux conventions de nommage lat/lon coexistent selon la source :
    pleine precision (dataset Kaggle gsv-cities) ou arrondie a 7
    decimales fixes (export Paris, cf. lib/export.py:gsv_name). Un zero
    final peut disparaitre dans la valeur du CSV (48.881378 vs le nom de
    fichier 48.8813780) donc il faut essayer les deux, par ligne.

    Retourne le chemin s'il existe, sinon None.
    """
    base = (
        f"{row['city_id']}_{row['place_id']:07d}_{row['year']:04d}_"
        f"{row['month']:02d}_{row['northdeg']:03d}"
    )
    for lat, lon in ((row["lat"], row["lon"]), (f"{row['lat']:.7f}", f"{row['lon']:.7f}")):
        path = os.path.join(img_dir, row["city_id"], f"{base}_{lat}_{lon}_{row['panoid']}.jpg")
        if os.path.exists(path):
            return path
    return None
