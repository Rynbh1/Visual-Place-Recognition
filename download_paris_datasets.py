#!/usr/bin/env python3
"""Telecharge UNIQUEMENT la partie Paris de deux datasets : OSV-5M et MegaScenes.

  - OSV-5M (HuggingFace osv5m/osv5m) : images street-level geolocalisees. On
    filtre par bbox de Paris via train.csv / test.csv, puis on extrait
    SELECTIVEMENT les images dans les zips distants (remotezip, lecture par
    plages HTTP -> on telecharge SEULEMENT les images de Paris, pas les 259 Go).

  - MegaScenes (bucket public s3://megascenes) : images Wikimedia Commons par
    scene, sans coordonnees -> on garde les categories dont le nom contient le
    token "paris" (ex: Notre-Dame_de_Paris). Telechargement HTTP individuel.

Principe IMPORTANT (cache) : determiner la liste des images de Paris demande de
lire de gros fichiers de metadonnees (surtout OSV train.csv ~2,9 Go). On ne le
fait qu'UNE fois : le resultat (liste des ids OSV + cles MegaScenes de Paris)
est ecrit dans `paris_lists/` a cote du script. Aux runs suivants, on lit cette
liste et on saute completement le telechargement des metadonnees.

Deroulement : (1) liste Paris (depuis le cache si present, sinon construite
depuis les metadonnees), (2) annonce taille + duree estimees, (3) confirmation
Y/N, (4) telechargement des images.

Usage :
    python download_paris_datasets.py                 # interactif
    python download_paris_datasets.py --yes           # sans confirmation
    python download_paris_datasets.py --prepare-only  # construit juste le cache
    python download_paris_datasets.py --refresh        # reconstruit le cache
    python download_paris_datasets.py --skip-osv
    python download_paris_datasets.py --skip-megascenes
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

# Paris intra-muros (ouest, sud, est, nord)
PARIS_BBOX = (2.224, 48.815, 2.470, 48.902)

OSV_BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"
OSV_SHARDS = {"train": 98, "test": 5}
OSV_AVG_BYTES = 48 * 1024  # ~48 Ko/image (mesure)

MS_BASE = "https://megascenes.s3.us-west-2.amazonaws.com"
MS_PARQUET_URL = f"{MS_BASE}/metadata/images_index.parquet"
MS_CATEGORIES_URL = f"{MS_BASE}/metadata/categories.json"

ASSUMED_MBPS = 12.0  # debit suppose pour l'estimation de duree (Mo/s)

SCRIPT_DIR = Path(__file__).resolve().parent
LIST_DIR = SCRIPT_DIR / "paris_lists"
OSV_LIST = LIST_DIR / "osv5m_paris.csv"          # id,split,latitude,longitude,city
MS_LIST = LIST_DIR / "megascenes_paris_keys.txt"  # une cle S3 par ligne


def human_bytes(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}"
        n /= 1024


def human_time(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def download_file(url, dest, desc):
    """Telechargement HTTP avec reprise (Range) et barre de progression."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    head = requests.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("content-length", 0))
    if pos and total and pos >= total:
        print(f"  {desc} : deja present ({human_bytes(pos)})")
        return
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if pos else "wb"
        bar = tqdm(total=total or None, initial=pos, unit="B", unit_scale=True,
                   desc=f"  {desc}")
        with dest.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()


# ---------------------------------------------------------------------------
# Construction du cache (metadonnees -> liste Paris)
# ---------------------------------------------------------------------------
def _is_paris_cat(cat, keywords):
    toks = re.split(r"[^a-z0-9]+", cat.lower())
    return any(kw in toks for kw in keywords)


def build_osv_list(meta_dir, bbox):
    """Telecharge train/test.csv, filtre Paris, ecrit OSV_LIST. Renvoie le nb."""
    w, s, e, n = bbox
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "test"):
        download_file(f"{OSV_BASE}/{split}.csv", meta_dir / f"{split}.csv",
                      f"OSV {split}.csv")
        kept = 0
        with (meta_dir / f"{split}.csv").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    lat, lon = float(r["latitude"]), float(r["longitude"])
                except (TypeError, ValueError, KeyError):
                    continue
                if s <= lat <= n and w <= lon <= e:
                    rows.append((r["id"], split, lat, lon, r.get("city", "")))
                    kept += 1
        print(f"  OSV {split} : {kept} images dans la bbox Paris")
    with OSV_LIST.open("w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["id", "split", "latitude", "longitude", "city"])
        wri.writerows(rows)
    print(f"  -> liste OSV ecrite : {OSV_LIST} ({len(rows)} images)")
    return len(rows)


def _ms_scene_path(sid):
    """images/<sid//1000>/<sid%1000>/ (zero-paddes sur 3 chiffres)."""
    return f"images/{sid // 1000:03d}/{sid % 1000:03d}"


def build_ms_list(meta_dir, keywords):
    """Telecharge parquet + categories.json, filtre Paris, resout la cle S3
    complete de chaque image et l'ecrit dans MS_LIST.

    Structure : la categorie mere `cat` a un id de scene (categories.json) qui
    donne le dossier `images/NNN/NNN/` ; la colonne `image` (relative, basee sur
    la sous-categorie) complete le chemin. Les espaces des noms de fichiers
    correspondent a des underscores cote S3 (convention Wikimedia Commons).
    """
    import json
    import pyarrow.parquet as pq
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    download_file(MS_PARQUET_URL, meta_dir / "images_index.parquet",
                  "MegaScenes parquet")
    download_file(MS_CATEGORIES_URL, meta_dir / "categories.json",
                  "MegaScenes categories.json")
    cat2sid = json.loads((meta_dir / "categories.json").read_text(encoding="utf-8"))

    pf = pq.ParquetFile(meta_dir / "images_index.parquet")
    keys, missing = [], 0
    for batch in pf.iter_batches(columns=["cat", "image"], batch_size=200_000):
        cats = batch.column("cat").to_pylist()
        imgs = batch.column("image").to_pylist()
        for c, im in zip(cats, imgs):
            if not (c and im and _is_paris_cat(c, keywords)):
                continue
            sid = cat2sid.get(c)
            if sid is None:
                missing += 1
                continue
            key = f"{_ms_scene_path(sid)}/{im}".replace(" ", "_")
            keys.append(key)
    MS_LIST.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    print(f"  -> liste MegaScenes ecrite : {MS_LIST} ({len(keys)} images"
          f"{f', {missing} categories sans scene ignorees' if missing else ''})")
    return len(keys)


def load_osv_list():
    out = {"train": [], "test": []}
    if not OSV_LIST.exists():
        return out
    with OSV_LIST.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["split"]].append((r["id"], float(r["latitude"]),
                                    float(r["longitude"]), r.get("city", "")))
    return out


def load_ms_list():
    if not MS_LIST.exists():
        return []
    return [l for l in MS_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Telechargement des images
# ---------------------------------------------------------------------------
def _osv_extract_shard(split, k, ids, out_dir):
    from remotezip import RemoteZip
    url = f"{OSV_BASE}/images/{split}/{k:02d}.zip"
    got = 0
    try:
        with RemoteZip(url) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                stem = Path(info.filename).stem
                if stem not in ids:
                    continue
                dest = out_dir / split / f"{stem}.jpg"
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, dest.open("wb") as f:
                    f.write(src.read())
                got += 1
    except Exception as exc:  # noqa: BLE001
        print(f"    shard {split}/{k:02d} ERREUR : {exc}", file=sys.stderr)
    return got


def osv_download(osv, out_dir, workers):
    total = 0
    for split, rows in osv.items():
        ids = {r[0] for r in rows}
        if not ids:
            continue
        nshards = OSV_SHARDS[split]
        print(f"  OSV {split} : {len(ids)} images a extraire de {nshards} zips...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_osv_extract_shard, split, k, ids, out_dir)
                    for k in range(nshards)]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"  OSV {split} zips"):
                total += fut.result()
    print(f"  OSV : {total} images -> {out_dir}")
    return total


def _ms_download_one(key, out_dir):
    dest = out_dir / key
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    try:
        r = requests.get(f"{MS_BASE}/{quote(key, safe='/')}", timeout=120)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return 1
    except requests.RequestException:
        return 0


def ms_estimate_avg(keys, sample=25):
    if not keys:
        return 0
    import random
    sizes = []
    for key in random.sample(keys, min(sample, len(keys))):
        try:
            r = requests.head(f"{MS_BASE}/{quote(key, safe='/')}",
                              allow_redirects=True, timeout=20)
            cl = r.headers.get("content-length")
            if cl:
                sizes.append(int(cl))
        except requests.RequestException:
            pass
    return sum(sizes) / len(sizes) if sizes else 500 * 1024


def ms_download(keys, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ms_download_one, k, out_dir) for k in keys]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  MegaScenes"):
            total += fut.result()
    print(f"  MegaScenes : {total} images -> {out_dir}")
    return total


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Telecharge la partie Paris d'OSV-5M et MegaScenes.")
    ap.add_argument("--dest", type=Path, default=SCRIPT_DIR / "datasets" / "paris_external")
    ap.add_argument("--keywords", default="paris",
                    help="Tokens recherches dans les categories MegaScenes (virgules)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-osv", action="store_true")
    ap.add_argument("--skip-megascenes", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="Reconstruit le cache de liste Paris")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Construit juste la liste Paris (cache), sans telecharger les images")
    ap.add_argument("--yes", action="store_true", help="Pas de confirmation")
    args = ap.parse_args()

    dest = args.dest
    meta_dir = dest / "_metadata"
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]

    # --- Liste Paris (cache si possible) ---
    need_osv = not args.skip_osv and (args.refresh or not OSV_LIST.exists())
    need_ms = not args.skip_megascenes and (args.refresh or not MS_LIST.exists())
    if need_osv or need_ms:
        print("=== Construction de la liste Paris (lecture des metadonnees) ===")
        print("  (une seule fois ; resultat mis en cache dans paris_lists/)\n")
        if need_osv:
            build_osv_list(meta_dir, PARIS_BBOX)
        if need_ms:
            build_ms_list(meta_dir, keywords)
    else:
        print("=== Liste Paris chargee depuis le cache (paris_lists/) ===")

    osv = {} if args.skip_osv else load_osv_list()
    ms_keys = [] if args.skip_megascenes else load_ms_list()

    if args.prepare_only:
        print("\nCache pret. (--prepare-only : pas de telechargement d'images)")
        return 0

    # --- Estimation ---
    osv_n = sum(len(v) for v in osv.values())
    osv_bytes = osv_n * OSV_AVG_BYTES
    ms_n = len(ms_keys)
    ms_avg = ms_estimate_avg(ms_keys) if ms_n else 0
    ms_bytes = int(ms_n * ms_avg)
    total_bytes = osv_bytes + ms_bytes
    n_req = osv_n + ms_n
    dl_time = total_bytes / (ASSUMED_MBPS * 1e6) + n_req * 0.15 / max(1, args.workers)
    if osv_n:
        dl_time += sum(OSV_SHARDS.values()) / max(1, args.workers)  # listing zips

    print("\n" + "-" * 56)
    if not args.skip_osv:
        print(f"  OSV-5M Paris      : {osv_n} images   ~ {human_bytes(osv_bytes)}")
    if not args.skip_megascenes:
        print(f"  MegaScenes Paris  : {ms_n} images   ~ {human_bytes(ms_bytes)} "
              f"(moy. {human_bytes(ms_avg)}/img)")
    print(f"  TOTAL             : {n_req} images  ~ {human_bytes(total_bytes)}")
    print(f"  DUREE estimee     : ~{human_time(dl_time)}  (a ~{ASSUMED_MBPS:.0f} Mo/s)")
    print("-" * 56)

    if n_req == 0:
        print("Rien a telecharger.")
        return 0

    if not args.yes:
        if input("Lancer le telechargement ? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
            print("Annule.")
            return 0

    print("\n=== Telechargement des images ===")
    if osv:
        osv_download(osv, dest / "osv5m", args.workers)
    if ms_keys:
        ms_download(ms_keys, dest / "megascenes", args.workers)
    print(f"\nTermine -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
