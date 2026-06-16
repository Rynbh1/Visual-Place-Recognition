import math
import re
import csv
import json
import shutil
import datetime as dt
from pathlib import Path
from lib.layout import meters

def year_month(captured_at):
    """(year, month) a partir d'un timestamp Mapillary (ms epoch). (0,0) si absent."""
    try:
        d = dt.datetime.fromtimestamp(int(captured_at) / 1000, dt.timezone.utc)
        return d.year, d.month
    except (TypeError, ValueError, OSError):
        return 0, 0


def gsv_name(city_id, place_int, year, month, north, lat, lon, panoid):
    """Nom de fichier au format GSV-Cities (lu par le dataloader de MegaLoc)."""
    return (f"{city_id}_{place_int:07d}_{year:04d}_{month:02d}_{north:03d}_"
            f"{lat:.7f}_{lon:.7f}_{panoid}.jpg")


def assign_split(places, block_m, test_ratio, sep_m):
    """Repartit les lieux en train/test par blocs geographiques, avec une zone
    tampon : tout lieu test a moins de `sep_m` d'un lieu train est ECARTE
    (etiquette None) pour eviter le chevauchement visuel (anti-fuite).

    places : dict place_int -> (lat, lon). Renvoie dict place_int -> 'train'|'test'|None.
    """
    if not places:
        return {}
    lat0 = sum(la for la, _ in places.values()) / len(places)
    cos0 = math.cos(math.radians(lat0))

    def block(lat, lon):
        x = lon * 111_320 * cos0
        y = lat * 111_320
        return (int(x // block_m), int(y // block_m))

    # lieux regroupes par bloc
    by_block = {}
    for pid, (lat, lon) in places.items():
        by_block.setdefault(block(lat, lon), []).append(pid)

    # on ordonne les blocs de maniere deterministe (hash) puis on en bascule en
    # test jusqu'a atteindre ~test_ratio des lieux -> ratio fiable meme avec peu
    # de blocs (sinon un tirage par bloc est tres instable sur une petite zone).
    ordered = sorted(by_block, key=lambda b: (b[0] * 73856093) ^ (b[1] * 19349663))
    target = test_ratio * len(places)
    label = {pid: "train" for pid in places}
    n_test = 0
    for b in ordered:
        if n_test >= target:
            break
        for pid in by_block[b]:
            label[pid] = "test"
        n_test += len(by_block[b])

    # index spatial des lieux train (cellules de sep_m) pour le tampon
    if sep_m > 0:
        buckets = {}
        for pid, (lat, lon) in places.items():
            if label[pid] != "train":
                continue
            cx = int(lon * 111_320 * cos0 // sep_m)
            cy = int(lat * 111_320 // sep_m)
            buckets.setdefault((cx, cy), []).append((lat, lon))
        for pid, (lat, lon) in places.items():
            if label[pid] != "test":
                continue
            cx = int(lon * 111_320 * cos0 // sep_m)
            cy = int(lat * 111_320 // sep_m)
            near = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (tla, tlo) in buckets.get((cx + dx, cy + dy), ()):
                        if meters(lon, lat, tlo, tla) < sep_m:
                            near = True
                            break
                    if near:
                        break
                if near:
                    break
            if near:
                label[pid] = None  # ecarte (zone tampon)
    return label


def cmd_export(args):
    src = Path(args.dataset)
    manifest_path = src / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl introuvable dans {src}")
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("Manifeste vide.")

    # city_id : alphanumerique sans underscore (separateur du format GSV-Cities)
    raw_city = args.city_id or src.name
    city_id = re.sub(r"[^A-Za-z0-9]", "", raw_city.title()) or "City"

    # centre GPS par lieu (moyenne des images du lieu)
    places_pts = {}
    for r in rows:
        pid = int(str(r["place"]).split("_")[-1])
        places_pts.setdefault(pid, []).append((r["lat"], r["lon"]))
    # MegaLoc/GSV-Cities echantillonne plusieurs images par lieu : on ecarte
    # les lieux avec trop peu d'images (impossibles a former en positifs).
    too_small = sum(1 for pts in places_pts.values() if len(pts) < args.min_imgs)
    places = {pid: (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
              for pid, pts in places_pts.items() if len(pts) >= args.min_imgs}

    label = assign_split(places, args.test_block, args.test_ratio, args.separation)

    out = Path(args.out) if args.out else src / "gsv_cities"
    img_dir = out / "Images" / city_id
    df_dir = out / "Dataframes"
    img_dir.mkdir(parents=True, exist_ok=True)
    df_dir.mkdir(parents=True, exist_ok=True)

    cols = ["place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"]
    df_rows, copied, skipped = [], 0, 0
    for r in rows:
        pid = int(str(r["place"]).split("_")[-1])
        if label.get(pid) is None:
            skipped += 1
            continue
        lat = round(float(places[pid][0]), 7)
        lon = round(float(places[pid][1]), 7)
        north = int(round(r.get("target_bearing") or r.get("compass_angle") or 0)) % 360
        panoid = re.sub(r"[^A-Za-z0-9]", "", str(r["id"]))
        y, m = year_month(r.get("captured_at"))
        fname = gsv_name(city_id, pid, y, m, north, lat, lon, panoid)
        src_img = src / r["file"]
        if not src_img.exists():
            skipped += 1
            continue
        shutil.copy2(src_img, img_dir / fname)
        copied += 1
        df_rows.append({
            "place_id": pid, "year": y, "month": m, "northdeg": north,
            "city_id": city_id, "lat": lat, "lon": lon, "panoid": panoid,
            "split": label[pid],
        })

    def write_csv(path, items):
        with path.open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wri.writeheader()
            wri.writerows(items)

    write_csv(df_dir / f"{city_id}.csv", df_rows)
    train = [r for r in df_rows if r["split"] == "train"]
    test = [r for r in df_rows if r["split"] == "test"]
    write_csv(df_dir / f"{city_id}_train.csv", train)
    write_csv(df_dir / f"{city_id}_test.csv", test)

    n_pl = len({r["place_id"] for r in df_rows})
    n_pl_tr = len({r["place_id"] for r in train})
    n_pl_te = len({r["place_id"] for r in test})
    print("\n" + "=" * 60)
    print(f"  EXPORT GSV-Cities -> {out}")
    print("=" * 60)
    print(f"  Ville (city_id)     : {city_id}")
    print(f"  Images copiees      : {copied}  (ecartees : {skipped})")
    print(f"  Lieux < {args.min_imgs} images   : {too_small} ecartes "
          f"(insuffisant pour des positifs)")
    frac = n_pl_te / max(1, n_pl)
    print(f"  Lieux               : {n_pl}  (train {n_pl_tr} / test {n_pl_te} "
          f"= {frac:.0%} reel, vise {args.test_ratio:.0%})")
    print(f"  Tampon train/test   : {args.separation:.0f} m  "
          f"(blocs de {args.test_block:.0f} m)")
    print(f"  Images/{city_id}/   : {copied} fichiers <ville>_<place>_<...>.jpg")
    print(f"  Dataframes/         : {city_id}.csv, {city_id}_train.csv, "
          f"{city_id}_test.csv")
    print("=" * 60)
    print("\n  Pour MegaLoc/GSV-Cities : pointer le dataloader sur "
          f"Dataframes/{city_id}_train.csv")
    return 0
