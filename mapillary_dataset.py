#!/usr/bin/env python3
"""Pipeline de creation d'un dataset VPR a partir de l'API Mapillary.

Entree : un nom de ville (ou un arrondissement + code postal).
Sortie : un dossier d'images street-level geolocalisees + un manifeste
         (id, GPS, cap/orientation, date, type de camera, sequence...).

La pipeline se deroule en DEUX phases pour pouvoir valider avant de telecharger :

  1) estimate : geocode la zone (Nominatim/OSM), balaye la couverture Mapillary
                via les tuiles vectorielles, compte les images, estime la
                taille et la duree, puis ecrit un plan (dataset_plan.json).

  2) download : lit le plan valide, recupere les metadonnees + les images
                en parallele (avec reprise et backoff), ecrit le manifeste.

Exemples :
    export MAPILLARY_TOKEN="MLY|xxx|yyy"
    python mapillary_dataset.py estimate --city "Paris" --postal 75011
    python mapillary_dataset.py download --plan datasets/paris_75011/dataset_plan.json

Modele de donnees Mapillary :
  - Tuiles de couverture : https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}
    couche "image" -> points avec id, sequence_id, captured_at, compass_angle, is_pano.
  - Graph API (metadonnees + URL telechargeables) :
    https://graph.mapillary.com/{id}?fields=...&access_token=...
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constantes Mapillary
# ---------------------------------------------------------------------------
TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
GRAPH_URL = "https://graph.mapillary.com/{id}"
COVERAGE_ZOOM = 14  # zoom auquel la couche "image" est disponible
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Taille moyenne d'une image JPEG selon la resolution (Mo) -> pour l'estimation
AVG_MB = {"256": 0.02, "1024": 0.18, "2048": 0.55, "original": 2.5}
THUMB_FIELD = {
    "256": "thumb_256_url",
    "1024": "thumb_1024_url",
    "2048": "thumb_2048_url",
    "original": "thumb_original_url",
}


def get_token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not tok:
        print(
            "ERREUR : variable d'environnement MAPILLARY_TOKEN absente.\n"
            "  Cree un token sur https://www.mapillary.com/dashboard/developers\n"
            '  puis : export MAPILLARY_TOKEN="MLY|xxxxx|yyyyy"',
            file=sys.stderr,
        )
        sys.exit(2)
    return tok


# ---------------------------------------------------------------------------
# 1. Geocodage de la zone (Nominatim / OpenStreetMap)
# ---------------------------------------------------------------------------
def geocode(city: str, postal: str | None):
    """Renvoie (bbox, polygon) ; bbox = (west, south, east, north) en degres.

    polygon = liste d'anneaux [[ (lon,lat), ... ]] si disponible (sinon None),
    pour filtrer precisement les images a l'interieur de la limite administrative.
    """
    query = f"{city} {postal}" if postal else city
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "polygon_geojson": 1,
        "addressdetails": 0,
    }
    headers = {"User-Agent": "vpr-dataset-builder/1.0 (contact: local)"}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise SystemExit(f"Zone introuvable via Nominatim : '{query}'")
    hit = data[0]
    s, n, w, e = (float(x) for x in hit["boundingbox"])  # south,north,west,east
    bbox = (w, s, e, n)

    polygon = None
    geom = hit.get("geojson", {})
    if geom.get("type") == "Polygon":
        polygon = geom["coordinates"]
    elif geom.get("type") == "MultiPolygon":
        polygon = [ring for poly in geom["coordinates"] for ring in poly]

    print(f"  Zone : {hit.get('display_name', query)}")
    print(f"  BBox : O={w:.4f} S={s:.4f} E={e:.4f} N={n:.4f}")
    print(f"  Polygone administratif : {'oui' if polygon else 'non (bbox seule)'}")
    return bbox, polygon


def bbox_area_km2(bbox) -> float:
    w, s, e, n = bbox
    lat_m = (n - s) * 111_320
    lon_m = (e - w) * 111_320 * math.cos(math.radians((n + s) / 2))
    return abs(lat_m * lon_m) / 1e6


# ---------------------------------------------------------------------------
# 2. Balayage de la couverture via les tuiles vectorielles
# ---------------------------------------------------------------------------
def point_in_polygon(lon, lat, rings) -> bool:
    """Ray-casting : vrai si (lon,lat) est dans le polygone (premier anneau =
    exterieur, suivants = trous). Suffisant et sans dependance lourde."""

    def inside(ring):
        c = False
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if ((yi > lat) != (yj > lat)) and (
                lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi
            ):
                c = not c
            j = i
        return c

    if not inside(rings[0]):
        return False
    return not any(inside(hole) for hole in rings[1:])


def scan_coverage(bbox, polygon, token, image_type, min_dist_m, max_images):
    """Parcourt les tuiles z14 et renvoie la liste des images filtrees.

    image_type : 'pano' (panoramas equirectangulaires -> base de reference P2E),
                 'flat' (images perspectives -> requetes), ou 'all'.
    min_dist_m : sous-echantillonnage spatial (1 image par cellule de cette taille)
                 pour eviter les images quasi-identiques d'une meme sequence.
    """
    import mapbox_vector_tile
    import mercantile

    w, s, e, n = bbox
    tiles = list(mercantile.tiles(w, s, e, n, COVERAGE_ZOOM))
    session = requests.Session()

    seen_ids = set()
    grid = {}  # cellule -> image retenue (sous-echantillonnage)
    images = []

    print(f"  {len(tiles)} tuiles z{COVERAGE_ZOOM} a balayer...")
    for k, t in enumerate(tiles, 1):
        url = TILE_URL.format(z=t.z, x=t.x, y=t.y)
        try:
            resp = session.get(url, params={"access_token": token}, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code != 200 or not resp.content:
            continue

        tb = mercantile.bounds(t)  # west south east north (degres)
        try:
            decoded = mapbox_vector_tile.decode(resp.content)
        except Exception:
            continue
        layer = decoded.get("image")
        if not layer:
            continue
        extent = layer.get("extent", 4096)

        for feat in layer["features"]:
            props = feat["properties"]
            iid = props.get("id")
            if iid is None or iid in seen_ids:
                continue
            is_pano = bool(props.get("is_pano", False))
            if image_type == "pano" and not is_pano:
                continue
            if image_type == "flat" and is_pano:
                continue

            geom = feat["geometry"]
            if geom["type"] != "Point":
                continue
            px, py = geom["coordinates"]
            lon = tb.west + (px / extent) * (tb.east - tb.west)
            lat = tb.south + (py / extent) * (tb.north - tb.south)

            if not (w <= lon <= e and s <= lat <= n):
                continue
            if polygon and not point_in_polygon(lon, lat, polygon):
                continue

            # sous-echantillonnage spatial
            if min_dist_m > 0:
                dlat = min_dist_m / 111_320
                dlon = min_dist_m / (111_320 * math.cos(math.radians(lat)) + 1e-9)
                cell = (round(lon / dlon), round(lat / dlat))
                if cell in grid:
                    continue
                grid[cell] = iid

            seen_ids.add(iid)
            images.append(
                {
                    "id": str(iid),
                    "lon": round(lon, 7),
                    "lat": round(lat, 7),
                    "compass_angle": props.get("compass_angle"),
                    "is_pano": is_pano,
                    "sequence_id": props.get("sequence_id"),
                    "captured_at": props.get("captured_at"),
                }
            )
            if max_images and len(images) >= max_images:
                print(f"  Plafond --max-images={max_images} atteint.")
                return images

        if k % 25 == 0 or k == len(tiles):
            print(f"    {k}/{len(tiles)} tuiles | {len(images)} images retenues")
    return images


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def cmd_estimate(args):
    token = get_token()
    out_dir = Path(args.outdir) / slug(args.city, args.postal)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Geocodage...")
    bbox, polygon = geocode(args.city, args.postal)
    area = bbox_area_km2(bbox)
    print(f"  Surface (bbox) ~ {area:.2f} km2")

    plan = {
        "city": args.city,
        "postal": args.postal,
        "method": args.method,
        "bbox": bbox,
        "area_km2": round(area, 2),
        "image_type": args.image_type,
        "resolution": args.resolution,
        "min_dist_m": args.min_dist,
        "place_size_m": args.place_size,
        "workers": args.workers,
        "out_dir": str(out_dir),
    }

    if args.method == "pano-crop":
        print("\n[2/2] Recherche des panoramas + planification des crops...")
        min_views = args.min_views if args.min_views is not None else args.views
        tasks, kept, dropped, hist = scan_pano_crops(
            bbox, polygon, token, args.place_size, args.views, min_views,
            args.max_images,
        )
        n = len(tasks)
        n_places = kept
        size_mb = n * AVG_CROP_MB
        est_time = n / max(1, args.workers * 0.5)
        plan.update(
            {
                "fov": args.fov,
                "crop_w": args.crop_w,
                "crop_h": args.crop_h,
                "views": args.views,
                "min_views": min_views,
                "count": n,
                "n_places": kept,
                "dropped_places": dropped,
                "est_size_mb": round(size_mb, 1),
                "images": tasks,
            }
        )
        print(f"\n  Carres avec >=1 panorama : {kept + dropped}")
        print(
            "    repartition panos/carre : "
            + "  ".join(f"{b if b < 5 else '5+'}:{hist.get(b, 0)}" for b in range(1, 6))
        )
        print(
            f"    -> {kept} carres gardes (>= {min_views} panos), "
            f"{dropped} abandonnes"
        )
    elif args.method == "both":
        print("\n[2/2] Panoramas (vers le centre) + perspectives par lieu...")
        tasks, n_places, n_pano, n_flat = scan_both(
            bbox, polygon, token, args.place_size, args.views, args.min_dist,
            args.max_images,
        )
        n = len(tasks)
        size_mb = n_pano * AVG_CROP_MB + n_flat * AVG_MB[args.resolution]
        est_time = n / max(1, args.workers * 0.6)
        plan.update(
            {
                "fov": args.fov,
                "crop_w": args.crop_w,
                "crop_h": args.crop_h,
                "views": args.views,
                "place_size_m": args.place_size,
                "count": n,
                "count_pano": n_pano,
                "count_flat": n_flat,
                "n_places": n_places,
                "est_size_mb": round(size_mb, 1),
                "images": tasks,
            }
        )
        print(f"\n  {n_pano} crops panoramiques + {n_flat} perspectives "
              f"sur {n_places} lieux")
    else:
        print("\n[2/2] Balayage de la couverture Mapillary...")
        images = scan_coverage(
            bbox, polygon, token, args.image_type, args.min_dist, args.max_images
        )
        n = len(images)
        n_places = len(
            {place_key(im["lon"], im["lat"], args.place_size) for im in images}
        )
        size_mb = n * AVG_MB[args.resolution]
        est_time = n / max(1, args.workers * 0.7)
        plan.update(
            {
                "count": n,
                "count_pano": sum(1 for im in images if im["is_pano"]),
                "count_flat": sum(1 for im in images if not im["is_pano"]),
                "n_places": n_places,
                "est_size_mb": round(size_mb, 1),
                "images": images,
            }
        )

    plan_path = out_dir / "dataset_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    title = f"{args.city}{' ' + args.postal if args.postal else ''}"
    print("\n" + "=" * 60)
    print(f"  ESTIMATION  ({title}) -- methode : {args.method}")
    print("=" * 60)
    if args.method == "pano-crop":
        print(f"  Lieux retenus           : {n_places}")
        print(f"  Crops a produire        : {n}  (<= {args.views}/lieu, vers le centre)")
        print(f"  Vues / lieu (moyenne)   : {n / max(1, n_places):.1f}")
        print(
            f"  FoV / taille crop       : {args.fov} deg / {args.crop_w}x{args.crop_h}"
        )
    elif args.method == "both":
        print(f"  Lieux                   : {n_places}")
        print(f"  Images a produire       : {n}  ({plan['count_pano']} pano + "
              f"{plan['count_flat']} perspectives)")
        print(f"  Images / lieu (moyenne) : {n / max(1, n_places):.1f}")
    else:
        print(f"  Images retenues         : {n}  ({n_places} lieux)")
    print(f"  Resolution source       : {args.resolution} px")
    print(f"  Taille estimee          : {size_mb / 1024:.2f} Go ({size_mb:.0f} Mo)")
    print(f"  Duree estimee (x{args.workers})  : {human_time(est_time)}")
    print(f"  Plan ecrit              : {plan_path}")
    print("=" * 60)
    print("\n  Pour lancer le telechargement :")
    print(f"    python {Path(sys.argv[0]).name} download --plan {plan_path}")
    return 0


# ---------------------------------------------------------------------------
# Organisation en sous-dossiers de lieux + nommage par angle
# ---------------------------------------------------------------------------
def place_key(lon, lat, size_m):
    """Cellule spatiale (ix, iy) a laquelle appartient un point GPS."""
    dlat = size_m / 111_320
    dlon = size_m / (111_320 * math.cos(math.radians(lat)) + 1e-9)
    return (round(lon / dlon), round(lat / dlat))


def build_layout(images, size_m):
    """id -> (place_id, filename).

    Regroupe les images par cellule spatiale (= un lieu = un "quadruplet"),
    un sous-dossier place_NNNNNN par lieu, et nomme chaque image par son angle
    de prise de vue : img_{angle}.jpg (suffixe _2, _3... si meme angle repete).
    """
    cells = {}
    for im in images:
        cells.setdefault(place_key(im["lon"], im["lat"], size_m), []).append(im)
    layout = {}
    for pidx, key in enumerate(sorted(cells)):
        place_id = f"place_{pidx:06d}"
        used = set()
        for im in sorted(cells[key], key=lambda x: str(x["id"])):
            a = im.get("compass_angle")
            base = f"img_{int(round(a)) % 360:03d}" if a is not None else "img_na"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            layout[str(im["id"])] = (place_id, f"{name}.jpg")
    return layout


# ---------------------------------------------------------------------------
# Mode "pano-crop" : 4 vues panoramiques recadrees vers le centre du carre
# ---------------------------------------------------------------------------
AVG_CROP_MB = 0.20  # taille moyenne d'un crop perspective JPEG (~1024x768)


def meters(lon1, lat1, lon2, lat2):
    dlat = (lat2 - lat1) * 111_320
    dlon = (lon2 - lon1) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def bearing_deg(lon1, lat1, lon2, lat2):
    """Cap (0=Nord, sens horaire) du point 1 vers le point 2 (approx. planaire)."""
    dlon = (lon2 - lon1) * 111_320 * math.cos(math.radians((lat1 + lat2) / 2))
    dlat = (lat2 - lat1) * 111_320
    return math.degrees(math.atan2(dlon, dlat)) % 360


def equirect_to_perspective(
    equ, rel_yaw_deg, fov_deg=90.0, out_w=1024, out_h=768, pitch_deg=0.0
):
    """Recadre une image equirectangulaire en une vue perspective.

    rel_yaw_deg : angle horizontal a viser, RELATIF au centre du panorama.
      Convention Mapillary : le centre de l'image equirectangulaire pointe vers
      compass_angle. Donc pour viser un cap absolu B :
          rel_yaw_deg = B - compass_angle.
    """
    H, W = equ.shape[:2]
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    u = np.arange(out_w) - (out_w - 1) / 2.0
    v = np.arange(out_h) - (out_h - 1) / 2.0
    uu, vv = np.meshgrid(u, v)
    x, y, z = uu, vv, np.full_like(uu, f)

    if pitch_deg:
        p = math.radians(pitch_deg)
        y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    a = math.radians(rel_yaw_deg)
    x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)

    lon = np.arctan2(x, z)  # 0 = colonne centrale
    lat = np.arctan2(-y, np.sqrt(x * x + z * z))  # >0 = vers le haut
    ex = (lon / (2 * math.pi) + 0.5) * W
    ey = (0.5 - lat / math.pi) * H

    x0 = np.floor(ex).astype(int)
    y0 = np.floor(ey).astype(int)
    wx = (ex - x0)[..., None]
    wy = (ey - y0)[..., None]
    x0w, x1w = x0 % W, (x0 + 1) % W  # enroulement horizontal
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y0 + 1, 0, H - 1)
    Ia, Ib = equ[y0c, x0w], equ[y0c, x1w]
    Ic, Id = equ[y1c, x0w], equ[y1c, x1w]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


def cell_center(ix, iy, size_m):
    dlat = size_m / 111_320
    lat_c = iy * dlat
    dlon = size_m / (111_320 * math.cos(math.radians(lat_c)) + 1e-9)
    return ix * dlon, lat_c, dlon, dlat


def cell_anchors(lon_c, lat_c, dlon, dlat, n):
    """n points de vise repartis sur le pourtour du carre (n=4 -> les 4 coins)."""
    rlon, rlat = dlon / 2, dlat / 2
    pts = []
    for k in range(n):
        ang = math.radians(45 + 360 * k / n)
        pts.append((lon_c + rlon * math.cos(ang), lat_c + rlat * math.sin(ang)))
    return pts


def scan_pano_crops(bbox, polygon, token, place_size, views, min_views, max_images):
    """Pour chaque carre contenant au moins `min_views` panoramas, choisit jusqu'a
    `views` panoramas repartis sur le pourtour, chacun recadre vers le CENTRE.

    Renvoie (taches, kept, dropped, hist). Un carre avec moins de `min_views`
    panoramas est ABANDONNE (pas assez de points de vue distincts)."""
    panos = scan_coverage(bbox, polygon, token, "pano", 0, 0)
    print(
        f"  {len(panos)} panoramas trouves. Regroupement en carres de "
        f"{place_size:.0f} m..."
    )

    cells = {}
    for p in panos:
        cells.setdefault(place_key(p["lon"], p["lat"], place_size), []).append(p)

    hist = {}
    for members in cells.values():
        b = min(len(members), 5)  # 5 = "5 et +"
        hist[b] = hist.get(b, 0) + 1

    tasks, kept, dropped = [], 0, 0
    for key in sorted(cells):
        members = cells[key]
        if len(members) < min_views:
            dropped += 1
            continue
        ix, iy = key
        lon_c, lat_c, dlon, dlat = cell_center(ix, iy, place_size)
        place_id = f"place_{kept:06d}"
        kept += 1
        remaining = list(members)
        used = set()
        for clon, clat in cell_anchors(lon_c, lat_c, dlon, dlat, views):
            if not remaining:
                break
            pick = min(remaining, key=lambda p: meters(p["lon"], p["lat"], clon, clat))
            remaining.remove(pick)
            B = bearing_deg(pick["lon"], pick["lat"], lon_c, lat_c)
            base = f"img_{int(round(B)) % 360:03d}"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            tasks.append(
                {
                    "id": pick["id"],
                    "pano_lon": pick["lon"],
                    "pano_lat": pick["lat"],
                    "compass_angle": pick.get("compass_angle"),
                    "center_lon": round(lon_c, 7),
                    "center_lat": round(lat_c, 7),
                    "target_bearing": round(B, 1),
                    "dist_to_center_m": round(
                        meters(pick["lon"], pick["lat"], lon_c, lat_c), 1
                    ),
                    "captured_at": pick.get("captured_at"),
                    "sequence_id": pick.get("sequence_id"),
                    "place_id": place_id,
                    "rel": f"places/{place_id}/{name}.jpg",
                }
            )
        if max_images and len(tasks) >= max_images:
            break
    return tasks, kept, dropped, hist


def scan_both(bbox, polygon, token, place_size, views, min_dist, max_images):
    """Mode hybride : pour chaque carre, jusqu'a `views` panoramas recadres vers
    le centre (fichiers pano_<cap>.jpg) ET les images perspectives du carre
    (fichiers flat_<cap>.jpg), regroupes dans le MEME sous-dossier de lieu.

    Renvoie (taches, n_lieux, n_pano, n_flat). Aucun carre n'est abandonne :
    le filtre du nombre min d'images se fait a l'export (--min-imgs)."""
    panos = scan_coverage(bbox, polygon, token, "pano", 0, 0)
    flats = scan_coverage(bbox, polygon, token, "flat", min_dist, 0)
    print(f"  {len(panos)} panoramas + {len(flats)} perspectives. "
          f"Regroupement en carres de {place_size:.0f} m...")

    pano_cells, flat_cells = {}, {}
    for p in panos:
        pano_cells.setdefault(place_key(p["lon"], p["lat"], place_size), []).append(p)
    for f in flats:
        flat_cells.setdefault(place_key(f["lon"], f["lat"], place_size), []).append(f)

    keys = sorted(set(pano_cells) | set(flat_cells))
    tasks, n_pano, n_flat = [], 0, 0
    for pidx, key in enumerate(keys):
        place_id = f"place_{pidx:06d}"
        ix, iy = key
        lon_c, lat_c, dlon, dlat = cell_center(ix, iy, place_size)

        # 1) panoramas recadres vers le centre (jusqu'a `views`, repartis)
        remaining = list(pano_cells.get(key, []))
        used = set()
        for clon, clat in cell_anchors(lon_c, lat_c, dlon, dlat, views):
            if not remaining:
                break
            pick = min(remaining, key=lambda p: meters(p["lon"], p["lat"], clon, clat))
            remaining.remove(pick)
            B = bearing_deg(pick["lon"], pick["lat"], lon_c, lat_c)
            base = f"pano_{int(round(B)) % 360:03d}"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            tasks.append({
                "kind": "pano",
                "id": pick["id"],
                "pano_lon": pick["lon"], "pano_lat": pick["lat"],
                "compass_angle": pick.get("compass_angle"),
                "center_lon": round(lon_c, 7), "center_lat": round(lat_c, 7),
                "target_bearing": round(B, 1),
                "dist_to_center_m": round(
                    meters(pick["lon"], pick["lat"], lon_c, lat_c), 1),
                "captured_at": pick.get("captured_at"),
                "sequence_id": pick.get("sequence_id"),
                "place_id": place_id, "rel": f"places/{place_id}/{name}.jpg",
            })
            n_pano += 1

        # 2) images perspectives du carre (telles quelles)
        usedf = set()
        for f in sorted(flat_cells.get(key, []), key=lambda x: str(x["id"])):
            a = f.get("compass_angle")
            base = f"flat_{int(round(a)) % 360:03d}" if a is not None else "flat_na"
            name, k = base, 2
            while name in usedf:
                name, k = f"{base}_{k}", k + 1
            usedf.add(name)
            tasks.append({
                "kind": "flat",
                "id": f["id"], "lon": f["lon"], "lat": f["lat"],
                "compass_angle": a,
                "captured_at": f.get("captured_at"),
                "sequence_id": f.get("sequence_id"),
                "place_id": place_id, "rel": f"places/{place_id}/{name}.jpg",
            })
            n_flat += 1

        if max_images and len(tasks) >= max_images:
            break
    return tasks, len(keys), n_pano, n_flat


def fetch_crop(session, token, task, out_dir, resolution, fov, ow, oh, north_aligned):
    """Telecharge le panorama, le recadre vers le centre du carre, l'enregistre."""
    field = THUMB_FIELD[resolution]
    fields = (
        f"id,{field},computed_compass_angle,compass_angle,camera_type,"
        "is_pano,width,height,computed_geometry"
    )
    url = GRAPH_URL.format(id=task["id"])
    dest = out_dir / task["rel"]
    for attempt in range(4):
        try:
            r = session.get(
                url, params={"fields": fields, "access_token": token}, timeout=30
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            meta = r.json()
            thumb = meta.get(field)
            if not thumb or not meta.get("is_pano"):
                return None  # pas un vrai panorama -> on saute
            if not dest.exists():
                ir = session.get(thumb, timeout=120)
                ir.raise_for_status()
                equ = np.asarray(Image.open(BytesIO(ir.content)).convert("RGB"))
                comp = (
                    meta.get("computed_compass_angle")
                    or meta.get("compass_angle")
                    or task.get("compass_angle")
                    or 0
                )
                rel_yaw = task["target_bearing"] - (0 if north_aligned else comp)
                crop = equirect_to_perspective(equ, rel_yaw, fov, ow, oh)
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(dest, "JPEG", quality=90)
            else:
                comp = (
                    meta.get("computed_compass_angle") or meta.get("compass_angle") or 0
                )
            return {
                "id": task["id"],
                "place": task["place_id"],
                "file": task["rel"],
                "lon": task["center_lon"],
                "lat": task["center_lat"],
                "target_bearing": task["target_bearing"],
                "pano_compass": round(float(comp), 2),
                "pano_lon": task["pano_lon"],
                "pano_lat": task["pano_lat"],
                "dist_to_center_m": task["dist_to_center_m"],
                "captured_at": task.get("captured_at"),
                "camera_type": meta.get("camera_type"),
                "is_pano": True,
            }
        except (requests.RequestException, OSError):
            time.sleep(2**attempt)
    return None


def download_pano_crops(plan, token, north_aligned):
    out_dir = Path(plan["out_dir"])
    resolution = plan["resolution"]
    fov, ow, oh = plan.get("fov", 90), plan.get("crop_w", 1024), plan.get("crop_h", 768)
    tasks = plan["images"]

    manifest_path = out_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    todo = [t for t in tasks if t["rel"] not in done]
    print(
        f"  {len(tasks)} crops au plan | {len(done)} deja faits | "
        f"{len(todo)} a produire"
    )
    if not todo:
        print("  Rien a faire.")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **k):
            return x

    session = requests.Session()
    ok = 0
    with (
        manifest_path.open("a", encoding="utf-8") as mf,
        ThreadPoolExecutor(max_workers=plan.get("workers", 12)) as pool,
    ):
        futures = [
            pool.submit(
                fetch_crop,
                session,
                token,
                t,
                out_dir,
                resolution,
                fov,
                ow,
                oh,
                north_aligned,
            )
            for t in todo
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  CROP"):
            row = fut.result()
            if row:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if rows:
        csv_path = out_dir / "manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()), extrasaction="ignore"
            )
            wri.writeheader()
            wri.writerows(rows)
        n_places = len({r.get("place") for r in rows})
        print(f"\n  {ok} nouveaux crops. {len(rows)} images, {n_places} lieux.")
        print(f"  Structure : {out_dir / 'places'}/place_NNNNNN/img_<cap>.jpg")
    return 0


def write_manifest_csv(out_dir, manifest_path):
    """Reecrit manifest.csv depuis le jsonl en gerant des lignes heterogenes
    (colonnes = union des cles ; valeurs manquantes laissees vides)."""
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if not rows:
        return rows
    cols = []
    for r in rows:
        for kk in r:
            if kk not in cols:
                cols.append(kk)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        wri.writeheader()
        wri.writerows(rows)
    return rows


def download_both(plan, token, args):
    """Mode hybride : telecharge les crops panoramiques (fetch_crop) ET les
    perspectives (fetch_one) dans les memes sous-dossiers de lieu."""
    out_dir = Path(plan["out_dir"])
    resolution = plan["resolution"]
    fov, ow, oh = plan.get("fov", 90), plan.get("crop_w", 1024), plan.get("crop_h", 768)
    tasks = plan["images"]

    manifest_path = out_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass
    todo = [t for t in tasks if t["rel"] not in done]
    print(f"  {len(tasks)} images au plan | {len(done)} deja faites | "
          f"{len(todo)} a produire")
    if not todo:
        print("  Rien a faire.")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    session = requests.Session()

    def run(task):
        if task["kind"] == "pano":
            row = fetch_crop(session, token, task, out_dir, resolution, fov, ow, oh,
                             args.assume_north_aligned)
        else:
            dest = out_dir / task["rel"]
            row = fetch_one(session, token, task, dest, task["rel"], task["place_id"],
                            resolution, args.min_quality, args.min_sharpness)
        if row:
            row["kind"] = task["kind"]
        return row

    ok = 0
    with (
        manifest_path.open("a", encoding="utf-8") as mf,
        ThreadPoolExecutor(max_workers=plan.get("workers", 12)) as pool,
    ):
        futures = [pool.submit(run, t) for t in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  DL"):
            row = fut.result()
            if row:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1

    rows = write_manifest_csv(out_dir, manifest_path)
    n_places = len({r.get("place") for r in rows})
    n_pano = sum(1 for r in rows if r.get("kind") == "pano")
    print(f"\n  {ok} nouvelles images gardees. {len(rows)} au total "
          f"({n_pano} pano + {len(rows) - n_pano} perspectives), {n_places} lieux.")
    print(f"  Structure : {out_dir / 'places'}/place_NNNNNN/(pano|flat)_<cap>.jpg")
    return 0


# ---------------------------------------------------------------------------
# 3. Telechargement (metadonnees + images) -- mode "shared" (photos partagees)
# ---------------------------------------------------------------------------
def looks_bad(arr, min_sharpness):
    """Heuristique pas chere : rejette les images quasi noires/blanches (capteur
    obstrue, sur/sous-exposition) ou floues/uniformes (variance de Laplacien)."""
    g = arr.mean(axis=2)
    m = float(g.mean())
    if m < 12 or m > 244:
        return "luminosite"
    lap = g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1]
    if float(lap.var()) < min_sharpness:
        return "flou/uniforme"
    return None


def fetch_one(session, token, img, dest, rel, place_id, resolution,
              min_quality, min_sharpness):
    """Recupere metadonnees + image pour un id. Renvoie la ligne de manifeste,
    ou None si l'image est filtree (faible qualite, hors-sujet, illisible).

    dest = chemin complet du fichier ; rel = chemin relatif (stocke au manifeste).
    """
    iid = img["id"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    field = THUMB_FIELD[resolution]
    fields = (
        f"id,{field},computed_geometry,geometry,compass_angle,"
        "computed_compass_angle,captured_at,camera_type,is_pano,"
        "quality_score,height,width,sequence"
    )
    url = GRAPH_URL.format(id=iid)

    for attempt in range(4):
        try:
            r = session.get(
                url, params={"fields": fields, "access_token": token}, timeout=30
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            meta = r.json()
            thumb = meta.get(field)
            if not thumb:
                return None  # resolution non disponible pour cette image
            # filtres metadonnees (avant de telecharger les pixels)
            if meta.get("camera_type") in ("spherical", "equirectangular"):
                return None
            q = meta.get("quality_score")
            if min_quality is not None and q is not None and q < min_quality:
                return None
            if not dest.exists():
                ir = session.get(thumb, timeout=60)
                ir.raise_for_status()
                content = ir.content
                try:
                    arr = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
                except OSError:
                    return None
                if min_sharpness and looks_bad(arr, min_sharpness):
                    return None
                dest.write_bytes(content)
            # GPS : on privilegie computed_geometry (raffine par SfM)
            coords = (meta.get("computed_geometry") or meta.get("geometry") or {}).get(
                "coordinates", [img["lon"], img["lat"]]
            )
            return {
                "id": iid,
                "place": place_id,
                "file": rel,
                "lon": coords[0],
                "lat": coords[1],
                "compass_angle": meta.get("computed_compass_angle")
                or meta.get("compass_angle"),
                "captured_at": meta.get("captured_at"),
                "camera_type": meta.get("camera_type"),
                "quality_score": q,
                "is_pano": meta.get("is_pano"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "sequence": meta.get("sequence"),
            }
        except requests.RequestException:
            time.sleep(2**attempt)
    return None


def cmd_download(args):
    token = get_token()
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("method") == "pano-crop":
        return download_pano_crops(plan, token, args.assume_north_aligned)
    if plan.get("method") == "both":
        return download_both(plan, token, args)

    out_dir = Path(plan["out_dir"])
    resolution = plan["resolution"]
    place_size = plan.get("place_size_m", 25.0)
    layout = build_layout(plan["images"], place_size)

    manifest_path = out_dir / "manifest.jsonl"
    # reprise : on saute les ids deja presents dans le manifeste
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [im for im in plan["images"] if im["id"] not in done]
    print(
        f"  {len(plan['images'])} images au plan | {len(done)} deja faites | "
        f"{len(todo)} a telecharger"
    )
    if not todo:
        print("  Rien a faire.")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **k):
            return x

    session = requests.Session()
    ok = 0
    with (
        manifest_path.open("a", encoding="utf-8") as mf,
        ThreadPoolExecutor(max_workers=plan.get("workers", 12)) as pool,
    ):
        futures = {}
        for im in todo:
            place_id, fname = layout[str(im["id"])]
            rel = f"places/{place_id}/{fname}"
            dest = out_dir / rel
            futures[
                pool.submit(
                    fetch_one, session, token, im, dest, rel, place_id, resolution,
                    args.min_quality, args.min_sharpness,
                )
            ] = im
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  DL"):
            row = fut.result()
            if row:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1

    # export CSV pratique en plus du jsonl
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if rows:
        csv_path = out_dir / "manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)
        n_places = len({r.get("place") for r in rows})
        filtered = len(todo) - ok
        print(f"\n  {ok} nouvelles images gardees, {filtered} ecartees "
              f"(qualite/illisibles/echecs).")
        print(f"  CSV : {csv_path}  ({len(rows)} images, {n_places} lieux)")
        print(f"  Images : {out_dir / 'places'}/place_NNNNNN/img_<angle>.jpg")
    return 0


# ---------------------------------------------------------------------------
# Reorganisation d'un dataset deja telecharge (sans re-telechargement)
# ---------------------------------------------------------------------------
def cmd_reorganize(args):
    out_dir = Path(args.dataset)
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl introuvable dans {out_dir}")

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    layout = build_layout(rows, args.place_size)

    moved, missing = 0, 0
    new_rows = []
    for row in rows:
        place_id, fname = layout[str(row["id"])]
        rel = f"places/{place_id}/{fname}"
        dest = out_dir / rel
        # ancien emplacement : valeur 'file' du manifeste, sinon images/{id}.jpg
        old = row.get("file", f"{row['id']}.jpg")
        src = out_dir / old
        if not src.exists():
            src = out_dir / "images" / Path(old).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            pass  # deja en place
        elif src.exists():
            src.rename(dest)
            moved += 1
        else:
            missing += 1
        row["place"] = place_id
        row["file"] = rel
        new_rows.append(row)

    # reecrit le manifeste (jsonl + csv) avec place + chemins relatifs
    with manifest_path.open("w", encoding="utf-8") as mf:
        for row in new_rows:
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
    if new_rows:
        ordered = [
            "id",
            "place",
            "file",
            "lon",
            "lat",
            "compass_angle",
            "captured_at",
            "camera_type",
            "is_pano",
            "width",
            "height",
            "sequence",
        ]
        keys = [k for k in ordered if k in new_rows[0]]
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wri.writeheader()
            wri.writerows(new_rows)

    # supprime l'ancien dossier images/ s'il est vide
    old_img = out_dir / "images"
    if old_img.is_dir() and not any(old_img.iterdir()):
        old_img.rmdir()

    n_places = len({r["place"] for r in new_rows})
    print(f"  {moved} fichiers deplaces, {missing} manquants.")
    print(f"  {len(new_rows)} images reparties en {n_places} lieux.")
    print(f"  Structure : {out_dir / 'places'}/place_NNNNNN/img_<angle>.jpg")
    return 0


# ---------------------------------------------------------------------------
# Export au format GSV-Cities (entrainable directement par MegaLoc)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def slug(city: str, postal: str | None) -> str:
    base = city.lower().strip().replace(" ", "_")
    return f"{base}_{postal}" if postal else base


def build_parser():
    p = argparse.ArgumentParser(description="Pipeline dataset VPR via Mapillary.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--city", required=True, help="Nom de la ville")
    common.add_argument(
        "--postal",
        default=None,
        help="Code postal (obligatoire pour un arrondissement)",
    )
    common.add_argument(
        "--image-type",
        choices=["pano", "flat", "all"],
        default="flat",
        help="pano=panoramas 360 (base reference P2E), "
        "flat=perspectives (requetes), all=tout",
    )
    common.add_argument(
        "--resolution", choices=["256", "1024", "2048", "original"], default="2048"
    )
    common.add_argument(
        "--min-dist",
        type=float,
        default=5.0,
        help="Distance min entre 2 images en m (anti-doublons)",
    )
    common.add_argument(
        "--place-size",
        type=float,
        default=25.0,
        help="Taille de cellule d'un lieu en m (1 sous-dossier/lieu)",
    )
    common.add_argument(
        "--max-images", type=int, default=0, help="Plafond d'images (0 = illimite)"
    )
    common.add_argument("--workers", type=int, default=12)
    common.add_argument("--outdir", default="datasets")
    common.add_argument(
        "--method",
        choices=["pano-crop", "shared", "both"],
        default="pano-crop",
        help="pano-crop=panoramas 360 recadres vers le centre ; "
        "shared=perspectives partagees telles quelles ; "
        "both=les deux dans le meme lieu (pano_<cap>.jpg + flat_<cap>.jpg)",
    )
    common.add_argument(
        "--views",
        type=int,
        default=4,
        help="(pano-crop) nb de vues par lieu, reparties sur le pourtour du carre",
    )
    common.add_argument(
        "--min-views",
        type=int,
        default=None,
        help="(pano-crop) nb min de panos pour garder un carre "
        "(defaut = --views ; mettre 1 pour tout garder)",
    )
    common.add_argument(
        "--fov", type=float, default=90.0,
        help="(pano-crop) champ de vision horizontal des crops en degres",
    )
    common.add_argument(
        "--crop-w", type=int, default=1024, help="(pano-crop) largeur des crops"
    )
    common.add_argument(
        "--crop-h", type=int, default=768, help="(pano-crop) hauteur des crops"
    )

    pe = sub.add_parser(
        "estimate", parents=[common], help="Geocode + balaye + estime + ecrit le plan"
    )
    pe.set_defaults(func=cmd_estimate)

    pd = sub.add_parser("download", help="Telecharge a partir d'un plan valide")
    pd.add_argument("--plan", required=True, help="Chemin du dataset_plan.json")
    pd.add_argument(
        "--assume-north-aligned",
        action="store_true",
        help="(pano-crop) supposer les panoramas alignes au Nord "
        "(centre image = Nord) au lieu de centre = compass_angle",
    )
    pd.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="(flat) seuil sur le quality_score Mapillary (ex: 0.5). "
        "Defaut: pas de filtre (le champ varie selon les images)",
    )
    pd.add_argument(
        "--min-sharpness",
        type=float,
        default=5.0,
        help="(flat) seuil de nettete local (variance de Laplacien) ; "
        "rejette images noires/blanches/floues. Mettre 0 pour desactiver",
    )
    pd.set_defaults(func=cmd_download)

    pr = sub.add_parser(
        "reorganize",
        help="Restructure un dataset deja telecharge (sans re-DL) "
        "en sous-dossiers de lieux + nommage par angle",
    )
    pr.add_argument(
        "--dataset", required=True, help="Dossier du dataset (contenant manifest.jsonl)"
    )
    pr.add_argument(
        "--place-size",
        type=float,
        default=25.0,
        help="Taille de cellule d'un lieu en m",
    )
    pr.set_defaults(func=cmd_reorganize)

    px = sub.add_parser(
        "export",
        help="Exporte un dataset telecharge au format GSV-Cities "
        "(entrainable par MegaLoc) + split train/test geographique",
    )
    px.add_argument(
        "--dataset", required=True, help="Dossier du dataset (contenant manifest.jsonl)"
    )
    px.add_argument("--out", default=None, help="Dossier de sortie (defaut: <dataset>/gsv_cities)")
    px.add_argument("--city-id", default=None,
                    help="Nom de ville (city_id GSV-Cities ; defaut: nom du dossier)")
    px.add_argument("--test-ratio", type=float, default=0.2,
                    help="Proportion de blocs en test (defaut: 0.2)")
    px.add_argument("--test-block", type=float, default=250.0,
                    help="Taille des blocs train/test en m (defaut: 250)")
    px.add_argument("--separation", type=float, default=50.0,
                    help="Zone tampon train/test en m, anti-fuite (defaut: 50)")
    px.add_argument("--min-imgs", type=int, default=4,
                    help="Nb min d'images par lieu pour le garder (defaut: 4, "
                    "= img_per_place de GSV-Cities)")
    px.set_defaults(func=cmd_export)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
