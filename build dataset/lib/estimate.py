import math
import json
import sys
from pathlib import Path

from lib.CONSTANTS import AVG_CROP_MB, AVG_MB, MIN_PHOTOS_PER_PLACE
from lib.geocode import geocode, bbox_area_km2
from lib.tilling import scan_coverage
from lib.layout import place_key, meters, bearing_deg, cell_center, cell_anchors
from lib.supplement import _pano_bbox_scan, _kv_bbox_scan, _build_supplement, write_coverage_map
from lib.utils import get_token, slug

def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


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


def cmd_estimate(args):
    token = get_token()
    out_dir = Path(args.outdir) / slug(args.city, args.postal)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Geocodage...")
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
        print("\n[2/3] Recherche des panoramas + planification des crops...")
        min_views = args.min_views if args.min_views is not None else args.views
        tasks, kept, dropped, hist = scan_pano_crops(
            bbox, polygon, token, args.place_size, args.views, min_views,
            args.max_images,
        )
        n = len(tasks)
        n_places = kept
        size_mb = n * AVG_CROP_MB
        est_time = n / max(1, args.workers * 0.5)
        # Supplement : cellules associees a chaque place_id (pour topup)
        place_cell_map = {}
        for t in tasks:
            pid = t["place_id"]
            cell = place_key(t["pano_lon"], t["pano_lat"], args.place_size)
            place_cell_map[pid] = cell
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
        # index Mapillary by cell pour la coverage map
        mly_by_cell = {}
        for t in tasks:
            cell = place_key(t["pano_lon"], t["pano_lat"], args.place_size)
            mly_by_cell.setdefault(cell, []).append(t)
    elif args.method == "both":
        print("\n[2/3] Panoramas (vers le centre) + perspectives par lieu...")
        tasks, n_places, n_pano, n_flat = scan_both(
            bbox, polygon, token, args.place_size, args.views, args.min_dist,
            args.max_images,
        )
        n = len(tasks)
        size_mb = n_pano * AVG_CROP_MB + n_flat * AVG_MB[args.resolution]
        est_time = n / max(1, args.workers * 0.6)
        place_cell_map = {}
        for t in tasks:
            pid = t["place_id"]
            lon = t.get("pano_lon") or t.get("lon", 0)
            lat = t.get("pano_lat") or t.get("lat", 0)
            place_cell_map[pid] = place_key(lon, lat, args.place_size)
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
        mly_by_cell = {}
        for t in tasks:
            lon = t.get("pano_lon") or t.get("lon", 0)
            lat = t.get("pano_lat") or t.get("lat", 0)
            cell = place_key(lon, lat, args.place_size)
            mly_by_cell.setdefault(cell, []).append(t)
    else:  # shared
        print("\n[2/3] Balayage de la couverture Mapillary...")
        images = scan_coverage(
            bbox, polygon, token, args.image_type, args.min_dist, args.max_images
        )
        n = len(images)
        # groupe par cellule pour la coverage map et le supplement
        mly_by_cell = {}
        for im in images:
            cell = place_key(im["lon"], im["lat"], args.place_size)
            mly_by_cell.setdefault(cell, []).append(im)
        n_places = len(mly_by_cell)
        size_mb = n * AVG_MB[args.resolution]
        est_time = n / max(1, args.workers * 0.7)
        # place_cell_map pour le supplement : place_NNNNNN -> cellule
        place_cell_map = {}
        for pidx, cell in enumerate(sorted(mly_by_cell)):
            place_cell_map[f"place_{pidx:06d}"] = cell
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

    # -----------------------------------------------------------------
    # [3/3] Scan Panoramax + KartaView -> supplement + coverage map
    # -----------------------------------------------------------------
    print("\n[3/3] Scan Panoramax + KartaView (complement de couverture)...")
    pano_by_cell = _pano_bbox_scan(bbox, polygon, args.place_size)
    kv_by_cell   = _kv_bbox_scan(bbox, polygon, args.place_size)

    supplement = _build_supplement(pano_by_cell, kv_by_cell, place_cell_map)
    plan["supplement"] = supplement

    map_path = out_dir / "coverage_map.html"
    write_coverage_map(bbox, args.place_size, mly_by_cell, pano_by_cell,
                       kv_by_cell, map_path)

    # compter les cellules sans aucune couverture dans la bbox
    all_covered = set(mly_by_cell) | set(pano_by_cell) | set(kv_by_cell)
    needs_topup = sum(
        1 for pid, cell in place_cell_map.items()
        if len(mly_by_cell.get(cell, [])) < MIN_PHOTOS_PER_PLACE
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
    print(f"  Panoramax               : {sum(len(v) for v in pano_by_cell.values())} images "
          f"({len(pano_by_cell)} cellules)")
    print(f"  KartaView               : {sum(len(v) for v in kv_by_cell.values())} images "
          f"({len(kv_by_cell)} cellules)")
    print(f"  Lieux a completer       : {needs_topup} < {MIN_PHOTOS_PER_PLACE} photos "
          f"(seront completes par Panoramax/KartaView au download)")
    print(f"  Plan ecrit              : {plan_path}")
    print(f"  Carte couverture        : {map_path}")
    print("=" * 60)
    print("\n  Pour lancer le telechargement :")
    print(f"    python {Path(sys.argv[0]).name} download --plan {plan_path}")
    return 0
