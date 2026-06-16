import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from lib.CONSTANTS import THUMB_FIELD, GRAPH_URL, MIN_PHOTOS_PER_PLACE
from lib.layout import build_layout, equirect_to_perspective
from lib.supplement import topup_places
from lib.utils import get_token

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


def cmd_download(args):
    token = get_token()
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("method") == "pano-crop":
        rc = download_pano_crops(plan, token, args.assume_north_aligned)
        supplement = plan.get("supplement", {})
        if supplement:
            manifest_path = Path(plan["out_dir"]) / "manifest.jsonl"
            if manifest_path.exists():
                topup_places(Path(plan["out_dir"]), supplement, manifest_path)
        return rc
    if plan.get("method") == "both":
        rc = download_both(plan, token, args)
        supplement = plan.get("supplement", {})
        if supplement:
            manifest_path = Path(plan["out_dir"]) / "manifest.jsonl"
            if manifest_path.exists():
                topup_places(Path(plan["out_dir"]), supplement, manifest_path)
        return rc

    out_dir = Path(plan["out_dir"])
    resolution = plan["resolution"]
    place_size = plan.get("place_size_m", 25.0)
    layout = build_layout(plan["images"], place_size)

    manifest_path = out_dir / "manifest.jsonl"
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
        supplement = plan.get("supplement", {})
        if supplement and manifest_path.exists():
            topup_places(out_dir, supplement, manifest_path)
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

    supplement = plan.get("supplement", {})
    if supplement:
        topup_places(out_dir, supplement, manifest_path)

    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if rows:
        csv_path = out_dir / "manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wri.writeheader()
            wri.writerows(rows)
        n_places = len({r.get("place") for r in rows})
        filtered = len(todo) - ok
        print(f"\n  {ok} nouvelles images Mapillary, {filtered} ecartees "
              f"(qualite/illisibles/echecs).")
        print(f"  CSV : {csv_path}  ({len(rows)} images, {n_places} lieux)")
        print(f"  Images : {out_dir / 'places'}/place_NNNNNN/img_<angle>.jpg")
    return 0
