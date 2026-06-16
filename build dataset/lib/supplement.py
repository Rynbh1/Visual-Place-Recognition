import math
import re
import json
import time
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from lib.CONSTANTS import (
    PANORAMAX_SEARCH,
    KARTAVIEW_PHOTOS,
    MIN_PHOTOS_PER_PLACE,
    _COVERAGE_MAP_TMPL,
)
from lib.layout import place_key
from lib.tilling import point_in_polygon

def _pano_bbox_scan(bbox, polygon, place_size):
    """Scanne Panoramax IGN sur la bbox ; renvoie dict {cell -> [candidats]}."""
    w, s, e, n = bbox
    session = requests.Session()
    by_cell = {}
    url = PANORAMAX_SEARCH
    params = {"bbox": f"{w},{s},{e},{n}", "limit": 500}
    fetched = 0
    while url:
        try:
            r = session.get(url, params=params, timeout=30)
            params = {}
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        for feat in data.get("features", []):
            coords = (feat.get("geometry") or {}).get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            if polygon and not point_in_polygon(lon, lat, polygon):
                continue
            assets = feat.get("assets", {})
            img_url = ""
            for k in ("hd", "sd"):
                v = assets.get(k) or {}
                if v.get("href"):
                    img_url = v["href"]
                    break
            if not img_url:
                img_url = next(
                    (lk.get("href", "") for lk in feat.get("links", [])
                     if lk.get("rel") == "enclosure"), ""
                )
            if not img_url:
                continue
            cell = place_key(lon, lat, place_size)
            by_cell.setdefault(cell, []).append({
                "id": f"pano_{feat.get('id', '')}",
                "url": img_url, "lon": lon, "lat": lat, "source": "panoramax",
            })
            fetched += 1
        url = next(
            (lk["href"] for lk in data.get("links", []) if lk.get("rel") == "next"),
            None,
        )
    print(f"  Panoramax : {fetched} images ({len(by_cell)} cellules)")
    return by_cell


def _kv_bbox_scan(bbox, polygon, place_size):
    """Scanne KartaView sur la bbox ; renvoie dict {cell -> [candidats]}."""
    w, s, e, n = bbox
    session = requests.Session()
    by_cell = {}
    page = 1
    fetched = 0
    while True:
        try:
            r = session.get(
                KARTAVIEW_PHOTOS,
                params={"bbBottomLeft": f"{s},{w}", "bbTopRight": f"{n},{e}",
                        "page": page, "itemsPerPage": 500},
                timeout=30,
            )
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        items = (data.get("result") or {}).get("data") or []
        if not items:
            break
        for item in items:
            try:
                lat = float(item["lat"])
                lon = float(item["lng"])
            except (KeyError, ValueError, TypeError):
                continue
            if polygon and not point_in_polygon(lon, lat, polygon):
                continue
            url = (item.get("filePath") or item.get("largeThumbnailPath") or "").strip()
            if not url:
                continue
            if url.startswith("/"):
                url = "https://kartaview.org" + url
            cell = place_key(lon, lat, place_size)
            by_cell.setdefault(cell, []).append({
                "id": f"kv_{item.get('id', '')}",
                "url": url, "lon": lon, "lat": lat, "source": "kartaview",
            })
            fetched += 1
        tot = (data.get("result") or {}).get("totalFilteredItems")
        if isinstance(tot, list):
            tot = tot[0] if tot else 0
        tot = int(tot or 0)
        if page * 500 >= tot or not tot:
            break
        page += 1
    print(f"  KartaView : {fetched} images ({len(by_cell)} cellules)")
    return by_cell


def write_coverage_map(bbox, place_size, mly_by_cell, pano_by_cell, kv_by_cell, out_path):
    """Genere coverage_map.html (Leaflet) : grille coloree par couverture."""
    w, s, e, n = bbox
    all_cells = set(mly_by_cell) | set(pano_by_cell) | set(kv_by_cell)
    features = []
    for cell in all_cells:
        ix, iy = cell
        dlat = place_size / 111_320
        lat_c = iy * dlat
        dlon = place_size / (111_320 * math.cos(math.radians(lat_c)) + 1e-9)
        lon_c = ix * dlon
        lon0, lon1 = lon_c - dlon / 2, lon_c + dlon / 2
        lat0, lat1 = lat_c - dlat / 2, lat_c + dlat / 2
        n_mly  = len(mly_by_cell.get(cell, []))
        n_pano = len(pano_by_cell.get(cell, []))
        n_kv   = len(kv_by_cell.get(cell, []))
        total  = n_mly + n_pano + n_kv
        if n_mly >= MIN_PHOTOS_PER_PLACE:
            color = "#2ecc71"
        elif total >= MIN_PHOTOS_PER_PLACE:
            color = "#27ae60"
        elif total > 0:
            color = "#e67e22"
        else:
            color = "#e74c3c"
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon0, lat0], [lon1, lat0],
                                  [lon1, lat1], [lon0, lat1], [lon0, lat0]]]
            },
            "properties": {
                "n_mly": n_mly, "n_pano": n_pano, "n_kv": n_kv, "total": total,
                "color": color,
                "popup": (f"Mapillary:{n_mly} | Panoramax:{n_pano} | "
                          f"KartaView:{n_kv} | Total:{total}"),
            }
        })
    geojson = json.dumps(
        {"type": "FeatureCollection", "features": features}, ensure_ascii=False
    )
    lat_ctr = (s + n) / 2
    lon_ctr = (w + e) / 2
    html = (
        _COVERAGE_MAP_TMPL
        .replace("LAT_CTR", f"{lat_ctr:.5f}")
        .replace("LON_CTR", f"{lon_ctr:.5f}")
        .replace("GEOJSON", geojson)
        .replace("MIN_P", str(MIN_PHOTOS_PER_PLACE))
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"  Carte couverture -> {out_path}")


def _build_supplement(pano_by_cell, kv_by_cell, place_cell_map, max_per_place=None):
    """Construit {place_id -> [candidats]} pour le champ 'supplement' du plan."""
    limit = max_per_place or MIN_PHOTOS_PER_PLACE
    supp = {}
    for pid, cell in place_cell_map.items():
        cands = []
        for src in (pano_by_cell.get(cell, []), kv_by_cell.get(cell, [])):
            for c in src:
                if len(cands) >= limit:
                    break
                cands.append(c)
            if len(cands) >= limit:
                break
        if cands:
            supp[pid] = cands
    return supp


def _dl_supplement(session, url, dest):
    """Telecharge une image de complement ; renvoie True si OK."""
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            content = r.content
            Image.open(BytesIO(content)).convert("RGB")
            dest.write_bytes(content)
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False


def topup_places(out_dir, supplement, manifest_path, min_photos=MIN_PHOTOS_PER_PLACE):
    """Apres le DL Mapillary, complete les lieux avec < min_photos images
    depuis Panoramax / KartaView en utilisant les candidats preselectionnes."""
    if not supplement:
        return 0
    rows = [json.loads(l)
            for l in manifest_path.read_text().splitlines() if l.strip()]
    count_by = {}
    for r in rows:
        count_by[r["place"]] = count_by.get(r["place"], 0) + 1
    to_fill = {pid: cands for pid, cands in supplement.items()
               if count_by.get(pid, 0) < min_photos}
    if not to_fill:
        return 0
    print(f"\n  Complement Panoramax/KartaView : "
          f"{len(to_fill)} lieu(x) < {min_photos} photos")
    session = requests.Session()
    new_rows, total = [], 0
    for place_id, candidates in to_fill.items():
        need = min_photos - count_by.get(place_id, 0)
        place_dir = out_dir / "places" / place_id
        place_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for cand in candidates:
            if added >= need:
                break
            safe = re.sub(r"[^A-Za-z0-9]", "", str(cand["id"]))[:40]
            fname = f"supp_{cand['source']}_{safe}.jpg"
            dest = place_dir / fname
            if dest.exists() or _dl_supplement(session, cand["url"], dest):
                rel = f"places/{place_id}/{fname}"
                new_rows.append({
                    "id": cand["id"], "place": place_id, "file": rel,
                    "lon": cand.get("lon"), "lat": cand.get("lat"),
                    "source": cand["source"],
                })
                added += 1
        if added:
            print(f"    {place_id} : +{added}")
        total += added
    if new_rows:
        with manifest_path.open("a", encoding="utf-8") as mf:
            for r in new_rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
    if total:
        print(f"  Total complement : {total} images ajoutees.")
    return total
