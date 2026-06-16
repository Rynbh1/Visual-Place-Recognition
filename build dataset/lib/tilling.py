import math
import requests
from lib.CONSTANTS import COVERAGE_ZOOM, TILE_URL

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
