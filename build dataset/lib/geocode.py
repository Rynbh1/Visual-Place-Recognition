import math
import requests
from lib.CONSTANTS import NOMINATIM_URL

# ---------------------------------------------------------------------------
# 1. Geocodage de la zone (Nominatim / OpenStreetMap)
# ---------------------------------------------------------------------------
def geocode(city: str, postal: str | None):
    """Renvoie (bbox, polygon) ; bbox = (west, south, east, north) en degres.

    polygon = liste d'anneaux [[ (lon,lat), ... ]] si disponible (sinon None),
    pour filtrer precisement les images a l'interieur de la limite administrative.
    """
    headers = {"User-Agent": "vpr-dataset-builder/1.0 (contact: local)"}
    data = []
    if postal:
        params = {
            "city": city,
            "postalcode": postal,
            "format": "jsonv2",
            "limit": 1,
            "polygon_geojson": 1,
            "addressdetails": 0,
        }
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception:
            data = []

    query = f"{city} {postal}" if postal else city
    if not data:
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "polygon_geojson": 1,
            "addressdetails": 0,
        }
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
