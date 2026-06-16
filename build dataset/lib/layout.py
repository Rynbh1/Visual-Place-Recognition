import math
import numpy as np

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
