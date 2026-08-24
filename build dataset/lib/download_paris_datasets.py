import sys
from pathlib import Path
from urllib.parse import quote
import requests

MS_BASE = "https://megascenes.s3.us-west-2.amazonaws.com"
MS_CATEGORIES_URL = f"{MS_BASE}/metadata/categories.json"
MS_PARQUET_URL = f"{MS_BASE}/metadata/images_index.parquet"

def download_file(url, dest_path, desc=""):
    dest_path = Path(dest_path)
    print(f"Downloading {desc or url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        try:
            from tqdm import tqdm
            progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
        except ImportError:
            progress = None
            
        with open(temp_path, "wb") as f:
            for data in r.iter_content(block_size):
                f.write(data)
                if progress:
                    progress.update(len(data))
        if progress:
            progress.close()
            
        temp_path.rename(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"Error downloading {desc or url}: {e}", file=sys.stderr)
        raise e

def _ms_scene_path(scene_id):
    # Scene ID is padded to 6 digits, split into two 3-digit subfolders
    sid_str = f"{int(scene_id):06d}"
    return f"{sid_str[:3]}/{sid_str[3:]}"

def _is_paris_cat(cat, keywords):
    cat_lower = str(cat).lower()
    return any(kw in cat_lower for kw in keywords)

def _ms_download_one(key, out_dir):
    out_dir = Path(out_dir)
    dest = out_dir / key
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    try:
        r = requests.get(f"{MS_BASE}/images/{quote(key, safe='/')}", timeout=120)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest.with_suffix(dest.suffix + ".tmp")
        temp_path.write_bytes(r.content)
        temp_path.rename(dest)
        return 1
    except requests.RequestException:
        return 0
