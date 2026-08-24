"""
Shared on-disk cache for per-image model outputs, reused across every VPR
inference/eval entry point: infer_megaloc.py, vpr_pipeline's MegaLocRetriever,
and evaluate_vpr.ipynb — for both MegaLoc descriptors and TextInPlace's
descriptor + OCR passes.

Encoding a full Paris-scale catalogue takes minutes to over an hour depending on
the model (MegaLoc: ~11-12 min for ~20k images; TextInPlace's OCR pass is a much
heavier per-image transformer and is far slower one image at a time). Before this
cache existed, every single script/notebook invocation re-ran the whole database
from scratch, even when nothing about it had changed since the last run five
minutes earlier. This module makes that a one-time cost per (weights, config,
image) triple, shared by every consumer that points at the same weights file.

Cache key: (weights checkpoint identity, optional extra config key, image
absolute path + mtime + size). Changing the weights file, replacing an image on
disk, or passing a different `extra_key` (e.g. input resolution or a detection
threshold that isn't captured by the weights file itself) correctly invalidates
just the affected entries instead of silently serving stale results.

Storage: one file per (weights checkpoint, extra_key), at
    <repo_root>/.cache/megaloc_descriptors/<weights-stem>-<fingerprint>[-<extra_key>].npz   (DescriptorCache)
    <repo_root>/.cache/megaloc_descriptors/<weights-stem>-<fingerprint>[-<extra_key>]-text.json   (TextCache)

Despite the directory name (kept for backward compatibility with existing
caches), this module and directory are shared by any model, not just MegaLoc.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def _weights_fingerprint(weights_path: Path) -> str:
    """Short id that changes when the weights file's content changes.

    Hashing a full multi-hundred-MB checkpoint on every run would itself be slow,
    so this only hashes the first megabyte plus the file size — enough to tell
    "this is a different checkpoint" apart without adding measurable overhead.
    """
    size = weights_path.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    with open(weights_path, "rb") as f:
        h.update(f.read(1_000_000))
    return h.hexdigest()[:10]


def _default_cache_dir() -> Path:
    # This file lives at MegaLoc/lib/descriptor_cache.py -> repo root is 2 parents up.
    return Path(__file__).resolve().parents[2] / ".cache" / "megaloc_descriptors"


def _cache_stem(weights_path: Path, extra_key: str) -> str:
    fingerprint = _weights_fingerprint(weights_path)
    stem = f"{weights_path.stem}-{fingerprint}"
    if extra_key:
        # Sanitise so arbitrary config strings (e.g. "th0.15-sz1000") stay filesystem-safe.
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in extra_key)
        stem = f"{stem}-{safe}"
    return stem


class DescriptorCache:
    """Disk-persisted cache of image descriptors for one weights checkpoint.

    Usage:
        cache = DescriptorCache(weights_path)
        descriptors = cache.get_or_encode(paths, encode_fn)          # one image at a time
        descriptors = cache.get_or_encode_batched(paths, encode_batch_fn, batch_size=32)
    """

    def __init__(self, weights_path: Path, cache_dir: Optional[Path] = None, extra_key: str = ""):
        """
        extra_key: folded into the cache filename alongside the weights fingerprint.
            Use it for any config that changes the output but isn't captured by the
            weights file itself (e.g. input resolution) — otherwise switching that
            config between runs would silently serve descriptors computed under the
            old config.
        """
        self.weights_path = Path(weights_path)
        cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{_cache_stem(self.weights_path, extra_key)}.npz"

        self._index: dict[str, int] = {}   # absolute path -> row in the lists below
        self._mtimes: list[float] = []
        self._sizes: list[int] = []
        self._descriptors: list[np.ndarray] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = np.load(self.path, allow_pickle=False)
            paths = data["paths"]
            self._mtimes = list(data["mtimes"])
            self._sizes = list(data["sizes"])
            self._descriptors = [row for row in data["descriptors"]]
            self._index = {str(p): i for i, p in enumerate(paths)}
            print(f"[descriptor_cache] Loaded {len(self._index)} cached descriptors "
                  f"from {self.path.name}")
        except Exception as exc:
            print(f"[descriptor_cache] Warning: could not read {self.path} ({exc}); starting fresh.")
            self._index, self._mtimes, self._sizes, self._descriptors = {}, [], [], []

    def _save(self) -> None:
        if not self._index:
            return
        paths_arr = np.array(list(self._index.keys()))
        mtimes_arr = np.array(self._mtimes, dtype=np.float64)
        sizes_arr = np.array(self._sizes, dtype=np.int64)
        desc_arr = np.stack(self._descriptors).astype(np.float32)

        # Write to a temp file and rename, so a crash mid-save can't corrupt the
        # existing cache (np.savez on the final path directly would truncate it).
        # np.savez silently appends ".npz" to string/Path targets that don't already
        # end with it (so a plain ".npz.tmp" path actually lands at ".npz.tmp.npz",
        # and the rename below would then fail to find it) — passing an open file
        # handle instead makes it write exactly where we tell it to.
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            np.savez(f, paths=paths_arr, mtimes=mtimes_arr, sizes=sizes_arr, descriptors=desc_arr)
        os.replace(tmp_path, self.path)

    def _is_fresh(self, abs_path: str, mtime: float, size: int) -> bool:
        idx = self._index.get(abs_path)
        if idx is None:
            return False
        return self._mtimes[idx] == mtime and self._sizes[idx] == size

    def _put(self, abs_path: str, mtime: float, size: int, descriptor: np.ndarray) -> None:
        idx = self._index.get(abs_path)
        if idx is not None:
            self._descriptors[idx] = descriptor
            self._mtimes[idx] = mtime
            self._sizes[idx] = size
        else:
            self._index[abs_path] = len(self._descriptors)
            self._descriptors.append(descriptor)
            self._mtimes.append(mtime)
            self._sizes.append(size)

    # ------------------------------------------------------------------
    def get_or_encode_batched(
        self,
        paths: list[str],
        encode_batch: Callable[[list[str]], np.ndarray],
        batch_size: int = 32,
        desc: str = "Encoding",
        save_every: int = 2000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return descriptors for `paths`, in order.

        Anything already cached (and not stale) is reused; everything else is
        encoded via `encode_batch(list_of_paths) -> array[len(paths), feat_dim]`
        in chunks of `batch_size`, then merged into the cache and periodically
        flushed to disk (every `save_every` newly-encoded images) so a crash on
        a 20k-image run doesn't lose all progress.
        """
        out: list[Optional[np.ndarray]] = [None] * len(paths)
        to_encode: list[tuple[int, str, str, float, int]] = []  # (idx, path, abs_path, mtime, size)

        for i, p in enumerate(paths):
            abs_path = str(Path(p).resolve())
            try:
                st = os.stat(abs_path)
            except OSError:
                # Let encode_batch surface the real error for a genuinely missing file.
                to_encode.append((i, p, abs_path, 0.0, 0))
                continue
            if self._is_fresh(abs_path, st.st_mtime, st.st_size):
                out[i] = self._descriptors[self._index[abs_path]]
            else:
                to_encode.append((i, p, abs_path, st.st_mtime, st.st_size))

        n_hit = len(paths) - len(to_encode)
        if n_hit:
            print(f"[descriptor_cache] {n_hit}/{len(paths)} descriptors reused from cache "
                  f"({self.path.name}); encoding {len(to_encode)} new/changed image(s).")

        if to_encode:
            chunks = [to_encode[s:s + batch_size] for s in range(0, len(to_encode), batch_size)]
            if show_progress:
                from tqdm import tqdm
                chunks = tqdm(chunks, desc=desc, ncols=80)

            since_save = 0
            for chunk in chunks:
                batch_paths = [p for _, p, _, _, _ in chunk]
                batch_desc = np.asarray(encode_batch(batch_paths), dtype=np.float32)
                for (i, _, abs_path, mtime, size), d in zip(chunk, batch_desc):
                    out[i] = d
                    if size:  # skip caching entries whose stat() failed above
                        self._put(abs_path, mtime, size, d)
                since_save += len(chunk)
                if since_save >= save_every:
                    self._save()
                    since_save = 0
            self._save()

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(f"[descriptor_cache] {len(missing)} descriptor(s) could not be produced.")

        return np.stack(out).astype(np.float32)

    def get_or_encode(
        self,
        paths: list[str],
        encode_fn: Callable[[str], np.ndarray],
        desc: str = "Encoding",
        save_every: int = 2000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Convenience wrapper for a one-image-at-a-time `encode_fn(path) -> descriptor`."""
        def encode_batch(batch_paths: list[str]) -> np.ndarray:
            return np.stack([encode_fn(p) for p in batch_paths])

        return self.get_or_encode_batched(
            paths, encode_batch, batch_size=1, desc=desc,
            save_every=save_every, show_progress=show_progress,
        )


class TextCache:
    """Disk-persisted cache of detected-text lists (e.g. TextInPlace's OCR pass).

    Same key/staleness model as DescriptorCache, but each entry is a ragged
    list[str] rather than a fixed-size float vector, so this stores JSON instead
    of a .npz — a plain array can't hold rows of different lengths.

    Usage:
        cache = TextCache(spotter_weights, extra_key=f"th{threshold}-sz{size}")
        texts = cache.get_or_encode(paths, encode_fn)   # encode_fn(path) -> list[str]
    """

    def __init__(self, weights_path: Path, cache_dir: Optional[Path] = None, extra_key: str = ""):
        self.weights_path = Path(weights_path)
        cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{_cache_stem(self.weights_path, extra_key)}-text.json"
        self._entries: dict[str, dict] = {}  # abs_path -> {"mtime":..., "size":..., "texts": [...]}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text())
            print(f"[descriptor_cache] Loaded {len(self._entries)} cached text entries "
                  f"from {self.path.name}")
        except Exception as exc:
            print(f"[descriptor_cache] Warning: could not read {self.path} ({exc}); starting fresh.")
            self._entries = {}

    def _save(self) -> None:
        if not self._entries:
            return
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._entries))
        os.replace(tmp_path, self.path)

    def _is_fresh(self, abs_path: str, mtime: float, size: int) -> bool:
        e = self._entries.get(abs_path)
        return e is not None and e["mtime"] == mtime and e["size"] == size

    def get_or_encode(
        self,
        paths: list[str],
        encode_fn: Callable[[str], list],
        desc: str = "Detecting text",
        save_every: int = 500,
        show_progress: bool = True,
    ) -> list[list[str]]:
        """Return detected-text lists for `paths`, in order, running `encode_fn` only
        on images that are missing or stale."""
        out: list[Optional[list]] = [None] * len(paths)
        to_encode: list[tuple[int, str, str, float, int]] = []

        for i, p in enumerate(paths):
            abs_path = str(Path(p).resolve())
            try:
                st = os.stat(abs_path)
            except OSError:
                to_encode.append((i, p, abs_path, 0.0, 0))
                continue
            if self._is_fresh(abs_path, st.st_mtime, st.st_size):
                out[i] = self._entries[abs_path]["texts"]
            else:
                to_encode.append((i, p, abs_path, st.st_mtime, st.st_size))

        n_hit = len(paths) - len(to_encode)
        if n_hit:
            print(f"[descriptor_cache] {n_hit}/{len(paths)} text results reused from cache "
                  f"({self.path.name}); running on {len(to_encode)} new/changed image(s).")

        if to_encode:
            iterator = to_encode
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(to_encode, desc=desc, ncols=80)

            since_save = 0
            for i, p, abs_path, mtime, size in iterator:
                texts = list(encode_fn(p))
                out[i] = texts
                if size:
                    self._entries[abs_path] = {"mtime": mtime, "size": size, "texts": texts}
                since_save += 1
                if since_save >= save_every:
                    self._save()
                    since_save = 0
            self._save()

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(f"[descriptor_cache] {len(missing)} text result(s) could not be produced.")
        return out
