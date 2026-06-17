"""
MegaLoc global retrieval branch with FAISS nearest-neighbour index.

VRAM budget (RTX 4060, 8 GB):
  - MegaLoc fp32 : ~2.8 GB  |  fp16 : ~1.4 GB  ← we use fp16 by default
  - The model is lazy-loaded on the first encode() call so that other
    pipeline stages can initialise first without touching VRAM.
  - Call free_vram() to explicitly release the GPU allocation between
    pipeline stages when VRAM pressure is high.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is required: pip install faiss-cpu  (or faiss-gpu for GPU indexing)"
    ) from exc

# Resolve the MegaLoc library path relative to this file
_MEGALOC_ROOT = Path(__file__).resolve().parents[1] / "MegaLoc"
if str(_MEGALOC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGALOC_ROOT))

from lib.megaloc_model import MegaLoc  # noqa: E402  (import after sys.path patch)

# Standard ImageNet normalisation used during MegaLoc training
_TRANSFORM = T.Compose([
    T.Resize((322, 322)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MegaLocRetriever:
    """Visual retrieval branch: MegaLoc descriptor extraction + FAISS flat index.

    The model is lazy-loaded on the first encode call. This keeps startup time
    and initial VRAM consumption at zero until the first image is processed.

    Args:
        weights_path: Path to the fine-tuned .pth checkpoint.
        device: Torch device for inference (cuda or cpu).
        use_fp16: Cast model to half precision to halve VRAM (cuda only).
                  MegaLoc's L2-normalised output stays numerically stable in fp16.
    """

    def __init__(
        self,
        weights_path: Path,
        device: torch.device,
        use_fp16: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        # fp16 only makes sense on CUDA; silently fall back to fp32 on CPU
        self.use_fp16 = use_fp16 and device.type == "cuda"

        self._model: Optional[MegaLoc] = None  # populated on first call
        self._index: Optional[faiss.Index] = None
        self._db_meta: list[dict] = []  # parallel to FAISS index rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate MegaLoc and transfer weights to device (idempotent)."""
        if self._model is not None:
            return

        model = MegaLoc()
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(self.device)

        if self.use_fp16:
            # fp16 halves the VRAM footprint from ~2.8 GB to ~1.4 GB.
            # Safe here because MegaLoc's forward pass ends with an L2 normalisation
            # which re-scales any fp16 precision loss before the descriptor is used.
            model = model.half()

        model.eval()
        self._model = model

    @torch.no_grad()
    def _encode(self, image_path: Path) -> np.ndarray:
        """Extract a single L2-normalised descriptor (float32, shape [feat_dim])."""
        self._load_model()

        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)
        if self.use_fp16:
            tensor = tensor.half()
        tensor = tensor.to(self.device)

        # Cast to fp32 before returning to CPU so FAISS always receives fp32
        descriptor = self._model(tensor).float().cpu().numpy().squeeze(0)
        return descriptor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        db_entries: list[dict],
        index_cache: Optional[Path] = None,
    ) -> None:
        """Build a FAISS inner-product index over the database images.

        Args:
            db_entries: List of dicts, each with keys:
                        {path, place_id, lat, lon}
            index_cache: Optional .faiss file. If it exists the index is loaded
                         from disk instead of re-encoding (speeds up repeated runs).
                         If it does not exist the index is saved there after build.
        """
        if index_cache is not None and Path(index_cache).exists():
            self._index = faiss.read_index(str(index_cache))
            self._db_meta = db_entries
            print(f"[retrieval] FAISS index loaded from cache: {index_cache}")
            return

        self._load_model()
        feat_dim: int = self._model.feat_dim

        descriptors = np.empty((len(db_entries), feat_dim), dtype="float32")
        for i, entry in enumerate(tqdm(db_entries, desc="Encoding database", ncols=80)):
            descriptors[i] = self._encode(Path(entry["path"]))

        # IndexFlatIP = exact cosine similarity search on L2-normalised vectors
        index = faiss.IndexFlatIP(feat_dim)
        index.add(descriptors)

        self._index = index
        self._db_meta = db_entries

        if index_cache is not None:
            faiss.write_index(index, str(index_cache))
            print(f"[retrieval] FAISS index saved to: {index_cache}")

    def search(self, image_path: Path, k: int = 10) -> list[dict]:
        """Retrieve the top-k closest database images for a query.

        Returns:
            List of dicts sorted by visual_score descending:
            {rank, place_id, lat, lon, path, visual_score}
        """
        if self._index is None:
            raise RuntimeError("Call build_index() before search().")

        query_desc = self._encode(image_path).reshape(1, -1)
        scores, indices = self._index.search(query_desc, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            meta = self._db_meta[int(idx)]
            results.append({
                "rank": rank,
                "place_id": meta["place_id"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "path": meta["path"],
                "visual_score": float(score),
            })
        return results

    def free_vram(self) -> None:
        """Unload the MegaLoc model from GPU to free VRAM for the OCR branch."""
        if self._model is not None and self.device.type == "cuda":
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            print("[retrieval] MegaLoc unloaded from VRAM.")
