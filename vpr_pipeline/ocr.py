"""
OCR branch: scene text extraction + Ollama/Qwen3 geographic coherence filter.

VRAM budget (RTX 4060, 8 GB):
  - EasyOCR with GPU enabled: ~700 MB (CRAFT detector + CRNN recogniser).
  - The reader is lazy-loaded on the first extract() call.
  - Call free_vram() after text extraction if MegaLoc also needs to run
    in the same pipeline pass (they share the 8 GB budget).
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Optional

import requests

# Vocabulary for decoding text predictions from TextInPlace
voc = list(string.printable[:-6])


def rec_decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < len(voc):
            s += voc[c]
        elif c == len(voc):
            return s
        else:
            s += u''
    return s


def get_detected_text(predictions):
    if len(predictions) == 0:
        return []
    pred = predictions[0]  # batch size is 1
    if "instances" not in pred:
        return []
    instances = pred["instances"].to("cpu")
    if not hasattr(instances, "recs"):
        return []
    rec_strings = []
    for rec in instances.recs:
        rec_strings.append(rec_decode(rec))
    return rec_strings


class SceneTextExtractor:
    """GPU-accelerated scene text extraction using TextInPlace or EasyOCR.

    If weights_path is provided, wraps TextInPlace. Otherwise, wraps EasyOCR's
    unified pipeline (CRAFT detection + CRNN recognition) and filters
    low-confidence detections before returning clean strings.

    Args:
        device: torch.device — GPU acceleration is enabled when device.type == "cuda".
        languages: ISO 639-1 codes passed to EasyOCR (default: French + English).
        confidence_threshold: Minimum recognition confidence to accept a detection.
        weights_path: Path to TextInPlace fine-tuned model weights.
    """

    def __init__(
        self,
        device,
        languages: Optional[list[str]] = None,
        confidence_threshold: float = 0.4,
        weights_path: Optional[Path] = None,
    ) -> None:
        self.device = device
        self.languages = languages or ["fr", "en"]
        self.confidence_threshold = confidence_threshold
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self._reader = None  # lazy — EasyOCR reader
        self._model = None   # lazy — TextInPlace model

    def _load_reader(self) -> None:
        """Instantiate EasyOCR reader (idempotent)."""
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is required: pip install easyocr"
            ) from exc

        # gpu=True routes the CRAFT and CRNN models to CUDA (~700 MB VRAM total)
        self._reader = easyocr.Reader(
            self.languages,
            gpu=(self.device.type == "cuda"),
            verbose=False,
        )

    def _load_textinplace(self) -> None:
        """Instantiate TextInPlace and transfer weights to device (idempotent)."""
        if self._model is not None:
            return

        import sys
        import torch
        from collections import OrderedDict

        weights_path = self.weights_path
        if not weights_path.exists() and str(weights_path).startswith("/TextInPlace"):
            project_root = Path(__file__).resolve().parents[1]
            alt_path = project_root / str(weights_path).lstrip("/")
            if alt_path.exists():
                weights_path = alt_path

        # Setup sys.path for TextInPlace imports
        project_root = Path(__file__).resolve().parents[1]
        repo_path = project_root / "TextInPlace" / "repo"
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        if str(repo_path / "detectron2") not in sys.path:
            sys.path.insert(0, str(repo_path / "detectron2"))

        from network import STVGLNet_test
        from backbone import setup_cfg

        print(f"[ocr] Loading TextInPlace weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Auto-detect VPR features dimension
        features_dim = 16384
        for k, v in state_dict.items():
            if k.endswith("aggregation.fc.weight"):
                detected_row_dim = v.shape[0]
                features_dim = detected_row_dim * 512
                break

        # Setup configuration mock object
        class MockArgs:
            def __init__(self, config_file, opts, confidence_threshold, features_dim):
                self.config_file = config_file
                self.opts = opts
                self.confidence_threshold = confidence_threshold
                self.features_dim = features_dim

        config_file = str(repo_path / "configs" / "Bridge" / "TotalText" / "R_50_poly.yaml")
        mock_args = MockArgs(
            config_file=config_file,
            opts=[],
            confidence_threshold=self.confidence_threshold,
            features_dim=features_dim
        )

        cfg = setup_cfg(mock_args)
        model = STVGLNet_test(cfg)

        if list(state_dict.keys())[0].startswith('module'):
            state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith("backbone.textmodel.") and (k.startswith("dptext_detr.") or k.startswith("recognizer.") or k.startswith("bridge.")):
                new_state_dict[f"backbone.textmodel.{k}"] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()

        self._model = model

    def extract(self, image_path: Path) -> list[str]:
        """Return cleaned text strings detected in the image.

        Very short tokens (< 2 chars) and low-confidence detections are removed
        to reduce LLM hallucination noise downstream.

        Args:
            image_path: Path to the image file.

        Returns:
            List of text strings, may be empty if no text is detected.
        """
        if self.weights_path is not None:
            self._load_textinplace()
            import torchvision.transforms as T
            from PIL import Image
            import torch

            img = Image.open(image_path).convert("RGB")
            val_transform = T.Compose([
                T.Resize((320, 320), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = val_transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions, _ = self._model(tensor)

            raw_texts = get_detected_text(predictions)
            texts = [
                text.strip()
                for text in raw_texts
                if len(text.strip()) >= 2
            ]
            return texts
        else:
            self._load_reader()
            raw = self._reader.readtext(str(image_path), detail=1)
            texts = [
                text.strip()
                for (_bbox, text, conf) in raw
                if conf >= self.confidence_threshold and len(text.strip()) >= 2
            ]
            return texts

    def free_vram(self) -> None:
        """Delete reader/model tensors and release CUDA memory."""
        import torch
        if self._reader is not None:
            del self._reader
            self._reader = None
        if self._model is not None:
            del self._model
            self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("[ocr] Text extractor unloaded from VRAM.")


class OllamaGeoFilter:
    """Geographic coherence filter powered by a local Ollama LLM (Qwen3).

    For each candidate location the LLM judges whether the OCR-extracted street
    text (shop names, street signs, bus routes…) is plausible for that GPS zone.

    Args:
        model: Ollama model tag, e.g. "qwen3:8b" (8 B param version needs ~5 GB RAM).
        base_url: Ollama server endpoint (default: localhost, standard port).
        zone_description: Human-readable region label for the prompt context.
        timeout_s: HTTP timeout per LLM call in seconds.
    """

    _PROMPT_TEMPLATE = (
        "You are a geolocalisation expert specialised in {zone}.\n"
        "The following text was extracted by OCR from a street-level photo "
        "taken near coordinates ({lat:.5f}, {lon:.5f}).\n\n"
        "OCR texts:\n{text_list}\n\n"
        "Does this text make geographic sense for this location? "
        "Consider street names, shop names, transit lines, and French signage conventions.\n"
        "Reply ONLY with a JSON object on a single line: "
        '{{\"coherent\": true or false, \"confidence\": 0.0 to 1.0, \"reason\": \"brief explanation\"}}'
    )

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        zone_description: str = "Paris, France",
        timeout_s: int = 30,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.zone_description = zone_description
        self.timeout_s = timeout_s

    def validate(
        self,
        texts: list[str],
        candidate_lat: float,
        candidate_lon: float,
    ) -> tuple[bool, float]:
        """Query the LLM to assess geographic coherence of OCR texts.

        Args:
            texts: List of strings extracted by SceneTextExtractor.
            candidate_lat: Latitude of the candidate database location.
            candidate_lon: Longitude of the candidate database location.

        Returns:
            Tuple of (is_coherent: bool, confidence: float in [0, 1]).
            Returns (False, 0.0) if texts is empty or if Ollama is unreachable.
        """
        if not texts:
            return False, 0.0

        # Cap at 20 texts to stay within Ollama's context window comfortably
        text_list = "\n".join(f"- {t}" for t in texts[:20])
        prompt = self._PROMPT_TEMPLATE.format(
            zone=self.zone_description,
            lat=candidate_lat,
            lon=candidate_lon,
            text_list=text_list,
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            raw_text: str = resp.json()["response"]

            # The model may emit reasoning tokens before the JSON; extract the object
            match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            if not match:
                return False, 0.0

            data = json.loads(match.group())
            coherent = bool(data.get("coherent", False))
            confidence = float(data.get("confidence", 0.0))
            return coherent, max(0.0, min(1.0, confidence))

        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError):
            # Ollama unavailable or malformed output → neutral fallback
            return False, 0.0

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to a health check."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
