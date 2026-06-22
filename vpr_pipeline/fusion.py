"""
Late-fusion orchestrator: combines MegaLoc visual scores with Ollama geo-validation.

Fusion formula:
    fused_score = α * visual_score_norm + (1 - α) * text_confidence

Dynamic α rule:
    - No OCR text detected → α = 1.0  (trust only the visual branch)
    - Ollama returns low confidence → α stays close to base_alpha
    - Ollama returns high confidence → α is reduced, boosting the text branch
    This ensures the visual branch always acts as a safe fallback when the
    semantic signal is absent or weak (e.g. featureless walls, blurry images).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .retrieval import MegaLocRetriever
from .ocr import SceneTextExtractor, OllamaGeoFilter


@dataclass
class RetrievalCandidate:
    """A single geo-localisation candidate after late fusion and re-ranking.

    Attributes:
        rank: Final rank after fusion (0 = best).
        place_id: Identifier of the matched place in the database.
        lat, lon: GPS coordinates of the matched location.
        db_image_path: Path to the database reference image.
        visual_score: Raw cosine similarity from MegaLoc (higher is better).
        visual_score_norm: visual_score normalised to [0, 1] within the candidate set.
        text_coherent: Whether the LLM judged the OCR texts geographically coherent.
        text_confidence: LLM confidence score in [0, 1].
        fused_score: Final weighted score used for ranking.
        alpha_used: The α value that was applied for this candidate.
        extracted_texts: OCR strings detected in the query image.
    """

    rank: int
    place_id: str
    lat: float
    lon: float
    db_image_path: Path
    visual_score: float
    visual_score_norm: float = 0.0
    text_coherent: bool = False
    text_confidence: float = 0.0
    fused_score: float = 0.0
    alpha_used: float = 1.0
    extracted_texts: list[str] = field(default_factory=list)
    # Populated by GeometricVerifier when triggered (None = not run).
    ransac_inliers: int = 0
    geom_confidence: float = 0.0
    geom_verified: Optional[bool] = None


class LateFusionOrchestrator:
    """Hybrid VPR pipeline: MegaLoc visual retrieval + OCR semantic re-ranking.

    Workflow for a single query image:
      1. MegaLoc + FAISS → top_k_visual candidates ranked by cosine similarity.
      2. EasyOCR → scene text strings extracted from the query image.
      3. Ollama/Qwen3 → geographic coherence score for the top_k_ocr candidates.
      4. Dynamic α weighting → fused score → re-ranked final list.

    Args:
        retriever: Initialised MegaLocRetriever with index already built.
        text_extractor: Initialised SceneTextExtractor.
        geo_filter: Initialised OllamaGeoFilter.
        base_alpha: Visual branch weight when text confidence is zero [0, 1].
                    Higher values give MegaLoc more influence.
        top_k_visual: Number of candidates returned by FAISS.
        top_k_ocr: Number of top visual candidates that are sent to the LLM.
                   Capped at top_k_visual. Set to 0 to disable the text branch.
    """

    def __init__(
        self,
        retriever: MegaLocRetriever,
        text_extractor: SceneTextExtractor,
        geo_filter: OllamaGeoFilter,
        base_alpha: float = 0.7,
        top_k_visual: int = 10,
        top_k_ocr: int = 5,
    ) -> None:
        self.retriever = retriever
        self.text_extractor = text_extractor
        self.geo_filter = geo_filter
        self.base_alpha = base_alpha
        self.top_k_visual = top_k_visual
        self.top_k_ocr = min(top_k_ocr, top_k_visual)

    # ------------------------------------------------------------------
    # Dynamic alpha computation
    # ------------------------------------------------------------------

    def _compute_alpha(self, texts: list[str], llm_confidence: float) -> float:
        """Return the visual branch weight α ∈ [0.3, 1.0].

        When OCR detects no text or the LLM is uncertain, α stays at base_alpha.
        When the LLM is highly confident, α is reduced so the text branch
        contributes more to the final score.

        Args:
            texts: OCR outputs from the query image.
            llm_confidence: Confidence returned by OllamaGeoFilter.validate().

        Returns:
            α coefficient for the fused score formula.
        """
        if not texts:
            # No readable text in the scene → rely entirely on visual branch
            return 1.0
        # Reduce visual weight proportionally to LLM confidence.
        # The floor of 0.3 prevents the visual branch from being ignored entirely.
        alpha = self.base_alpha * (1.0 - llm_confidence * 0.5)
        return max(0.3, alpha)

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def run(self, query_image_path: Path) -> list[RetrievalCandidate]:
        """Execute the full hybrid pipeline for a single query image.

        Args:
            query_image_path: Path to the query image file.

        Returns:
            List of RetrievalCandidate sorted by fused_score descending (rank 0 = best).
        """
        query_image_path = Path(query_image_path)

        # ── Step 1: Visual retrieval ──────────────────────────────────
        visual_results = self.retriever.search(query_image_path, k=self.top_k_visual)

        # ── Step 2: Scene text extraction ─────────────────────────────
        query_texts = self.text_extractor.extract(query_image_path)
        if query_texts:
            print(f"[fusion] OCR detected {len(query_texts)} text(s): {query_texts[:5]}")
        else:
            print("[fusion] No text detected — relying on visual branch only.")

        # ── Step 3: Normalise visual scores to [0, 1] ─────────────────
        raw_scores = [r["visual_score"] for r in visual_results]
        s_min, s_max = min(raw_scores), max(raw_scores)
        score_range = (s_max - s_min) if s_max != s_min else 1.0

        # ── Step 4: LLM geo-validation + fusion ───────────────────────
        candidates: list[RetrievalCandidate] = []

        for i, res in enumerate(visual_results):
            vis_norm = (res["visual_score"] - s_min) / score_range

            text_coherent = False
            text_conf = 0.0

            # Run the LLM only on the top-N visual candidates (cost control)
            if i < self.top_k_ocr and query_texts:
                text_coherent, text_conf = self.geo_filter.validate(
                    texts=query_texts,
                    candidate_lat=res["lat"],
                    candidate_lon=res["lon"],
                )

            alpha = self._compute_alpha(query_texts, text_conf)
            fused = alpha * vis_norm + (1.0 - alpha) * text_conf

            candidates.append(
                RetrievalCandidate(
                    rank=i,
                    place_id=str(res["place_id"]),
                    lat=res["lat"],
                    lon=res["lon"],
                    db_image_path=Path(res["path"]),
                    visual_score=res["visual_score"],
                    visual_score_norm=vis_norm,
                    text_coherent=text_coherent,
                    text_confidence=text_conf,
                    fused_score=fused,
                    alpha_used=alpha,
                    extracted_texts=query_texts,
                )
            )

        # ── Step 5: Re-rank by fused score ────────────────────────────
        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        for new_rank, c in enumerate(candidates):
            c.rank = new_rank

        return candidates
