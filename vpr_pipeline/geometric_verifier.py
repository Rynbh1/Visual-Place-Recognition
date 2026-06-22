"""
Conditional geometric verifier — based on
"To Match or Not to Match: Revisiting Image Matching for
Reliable Visual Place Recognition" (Sferrazza et al., CVPR 2025).

Why the old pipeline broke
--------------------------
The classic two-stage VPR pipeline (global retrieval → local RANSAC re-ranking)
assumed local matching always corrects global retrieval errors. Sferrazza et al.
showed empirically that this assumption fails with modern retrievers (MegaLoc,
EigenPlaces, SFRS) for two reasons:

  1. Modern global models already place the correct candidate at rank 1 with very
     high reliability under normal conditions.
  2. In adverse conditions (night, rain, seasonal change) local pixel gradients are
     destroyed. RANSAC then finds more inliers for an accidental texture match and
     demotes the correct candidate that the global model had correctly ranked first.

Systematic RANSAC re-ranking therefore degrades performance.

New role: uncertainty estimator, not re-ranker
----------------------------------------------
RANSAC is invoked conditionally and NEVER changes the ranking. It is called only
when the global fusion score is ambiguous (see two triggers below) and its sole
output is a confidence score:

    P(correct) = σ(w · i_q + b)

where i_q is the number of geometric inliers between the query and the TOP-1
database candidate. This probability is fed directly to a kill-switch: if
P(correct) < min_confidence the system abstains and returns no GPS coordinate,
preferring silence over a confident wrong answer.

Ambiguity triggers
------------------
  1. Quantitative doubt   — top fused score < ambiguity_threshold.
  2. Perceptual aliasing  — top-2 scores differ by < uniqueness_min_delta AND
                            the two candidates are geographically far apart
                            (architectural "false twins").

Computational cost note
-----------------------
LightGlue + RANSAC is expensive on edge hardware. Conditional triggering keeps it
off the critical path for the vast majority of easy, high-confidence queries.

Requires: pip install lightglue opencv-python
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


AMBIGUITY_THRESHOLD = 0.55   # trigger 1: top fused score below this
UNIQUENESS_MIN_DELTA = 0.08  # trigger 2: top-2 score gap below this …
ALIASING_MIN_DIST_M = 100.0  # … and the two candidates at least this far apart


@dataclass
class GeomVerificationResult:
    inlier_count: int          # RANSAC inliers between query and top-1 DB image
    confidence: float          # P(correct) = σ(w · i_q + b)
    triggered: bool            # False → scores were unambiguous, RANSAC not invoked
    triggered_reason: str      # "quantitative_doubt" | "perceptual_aliasing" | ""
    kill_switch: bool          # True → system should abstain (no GPS emitted)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class GeometricVerifier:
    """Conditional LightGlue + RANSAC confidence estimator.

    Verifies ONLY the global model's TOP-1 candidate — never reorders the list.
    See module docstring for the theoretical background.

    Args:
        device: Torch device for SuperPoint / LightGlue.
        logistic_w: Slope  w in P(correct) = σ(w · i_q + b). Default 0.05.
        logistic_b: Intercept b.                              Default -2.0.
            → 40 inliers ≈ 50% confidence, 60 inliers ≈ 73%.
        min_confidence: Kill-switch threshold. Localization is rejected below this.
        ambiguity_threshold: Trigger 1 — top fused_score below this value.
        uniqueness_min_delta: Trigger 2 — top-2 score gap below this value.
        max_keypoints: SuperPoint keypoint budget per image.
    """

    def __init__(
        self,
        device: torch.device,
        logistic_w: float = 0.05,
        logistic_b: float = -2.0,
        min_confidence: float = 0.5,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
        uniqueness_min_delta: float = UNIQUENESS_MIN_DELTA,
        max_keypoints: int = 1024,
    ) -> None:
        self.device = device
        self.logistic_w = logistic_w
        self.logistic_b = logistic_b
        self.min_confidence = min_confidence
        self.ambiguity_threshold = ambiguity_threshold
        self.uniqueness_min_delta = uniqueness_min_delta
        self.max_keypoints = max_keypoints
        self._extractor: Optional[object] = None
        self._matcher: Optional[object] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        if self._extractor is not None:
            return
        try:
            from lightglue import LightGlue, SuperPoint  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LightGlue is required for geometric verification. "
                "Install with: pip install lightglue"
            ) from exc

        print("[geom] Loading SuperPoint + LightGlue …")
        self._extractor = (
            SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
        )
        self._matcher = LightGlue(features="superpoint").eval().to(self.device)

    # ------------------------------------------------------------------
    # Inlier counting (query ↔ single DB image)
    # ------------------------------------------------------------------

    def _count_inliers(self, query_path: Path, db_path: Path) -> int:
        """LightGlue matching + RANSAC fundamental-matrix filter → inlier count i_q."""
        import numpy as np
        import cv2
        from lightglue.utils import load_image, rbd  # type: ignore

        img0 = load_image(str(query_path)).to(self.device)
        img1 = load_image(str(db_path)).to(self.device)

        with torch.no_grad():
            feats0 = self._extractor.extract(img0)
            feats1 = self._extractor.extract(img1)
            matches_out = self._matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches_out = [rbd(x) for x in [feats0, feats1, matches_out]]
        matches = matches_out["matches"]  # shape (K, 2)

        if len(matches) < 5:
            return int(len(matches))

        pts0 = feats0["keypoints"][matches[:, 0]].cpu().numpy().astype(np.float32)
        pts1 = feats1["keypoints"][matches[:, 1]].cpu().numpy().astype(np.float32)

        if len(pts0) < 8:
            return len(pts0)

        _, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.5,
            confidence=0.99,
        )
        return int(mask.sum()) if mask is not None else 0

    # ------------------------------------------------------------------
    # Ambiguity detection
    # ------------------------------------------------------------------

    def _detect_ambiguity(self, candidates: list) -> tuple[bool, str]:
        """Return (triggered, reason) based on late-fusion scores."""
        if not candidates:
            return False, ""

        top_score = candidates[0].fused_score

        if top_score < self.ambiguity_threshold:
            return True, "quantitative_doubt"

        if len(candidates) >= 2:
            gap = top_score - candidates[1].fused_score
            if gap < self.uniqueness_min_delta:
                dist = _haversine_m(
                    candidates[0].lat, candidates[0].lon,
                    candidates[1].lat, candidates[1].lon,
                )
                if dist > ALIASING_MIN_DIST_M:
                    return True, "perceptual_aliasing"

        return False, ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        query_path: Path,
        candidates: list,
    ) -> GeomVerificationResult:
        """Conditionally verify the global model's TOP-1 candidate.

        IMPORTANT: this method never reorders `candidates`. It only decides
        whether the localization is trustworthy enough to emit a GPS coordinate.

        Args:
            query_path: Path to the query image.
            candidates:  Ranked list from LateFusionOrchestrator (rank 0 = best).

        Returns:
            GeomVerificationResult with i_q, P(correct), and kill_switch flag.
        """
        triggered, reason = self._detect_ambiguity(candidates)

        if not triggered:
            return GeomVerificationResult(
                inlier_count=0,
                confidence=1.0,
                triggered=False,
                triggered_reason="",
                kill_switch=False,
            )

        print(f"[geom] RANSAC triggered — reason: {reason}")
        self._load_models()

        # Verify ONLY the top-1 candidate (never re-rank).
        top = candidates[0]
        inlier_count = 0
        if Path(top.db_image_path).exists():
            try:
                inlier_count = self._count_inliers(query_path, Path(top.db_image_path))
            except Exception as exc:
                print(f"[geom] Warning: matching failed for {top.db_image_path.name}: {exc}")

        confidence = _sigmoid(self.logistic_w * inlier_count + self.logistic_b)
        kill = confidence < self.min_confidence

        label = "[KILL-SWITCH — abstaining]" if kill else "[VALIDATED]"
        print(
            f"[geom] i_q={inlier_count} inliers → "
            f"P(correct)={confidence:.3f}  {label}"
        )

        return GeomVerificationResult(
            inlier_count=inlier_count,
            confidence=confidence,
            triggered=True,
            triggered_reason=reason,
            kill_switch=kill,
        )
