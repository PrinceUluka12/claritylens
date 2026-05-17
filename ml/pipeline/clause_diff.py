# ml/pipeline/clause_diff.py
# ============================================================
# Compares two versions of a contract and identifies
# added, removed, and modified clauses.
#
# Used by the frontend's clause diff viewer panel.
# Pure Python + numpy — no model forward passes needed
# for the diff itself. Classifier runs only on changed chunks.
# ============================================================

import time
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


# ── Constants ─────────────────────────────────────────────────
# Cosine similarity threshold for "same clause"
# Above this → same clause (possibly modified)
# Below this → different clauses entirely
SIMILARITY_THRESHOLD = 0.85

# Below this similarity, a matched clause is "modified"
# Above this → essentially unchanged
MODIFIED_THRESHOLD = 0.95


@dataclass
class ClauseDiff:
    """Represents a single clause change between two versions."""
    change_type:    str      # "added" | "removed" | "modified" | "unchanged"
    chunk_index_v1: int      # position in version 1 (-1 if added)
    chunk_index_v2: int      # position in version 2 (-1 if removed)
    text_v1:        str      # clause text in version 1 (empty if added)
    text_v2:        str      # clause text in version 2 (empty if removed)
    similarity:     float    # cosine similarity between versions
    risk_label_v1:  str = "" # risk label in version 1
    risk_label_v2:  str = "" # risk label in version 2
    risk_changed:   bool = False  # did risk classification change?


@dataclass
class DiffResult:
    """Complete diff between two document versions."""
    filename_v1:    str
    filename_v2:    str
    diffs:          list[ClauseDiff]
    added_count:    int
    removed_count:  int
    modified_count: int
    unchanged_count: int
    risk_delta:     float    # positive = more risky, negative = less risky
    inference_ms:   float


class ClauseDiffer:
    """
    Computes clause-level diffs between contract versions.
    Uses Word2Vec embeddings for semantic matching.
    """

    def __init__(self) -> None:
        # Lazy-load embedder — only needed when diff is called
        self._embedder = None
        logger.info("ClauseDiffer initialized")

    def _get_embedder(self):
        """Lazy-loads the embedder singleton."""
        if self._embedder is None:
            from ml.inference.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _embed_chunks(self, chunks: list) -> np.ndarray:
        """
        Embeds a list of chunks into a matrix.
        Shape: [num_chunks, embedding_dim]
        """
        embedder = self._get_embedder()
        texts = [
            c.text if hasattr(c, "text") else c.get("text", "")
            for c in chunks
        ]
        return embedder.embed_batch(texts)

    def _build_similarity_matrix(
        self,
        embeddings_v1: np.ndarray,
        embeddings_v2: np.ndarray,
    ) -> np.ndarray:
        """
        Computes pairwise cosine similarity between all chunks.
        Since embeddings are unit-normalized, this is a dot product.

        Returns matrix of shape [len_v1, len_v2].
        Entry [i, j] = similarity between chunk i in v1
                       and chunk j in v2.
        """
        # CPU NOTE: numpy matmul on unit vectors = cosine similarity
        # Much faster than computing norms repeatedly
        return embeddings_v1 @ embeddings_v2.T

    def diff(
        self,
        chunks_v1:      list,
        chunks_v2:      list,
        filename_v1:    str = "version_1",
        filename_v2:    str = "version_2",
        risk_results_v1: list = None,
        risk_results_v2: list = None,
    ) -> DiffResult:
        """
        Computes clause-level diff between two document versions.

        Args:
            chunks_v1:       Chunk objects from version 1
            chunks_v2:       Chunk objects from version 2
            filename_v1:     Name of version 1
            filename_v2:     Name of version 2
            risk_results_v1: ClauseResult objects for version 1
            risk_results_v2: ClauseResult objects for version 2

        Returns:
            DiffResult with per-clause change annotations
        """
        start_time = time.time()

        if not chunks_v1 and not chunks_v2:
            logger.warning("diff() called with empty chunk lists")
            return DiffResult(
                filename_v1    = filename_v1,
                filename_v2    = filename_v2,
                diffs          = [],
                added_count    = 0,
                removed_count  = 0,
                modified_count = 0,
                unchanged_count = 0,
                risk_delta     = 0.0,
                inference_ms   = 0.0,
            )

        # Handle edge cases where one version is empty
        if not chunks_v1:
            diffs = [
                ClauseDiff(
                    change_type    = "added",
                    chunk_index_v1 = -1,
                    chunk_index_v2 = i,
                    text_v1        = "",
                    text_v2        = c.text if hasattr(c, "text") else "",
                    similarity     = 0.0,
                )
                for i, c in enumerate(chunks_v2)
            ]
            return DiffResult(
                filename_v1     = filename_v1,
                filename_v2     = filename_v2,
                diffs           = diffs,
                added_count     = len(diffs),
                removed_count   = 0,
                modified_count  = 0,
                unchanged_count = 0,
                risk_delta      = 0.0,
                inference_ms    = (time.time() - start_time) * 1000,
            )

        if not chunks_v2:
            diffs = [
                ClauseDiff(
                    change_type    = "removed",
                    chunk_index_v1 = i,
                    chunk_index_v2 = -1,
                    text_v1        = c.text if hasattr(c, "text") else "",
                    text_v2        = "",
                    similarity     = 0.0,
                )
                for i, c in enumerate(chunks_v1)
            ]
            return DiffResult(
                filename_v1     = filename_v1,
                filename_v2     = filename_v2,
                diffs           = diffs,
                added_count     = 0,
                removed_count   = len(diffs),
                modified_count  = 0,
                unchanged_count = 0,
                risk_delta      = 0.0,
                inference_ms    = (time.time() - start_time) * 1000,
            )

        # ── Step 1: Embed both versions ───────────────────────
        logger.info(
            f"Diffing {filename_v1} ({len(chunks_v1)} chunks) vs "
            f"{filename_v2} ({len(chunks_v2)} chunks)"
        )
        embeddings_v1 = self._embed_chunks(chunks_v1)
        embeddings_v2 = self._embed_chunks(chunks_v2)

        # ── Step 2: Build similarity matrix ───────────────────
        sim_matrix = self._build_similarity_matrix(
            embeddings_v1, embeddings_v2
        )  # shape: [len_v1, len_v2]

        # ── Step 3: Greedy matching ───────────────────────────
        # Match each v1 chunk to its best v2 chunk if similarity
        # is above threshold. Greedy = first-come-first-served.
        # Good enough for contract diffs where clause order is
        # roughly preserved across versions.
        matched_v1 = {}   # v1_idx → v2_idx
        matched_v2 = set()

        # Sort by similarity descending so best matches go first
        pairs = []
        for i in range(len(chunks_v1)):
            for j in range(len(chunks_v2)):
                pairs.append((sim_matrix[i, j], i, j))
        pairs.sort(reverse=True)

        for sim, i, j in pairs:
            if sim < SIMILARITY_THRESHOLD:
                break  # sorted — no point checking lower similarities
            if i not in matched_v1 and j not in matched_v2:
                matched_v1[i] = j
                matched_v2.add(j)

        # ── Step 4: Build risk label lookups ──────────────────
        risk_v1 = {}
        risk_v2 = {}

        if risk_results_v1:
            for r in risk_results_v1:
                idx = r.chunk_index if hasattr(r, "chunk_index") else 0
                risk_v1[idx] = r.risk_label if hasattr(r, "risk_label") else ""

        if risk_results_v2:
            for r in risk_results_v2:
                idx = r.chunk_index if hasattr(r, "chunk_index") else 0
                risk_v2[idx] = r.risk_label if hasattr(r, "risk_label") else ""

        # ── Step 5: Build diff objects ────────────────────────
        diffs = []

        # Process all v1 chunks
        for i, chunk_v1 in enumerate(chunks_v1):
            text_v1 = chunk_v1.text if hasattr(chunk_v1, "text") else ""

            if i in matched_v1:
                j        = matched_v1[i]
                chunk_v2 = chunks_v2[j]
                text_v2  = chunk_v2.text if hasattr(chunk_v2, "text") else ""
                sim      = float(sim_matrix[i, j])

                # Determine change type based on similarity
                if sim >= MODIFIED_THRESHOLD:
                    change_type = "unchanged"
                else:
                    change_type = "modified"

                rl_v1       = risk_v1.get(i, "")
                rl_v2       = risk_v2.get(j, "")
                risk_changed = (rl_v1 != rl_v2 and
                                bool(rl_v1) and bool(rl_v2))

                diffs.append(ClauseDiff(
                    change_type    = change_type,
                    chunk_index_v1 = i,
                    chunk_index_v2 = j,
                    text_v1        = text_v1,
                    text_v2        = text_v2,
                    similarity     = round(sim, 4),
                    risk_label_v1  = rl_v1,
                    risk_label_v2  = rl_v2,
                    risk_changed   = risk_changed,
                ))
            else:
                # No match found — clause was removed
                diffs.append(ClauseDiff(
                    change_type    = "removed",
                    chunk_index_v1 = i,
                    chunk_index_v2 = -1,
                    text_v1        = text_v1,
                    text_v2        = "",
                    similarity     = 0.0,
                    risk_label_v1  = risk_v1.get(i, ""),
                ))

        # Process unmatched v2 chunks (added clauses)
        for j, chunk_v2 in enumerate(chunks_v2):
            if j not in matched_v2:
                text_v2 = chunk_v2.text if hasattr(chunk_v2, "text") else ""
                diffs.append(ClauseDiff(
                    change_type    = "added",
                    chunk_index_v1 = -1,
                    chunk_index_v2 = j,
                    text_v1        = "",
                    text_v2        = text_v2,
                    similarity     = 0.0,
                    risk_label_v2  = risk_v2.get(j, ""),
                ))

        # ── Step 6: Compute risk delta ─────────────────────────
        # Positive = v2 is riskier, negative = v2 is safer
        risk_delta = self._compute_risk_delta(diffs)

        # ── Step 7: Count change types ────────────────────────
        added_count     = sum(1 for d in diffs if d.change_type == "added")
        removed_count   = sum(1 for d in diffs if d.change_type == "removed")
        modified_count  = sum(1 for d in diffs if d.change_type == "modified")
        unchanged_count = sum(1 for d in diffs if d.change_type == "unchanged")

        inference_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Diff complete — "
            f"added={added_count}, removed={removed_count}, "
            f"modified={modified_count}, unchanged={unchanged_count}, "
            f"risk_delta={risk_delta:+.3f} | "
            f"{inference_ms:.1f}ms"
        )

        return DiffResult(
            filename_v1     = filename_v1,
            filename_v2     = filename_v2,
            diffs           = diffs,
            added_count     = added_count,
            removed_count   = removed_count,
            modified_count  = modified_count,
            unchanged_count = unchanged_count,
            risk_delta      = risk_delta,
            inference_ms    = round(inference_ms, 1),
        )

    def _compute_risk_delta(self, diffs: list) -> float:
        """
        Estimates whether version 2 is more or less risky.

        Logic:
        - Added high-risk clauses → positive delta (riskier)
        - Removed high-risk clauses → negative delta (safer)
        - Modified clauses with risk label change → small delta
        """
        # Risk weight per label — same as risk_scorer
        risk_weights = {
            "indemnity":     0.25,
            "liability_cap": 0.20,
            "ip_assignment": 0.20,
            "termination":   0.15,
            "non_compete":   0.12,
            "data_privacy":  0.08,
        }

        delta = 0.0

        for diff in diffs:
            if diff.change_type == "added":
                label  = diff.risk_label_v2
                weight = risk_weights.get(label, 0.05)
                delta += weight   # added clause increases risk

            elif diff.change_type == "removed":
                label  = diff.risk_label_v1
                weight = risk_weights.get(label, 0.05)
                delta -= weight   # removed clause decreases risk

            elif diff.change_type == "modified" and diff.risk_changed:
                # Risk label changed — small signal
                delta += 0.02

        return round(delta, 4)

    def diff_summary(self, result: DiffResult) -> dict:
        """
        Returns a JSON-serializable summary for the API.
        Only includes changed clauses — not unchanged ones.
        """
        changed_diffs = [
            {
                "change_type":    d.change_type,
                "chunk_index_v1": d.chunk_index_v1,
                "chunk_index_v2": d.chunk_index_v2,
                "text_v1":        d.text_v1[:300],
                "text_v2":        d.text_v2[:300],
                "similarity":     d.similarity,
                "risk_label_v1":  d.risk_label_v1,
                "risk_label_v2":  d.risk_label_v2,
                "risk_changed":   d.risk_changed,
            }
            for d in result.diffs
            if d.change_type != "unchanged"
        ]

        return {
            "filename_v1":    result.filename_v1,
            "filename_v2":    result.filename_v2,
            "added":          result.added_count,
            "removed":        result.removed_count,
            "modified":       result.modified_count,
            "unchanged":      result.unchanged_count,
            "risk_delta":     result.risk_delta,
            "risk_direction": (
                "increased" if result.risk_delta > 0.05
                else "decreased" if result.risk_delta < -0.05
                else "unchanged"
            ),
            "changes":        changed_diffs,
            "inference_ms":   result.inference_ms,
        }


# Singleton
_differ_instance = None


def get_clause_differ() -> ClauseDiffer:
    global _differ_instance
    if _differ_instance is None:
        _differ_instance = ClauseDiffer()
    return _differ_instance