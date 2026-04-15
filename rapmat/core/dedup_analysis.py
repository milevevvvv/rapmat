import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scipy.spatial.distance import pdist, squareform

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
except ImportError:
    StructureMatcher = None  # type: ignore[assignment,misc]

from rapmat.core.dedup import _to_pymatgen, forces_cosine_similarity


@dataclass
class DedupSimulationResult:

    total: int = 0
    kept: int = 0

    dropped_by_vector: int = 0
    rescued_by_pymatgen: int = 0
    rescued_by_forces: int = 0
    final_dropped: int = 0

    pymatgen_comparisons: int = 0
    pymatgen_mismatches: int = 0
    force_comparisons: int = 0
    force_mismatches: int = 0

    kept_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)


def compute_pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    return pdist(vectors, metric="euclidean")


def simulate_deduplication(
    structures: list[dict],
    *,
    threshold: float = 1e-2,
    use_pymatgen: bool = False,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    use_forces: bool = False,
    force_cosine_threshold: float = 0.95,
    progress_callback=None,
) -> DedupSimulationResult:
    result = DedupSimulationResult(total=len(structures))

    with_vec = [s for s in structures if s.get("vector") is not None]
    if not with_vec:
        result.kept = result.total
        result.kept_ids = [s["id"] for s in structures]
        return result

    with_vec.sort(key=lambda s: s["energy_per_atom"])

    N = len(with_vec)
    mat = np.vstack([s["vector"] for s in with_vec])
    dist_condensed = pdist(mat, metric="euclidean")
    dist_sq = squareform(dist_condensed)

    matcher = None
    if use_pymatgen and StructureMatcher is not None:
        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    dropped = set()

    for i in range(N):
        if progress_callback:
            progress_callback(i, N, False)

        if i in dropped:
            continue

        mask = dist_sq[i] < threshold
        mask[i] = False
        neighbours_idx = np.where(mask)[0]

        for j in neighbours_idx:
            if j in dropped or j <= i:
                continue

            result.dropped_by_vector += 1
            confirmed = True

            if matcher is not None:
                result.pymatgen_comparisons += 1
                try:
                    pmg_i = _to_pymatgen(with_vec[i]["atoms"])
                    pmg_j = _to_pymatgen(with_vec[j]["atoms"])
                    if not matcher.fit(pmg_i, pmg_j):
                        result.pymatgen_mismatches += 1
                        result.rescued_by_pymatgen += 1
                        result.dropped_by_vector -= 1
                        confirmed = False
                except Exception:
                    result.pymatgen_mismatches += 1
                    result.rescued_by_pymatgen += 1
                    result.dropped_by_vector -= 1
                    confirmed = False

            if confirmed and use_forces:
                f_i = with_vec[i].get("forces")
                f_j = with_vec[j].get("forces")
                if f_i is not None and f_j is not None:
                    result.force_comparisons += 1
                    cos_sim = forces_cosine_similarity(f_i, f_j)
                    if cos_sim < force_cosine_threshold:
                        result.force_mismatches += 1
                        result.rescued_by_forces += 1
                        result.dropped_by_vector -= 1
                        confirmed = False
                else:
                    result.rescued_by_forces += 1
                    result.dropped_by_vector -= 1
                    confirmed = False

            if confirmed:
                dropped.add(j)

    if progress_callback:
        progress_callback(N, N, False)

    result.final_dropped = len(dropped)
    result.kept = N - len(dropped)
    result.kept_ids = [with_vec[i]["id"] for i in range(N) if i not in dropped]
    result.dropped_ids = [with_vec[i]["id"] for i in sorted(dropped)]

    no_vec = [s for s in structures if s.get("vector") is None]
    result.kept += len(no_vec)
    result.kept_ids.extend(s["id"] for s in no_vec)
    result.total = len(structures)

    return result


def _greedy_dedup_count(dist_sq: np.ndarray, threshold: float) -> int:
    N = dist_sq.shape[0]
    dropped: set[int] = set()
    for i in range(N):
        if i in dropped:
            continue
        mask = dist_sq[i] < threshold
        mask[i] = False
        for j in np.where(mask)[0]:
            if j > i and j not in dropped:
                dropped.add(j)
    return N - len(dropped)


def find_threshold_for_survival(
    structures: list[dict],
    target_survival_ratio: float,
    max_distance: float,
    tolerance: int | None = None,
) -> tuple[float, int]:
    with_vec = [s for s in structures if s.get("vector") is not None]
    if not with_vec:
        return 0.0, len(structures)

    N = len(with_vec)
    if tolerance is None:
        tolerance = max(1, N // 50)
    target_count = max(1, int(round(N * target_survival_ratio)))

    if target_survival_ratio >= 1.0:
        return 0.0, N

    with_vec_sorted = sorted(with_vec, key=lambda s: s["energy_per_atom"])
    mat = np.vstack([s["vector"] for s in with_vec_sorted])
    dist_sq = squareform(pdist(mat, metric="euclidean"))

    low, high = 0.0, max_distance
    best_thresh, best_kept = 0.0, N

    for _ in range(20):
        mid = (low + high) / 2
        kept = _greedy_dedup_count(dist_sq, mid)

        if abs(kept - target_count) < abs(best_kept - target_count):
            best_thresh = mid
            best_kept = kept

        if kept > target_count:
            low = mid
        else:
            high = mid

    return best_thresh, best_kept


def plot_distance_histogram(
    distances: np.ndarray,
    *,
    threshold: Optional[float] = None,
    save_path: Path | str | None = None,
    title: str = "Pairwise Descriptor Distance Distribution",
    bins: int = 200,
) -> None:
    import matplotlib

    if save_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    dist_sq = squareform(distances)
    np.fill_diagonal(dist_sq, np.inf)
    closest_neighbor = np.min(dist_sq, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(distances, bins=bins, edgecolor="none", alpha=0.75)
    ax1.set_xlabel("L2 Distance")
    ax1.set_ylabel("Pair Count")
    ax1.set_title(title)

    if threshold is not None:
        ax1.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {threshold}",
        )
        ax1.legend()

    ax2.hist(
        closest_neighbor,
        bins=min(bins, len(closest_neighbor) // 2 or 1),
        edgecolor="none",
        alpha=0.75,
    )
    ax2.set_xlabel("L2 Distance to Closest Neighbor")
    ax2.set_ylabel("Structure Count")
    ax2.set_title("Closest Neighbor Distance Distribution")

    if threshold is not None:
        ax2.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {threshold}",
        )
        ax2.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()
