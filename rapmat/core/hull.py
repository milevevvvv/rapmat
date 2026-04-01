"""Convex hull of thermodynamic stability.

All functions are pure readers — they query the store but never mutate it.
Formation energies and reference energies are computed on-the-fly (never cached).
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from rapmat.utils.common import parse_system

from rapmat.storage.base import StructureStore

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def get_composition_fraction(formula: dict[str, int], element: str) -> float:
    """Return the atomic fraction of *element* in *formula*.

    >>> get_composition_fraction({"Al": 2, "O": 3}, "Al")
    0.4
    """
    total = sum(formula.values())
    return formula.get(element, 0) / total if total else 0.0


# ------------------------------------------------------------------ #
#  Reference energies
# ------------------------------------------------------------------ #


def _study_has_pressure(runs: list[dict]) -> bool:
    """Return True if any run in the study was performed under pressure."""
    return any(run["config"].get("pressure_gpa", 0.0) > 0 for run in runs)


def _effective_epa(s: dict, use_enthalpy: bool) -> float:
    """Return enthalpy_per_atom when available and requested, else energy_per_atom."""
    if use_enthalpy and s.get("enthalpy_per_atom") is not None:
        return s["enthalpy_per_atom"]
    return s["energy_per_atom"]


def get_reference_energies(
    store: StructureStore,
    study_id: str,
    *,
    use_enthalpy: bool = False,
) -> dict[str, float]:
    """Return ``{element: value_per_atom}`` for each element in the study.

    When *use_enthalpy* is True the reference value is the minimum
    ``enthalpy_per_atom``; otherwise it is the minimum ``energy_per_atom``.

    Raises
    ------
    ValueError
        If a pure-element run is missing or has no relaxed structures.
    """
    study = store.get_study(study_id)
    if study is None:
        raise ValueError(f"Study '{study_id}' not found.")

    elements = parse_system(study["system"])
    runs = store.get_study_runs(study_id)

    ref_energies: dict[str, float] = {}
    for el in elements:
        best_epa: float | None = None
        for run in runs:
            run_formula: dict = run["config"].get("formula", {})
            if set(run_formula.keys()) != {el}:
                continue
            structures = store.get_run_structures(run["name"], status="relaxed")
            for s in structures:
                epa = _effective_epa(s, use_enthalpy)
                if best_epa is None or epa < best_epa:
                    best_epa = epa

        if best_epa is None:
            raise ValueError(
                f"No relaxed pure-{el} structures found in study '{study_id}'."
            )
        ref_energies[el] = best_epa

    return ref_energies


# ------------------------------------------------------------------ #
#  Phase diagram construction
# ------------------------------------------------------------------ #


def build_phase_diagram(
    store: StructureStore,
    study_id: str,
    symprec: float = 1e-3,
    *,
    show_all: bool = False,
) -> tuple[PhaseDiagram, list[dict], bool]:
    """Build a pymatgen ``PhaseDiagram`` from study data.

    Parameters
    ----------
    store
        SurrealDB storage backend.
    study_id
        Identifier of the study to build the hull for.
    symprec
        Symmetry precision for space-group labels.
    show_all
        If *True*, include every relaxed structure in the returned list
        (not just the ground state per composition).

    Returns
    -------
    pd : PhaseDiagram
        The pymatgen phase diagram object.
    structure_data : list[dict]
        Per-structure records used for plotting / reporting.
    use_enthalpy : bool
        Whether enthalpy was used (True when any run has pressure > 0).

    Raises
    ------
    ValueError
        On invalid system size, missing endpoints, or insufficient data.
    """
    study = store.get_study(study_id)
    if study is None:
        raise ValueError(f"Study '{study_id}' not found.")

    elements = parse_system(study["system"])
    if len(elements) < 2:
        raise ValueError(
            "Convex hull requires a binary or larger system (2+ elements)."
        )
    if len(elements) > 2:
        raise ValueError(
            "Only binary systems are supported for now. Ternary support is planned."
        )

    runs = store.get_study_runs(study_id)
    use_enthalpy = _study_has_pressure(runs)
    ref_energies = get_reference_energies(store, study_id, use_enthalpy=use_enthalpy)

    entries: list[PDEntry] = []
    structure_data: list[dict] = []
    compositions_seen: set[str] = set()

    for run in runs:
        structures = store.get_run_structures(
            run["name"], status="relaxed", symprec=symprec
        )
        for s in structures:
            formula_str = s["formula"]
            comp = Composition(formula_str)

            epa = _effective_epa(s, use_enthalpy)
            n_atoms = int(comp.num_atoms)
            total_value = epa * n_atoms
            entries.append(PDEntry(comp, total_value))

            e_ref = sum(comp[el_str] * ref_energies[el_str] for el_str in ref_energies)
            formation_energy = (total_value - e_ref) / n_atoms

            reduced = comp.reduced_formula
            compositions_seen.add(reduced)

            structure_data.append(
                {
                    "formula": formula_str,
                    "reduced_formula": reduced,
                    "composition_frac": comp.get_atomic_fraction(elements[1]),
                    "energy_per_atom": s["energy_per_atom"],
                    "enthalpy_per_atom": s.get("enthalpy_per_atom"),
                    "effective_per_atom": epa,
                    "effective_total": total_value,
                    "formation_energy": formation_energy,
                    "run_name": run["name"],
                    "structure_id": s["id"],
                }
            )

    pure_formulas = {el for el in elements}
    intermediate = compositions_seen - pure_formulas
    if not intermediate:
        raise ValueError(
            "Need at least one intermediate composition between pure endpoints. "
            "Only pure-element runs found."
        )

    pd = PhaseDiagram(entries)

    for sd in structure_data:
        entry = PDEntry(Composition(sd["formula"]), sd["effective_total"])
        sd["energy_above_hull"] = pd.get_e_above_hull(entry)
        sd["is_stable"] = sd["energy_above_hull"] < 1e-6

    if not show_all:
        best: dict[str, dict] = {}
        for sd in structure_data:
            key = sd["reduced_formula"]
            if (
                key not in best
                or sd["formation_energy"] < best[key]["formation_energy"]
            ):
                best[key] = sd
        structure_data = list(best.values())

    structure_data.sort(key=lambda d: d["composition_frac"])
    return pd, structure_data, use_enthalpy


# ------------------------------------------------------------------ #
#  Binary hull plotting
# ------------------------------------------------------------------ #


def plot_binary_hull(
    structure_data: list[dict],
    system: str,
    *,
    save_path: Optional[Path] = None,
    show: bool = True,
    use_enthalpy: bool = False,
) -> Figure:
    """Plot a binary convex hull of formation energy/enthalpy vs composition.

    Parameters
    ----------
    structure_data
        Records produced by :func:`build_phase_diagram`.
    system
        Chemical system string, e.g. ``"Al-O"``.
    save_path
        If given, save the figure to this path.
    show
        If *True*, call ``plt.show()`` (set *False* in headless environments).
    use_enthalpy
        If *True*, label the y-axis as "Formation enthalpy" instead of
        "Formation energy".

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    elements = parse_system(system)
    if len(elements) != 2:
        raise ValueError("plot_binary_hull only supports binary systems.")

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = np.array([d["composition_frac"] for d in structure_data])
    ys = np.array([d["formation_energy"] for d in structure_data])
    stable = np.array([d["is_stable"] for d in structure_data])

    # Plot unstable points
    if (~stable).any():
        ax.scatter(
            xs[~stable],
            ys[~stable],
            marker="o",
            s=50,
            facecolors="none",
            edgecolors="#999999",
            linewidths=1.0,
            zorder=3,
            label="Unstable",
        )
        for x, y, eah in zip(
            xs[~stable],
            ys[~stable],
            [d["energy_above_hull"] for d, s in zip(structure_data, stable) if not s],
        ):
            hull_y = y - eah
            ax.plot(
                [x, x],
                [hull_y, y],
                color="#cccccc",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )

    # Plot stable points and hull line
    if stable.any():
        hull_xs = np.concatenate([[0.0], xs[stable], [1.0]])
        hull_ys = np.concatenate([[0.0], ys[stable], [0.0]])
        order = np.argsort(hull_xs)
        ax.plot(
            hull_xs[order],
            hull_ys[order],
            color="#2176ff",
            linewidth=1.5,
            zorder=2,
            label="Hull",
        )
        ax.scatter(
            xs[stable],
            ys[stable],
            marker="o",
            s=70,
            color="#2176ff",
            zorder=4,
            label="Stable",
        )
        for sd in (d for d, s in zip(structure_data, stable) if s):
            ax.annotate(
                sd["reduced_formula"],
                (sd["composition_frac"], sd["formation_energy"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
    else:
        ax.plot([0, 1], [0, 0], color="#2176ff", linewidth=1.5, zorder=2)

    quantity = "enthalpy" if use_enthalpy else "energy"
    ax.axhline(0, color="black", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"$x$ in {elements[0]}$_{{1-x}}${elements[1]}$_x$")
    ax.set_ylabel(f"Formation {quantity} (eV/atom)")
    ax.set_title(f"Convex hull — {elements[0]}-{elements[1]}")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
