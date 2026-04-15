import warnings

from ase import Atoms
from typing import List, Tuple
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# NOTE: still in development


class DefectGenerator:
    def __init__(self, atoms: Atoms):
        self.original_atoms = atoms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pmg_structure = AseAtomsAdaptor.get_structure(atoms)
        self.sga = SpacegroupAnalyzer(self.pmg_structure)
        self.symmetrized_structure = self.sga.get_symmetrized_structure()

    def get_unique_sites_info(self) -> dict[int, dict]:
        unique_sites = {}
        equivalents = self.symmetrized_structure.equivalent_indices

        for eq_group in equivalents:
            rep_idx = eq_group[0]
            site = self.pmg_structure[rep_idx]
            unique_sites[rep_idx] = {
                "element": site.specie.symbol,
                "multiplicity": len(eq_group),
                "equivalent_indices": eq_group,
                "wyckoff": self.sga.get_symmetry_dataset()["wyckoffs"][rep_idx],
            }
        return unique_sites

    def generate_vacancies(
        self, supercell: Tuple[int, int, int] = (1, 1, 1)
    ) -> List[dict]:
        unique_sites = self.get_unique_sites_info()
        defects = []

        for idx, info in unique_sites.items():
            sc_structure = self.pmg_structure.copy()
            sc_structure.make_supercell(supercell)

            defect_sc = sc_structure.copy()
            defect_sc.remove_sites([idx])

            defect_ase = AseAtomsAdaptor.get_atoms(defect_sc)

            defect_name = f"Vac_{info['element']}_{idx}"
            defect_ase.info["defect_type"] = "vacancy"
            defect_ase.info["defect_site_index"] = idx
            defect_ase.info["defect_name"] = defect_name

            defects.append(
                {"name": defect_name, "atoms": defect_ase, "site_info": info}
            )

        return defects

    def generate_substitutions(
        self,
        substitutions: dict[str, str | List[str]],
        supercell: Tuple[int, int, int] = (1, 1, 1),
    ) -> List[dict]:
        unique_sites = self.get_unique_sites_info()
        defects = []

        for idx, info in unique_sites.items():
            original_el = info["element"]

            if original_el in substitutions:
                targets = substitutions[original_el]
                if isinstance(targets, str):
                    targets = [targets]

                for target_el in targets:
                    sc_structure = self.pmg_structure.copy()
                    sc_structure.make_supercell(supercell)

                    sc_structure.replace(idx, target_el)

                    defect_ase = AseAtomsAdaptor.get_atoms(sc_structure)

                    defect_name = f"Sub_{original_el}_{target_el}_{idx}"
                    defect_ase.info["defect_type"] = "substitution"
                    defect_ase.info["defect_site_index"] = idx
                    defect_ase.info["defect_original"] = original_el
                    defect_ase.info["defect_target"] = target_el
                    defect_ase.info["defect_name"] = defect_name

                    defects.append(
                        {"name": defect_name, "atoms": defect_ase, "site_info": info}
                    )

        return defects
