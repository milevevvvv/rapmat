import hashlib
from typing import List

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from rapmat.storage.base import StructureDescriptor


class SOAPDescriptor(StructureDescriptor):
    """
    Structure descriptor using Smooth Overlap of Atomic Positions (SOAP).
    Uses global averaging (average='inner') to produce a single vector per structure.
    """

    def __init__(
        self,
        species: List[str],
        r_cut: float = 6.0,
        n_max: int = 8,
        l_max: int = 6,
        periodic: bool = True,
    ):
        """
        Args:
            species: List of atomic species (elements) present in the structures.
            r_cut: Cutoff radius in Angstroms.
            n_max: Number of radial basis functions.
            l_max: Maximum degree of spherical harmonics.
            periodic: Whether the system is periodic.
        """
        self._soap = SOAP(
            species=species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average="inner",
            sparse=False,
        )
        self._dim = self._soap.get_number_of_features()
        self._species = species
        self._r_cut = r_cut
        self._n_max = n_max
        self._l_max = l_max
        self._periodic = periodic

    def dimension(self) -> int:
        return self._dim

    def compute(self, atoms: Atoms) -> np.ndarray:
        vec = self._soap.create(atoms)
        return vec.flatten()

    def code_version(self) -> str:
        return "1.0"

    def descriptor_id(self) -> str:
        descriptor_name = "SOAPDescriptor"
        params = (
            self.dimension(),
            self._r_cut,
            self._n_max,
            self._l_max,
            self._periodic,
        )
        payload = descriptor_name + repr(params) + self.code_version()
        return hashlib.sha256(payload.encode()).hexdigest()
