"""SurrealDB storage backend for structure search and persistence.

Connection modes supported via the sync Python SDK:
- Embedded persistent: ``file://<path>``
- Embedded in-memory:  ``mem://``
- Dedicated server:    ``ws://<host>:<port>/rpc`` (requires username/password)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io.jsonio import decode as ase_decode
from ase.io.jsonio import encode as ase_encode
from rapmat.storage.base import StructureStore
from rapmat.utils.console import console as default_console
from rapmat.utils.structure import format_spg
from rich.console import Console
from surrealdb import RecordID, Surreal

# ------------------------------------------------------------------ #
#  Schema DDL (idempotent — safe to run on every connection)
# ------------------------------------------------------------------ #


# TODO: introduce new indexes when necessary
_SCHEMA_DDL = """\
DEFINE TABLE IF NOT EXISTS study SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS system     ON study TYPE string;
DEFINE FIELD IF NOT EXISTS domain     ON study TYPE string;
DEFINE FIELD IF NOT EXISTS calculator ON study TYPE string;
DEFINE FIELD IF NOT EXISTS config_json ON study TYPE string;
DEFINE FIELD IF NOT EXISTS timestamp  ON study TYPE string;

DEFINE TABLE IF NOT EXISTS run SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name        ON run TYPE string;
DEFINE FIELD IF NOT EXISTS batch_config_json ON run TYPE string;
DEFINE FIELD IF NOT EXISTS timestamp   ON run TYPE string;
DEFINE FIELD IF NOT EXISTS study       ON run TYPE record<study>;
DEFINE FIELD IF NOT EXISTS run_status  ON run TYPE option<string>
    ASSERT $value IS NONE OR $value IN [
        "pending", "generating", "processing", "completed", "failed", "interrupted"
    ];
DEFINE FIELD IF NOT EXISTS worker_id   ON run TYPE option<string>;
DEFINE FIELD IF NOT EXISTS heartbeat   ON run TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_run_name ON run FIELDS name UNIQUE;

DEFINE TABLE IF NOT EXISTS structure SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS run              ON structure TYPE record<run>;
DEFINE FIELD IF NOT EXISTS status           ON structure TYPE string
    ASSERT $value IN ["generating", "generated", "relaxed", "discarded", "error"];
DEFINE FIELD IF NOT EXISTS gen_spg          ON structure TYPE option<int>;
DEFINE FIELD IF NOT EXISTS gen_fu           ON structure TYPE option<int>;
DEFINE FIELD IF NOT EXISTS formula          ON structure TYPE string;
DEFINE FIELD IF NOT EXISTS energy_per_atom  ON structure TYPE float;
DEFINE FIELD IF NOT EXISTS energy_total     ON structure TYPE float;
DEFINE FIELD IF NOT EXISTS fmax             ON structure TYPE float;
DEFINE FIELD IF NOT EXISTS converged        ON structure TYPE bool;
DEFINE FIELD IF NOT EXISTS thickness        ON structure TYPE option<float>;
DEFINE FIELD IF NOT EXISTS enthalpy_per_atom ON structure TYPE option<float>;
DEFINE FIELD IF NOT EXISTS volume            ON structure TYPE option<float>;
DEFINE FIELD IF NOT EXISTS min_phonon_freq   ON structure TYPE option<float>;
DEFINE FIELD IF NOT EXISTS duplicate          ON structure TYPE option<bool>;
DEFINE FIELD IF NOT EXISTS initial_atoms_json ON structure TYPE string;
DEFINE FIELD IF NOT EXISTS final_atoms_json   ON structure TYPE string;
DEFINE FIELD IF NOT EXISTS timestamp        ON structure TYPE string;
DEFINE INDEX IF NOT EXISTS idx_struct_run        ON structure FIELDS run;
DEFINE INDEX IF NOT EXISTS idx_struct_status     ON structure FIELDS status;
DEFINE INDEX IF NOT EXISTS idx_struct_run_status ON structure FIELDS run, status;

DEFINE TABLE IF NOT EXISTS descriptor SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS col_name     ON descriptor TYPE string;
DEFINE FIELD IF NOT EXISTS dimension    ON descriptor TYPE int;
DEFINE FIELD IF NOT EXISTS desc_type    ON descriptor TYPE string;
DEFINE FIELD IF NOT EXISTS params_json  ON descriptor TYPE string;
DEFINE FIELD IF NOT EXISTS code_version ON descriptor TYPE string;
DEFINE FIELD IF NOT EXISTS timestamp    ON descriptor TYPE string;
DEFINE INDEX IF NOT EXISTS idx_desc_col ON descriptor FIELDS col_name UNIQUE;

DEFINE TABLE IF NOT EXISTS evaluation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS structure       ON evaluation TYPE record<structure>;
DEFINE FIELD IF NOT EXISTS run             ON evaluation TYPE record<run>;
DEFINE FIELD IF NOT EXISTS calculator      ON evaluation TYPE string;
DEFINE FIELD IF NOT EXISTS config_json     ON evaluation TYPE string;
DEFINE FIELD IF NOT EXISTS energy_per_atom ON evaluation TYPE float;
DEFINE FIELD IF NOT EXISTS energy_total    ON evaluation TYPE float;
DEFINE FIELD IF NOT EXISTS min_phonon_freq ON evaluation TYPE option<float>;
DEFINE FIELD IF NOT EXISTS timestamp       ON evaluation TYPE string;
DEFINE INDEX IF NOT EXISTS idx_eval_run    ON evaluation FIELDS run;
DEFINE INDEX IF NOT EXISTS idx_eval_struct ON evaluation FIELDS structure;
"""


def _as_rows(response) -> list[dict]:
    """Normalise a SurrealDB query response to a list of row dicts.

    The Python SDK returns results as a flat ``list[dict]``.
    """
    if isinstance(response, list):
        return [r for r in response if isinstance(r, dict)]
    return []


def _record_id(table: str, key: str) -> str:
    """Build a SurrealDB record-id string, escaping special characters."""
    return f"{table}:⟨{key}⟩"


class SurrealDBStore(StructureStore):
    """Structure storage backend using embedded SurrealDB.

    Tables
    ------
    ``study``
        Phase-diagram study metadata (system, constraints).
    ``run``
        Run-level metadata with optional record link to a study.
    ``structure``
        Per-structure data with lifecycle status tracking and optional
        per-descriptor vector columns for ANN deduplication.
    ``descriptor``
        Registry of descriptor configurations and their vector column names.
    """

    # ------------------------------------------------------------------ #
    #  Construction / connection
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        db_url: str = "mem://",
        *,
        namespace: str = "rapmat",
        database: str = "main",
        username: str | None = None,
        password: str | None = None,
        reclaim_stale_minutes: int | None = 10,
        console: Console = default_console,
    ):
        self._db_url = db_url
        self._console = console
        self._db = Surreal(db_url)
        self._active_vec_col: str | None = None

        if username is not None and password is not None:
            self._db.signin({"username": username, "password": password})
        self._db.use(namespace, database)
        self._define_schema()

        if reclaim_stale_minutes is not None:
            reclaimed_runs = self.reclaim_stale_runs(
                timeout_minutes=reclaim_stale_minutes
            )

            if reclaimed_runs:
                console.print(
                    f"Reclaimed stale runs afrer {reclaim_stale_minutes} heartbeat timeout: {reclaimed_runs}"
                )

    @classmethod
    def from_path(cls, db_path: Path, **kwargs) -> "SurrealDBStore":
        """Create a persistent store backed by a directory on disk."""
        db_path.mkdir(parents=True, exist_ok=True)
        return cls(db_url=f"file://{db_path.as_posix()}", **kwargs)

    def _define_schema(self) -> None:
        self._db.query(_SCHEMA_DDL)

    # ------------------------------------------------------------------ #
    #  Descriptor registration
    # ------------------------------------------------------------------ #

    def register_descriptor(
        self,
        desc_id: str,
        dim: int,
        meta: Optional[dict] = None,
    ) -> str:
        """Register a descriptor configuration and ensure its vector column exists.

        Creates (idempotently) a per-descriptor column on ``structure`` with a
        fixed dimension, then records metadata in the
        ``descriptor`` table.  Sets ``_active_vec_col`` so all subsequent
        vector operations use this column by default.

        Parameters
        ----------
        desc_id
            Full descriptor id string (from ``StructureDescriptor.descriptor_id()``).
        dim
            Vector dimension for this descriptor configuration.
        meta
            Optional extra metadata (e.g. ``{"type": "SOAP", "species": [...]}``)
            stored in the ``descriptor`` table.

        Returns
        -------
        str
            The column name (``vec_{desc_id[:12]}``).
        """
        col = f"vec_{desc_id[:12]}"
        m = meta or {}

        self._db.query(
            f"DEFINE FIELD IF NOT EXISTS {col} ON structure TYPE option<array<float>>;"
        )

        self._db.query(
            f"UPSERT {_record_id('descriptor', desc_id[:12])} CONTENT $data",
            {
                "data": {
                    "col_name": col,
                    "dimension": int(dim),
                    "desc_type": m.get("type", ""),
                    "params_json": json.dumps(
                        {k: v for k, v in m.items() if k != "type"}
                    ),
                    "code_version": m.get("code_version", ""),
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

        self._active_vec_col = col
        return col

    def _vec_col(self, override: str | None = None) -> str:
        """Return the active vector column name, with optional override.

        Raises ``RuntimeError`` if no column has been registered and no
        override is provided.
        """
        col = override or self._active_vec_col
        if not col:
            raise RuntimeError(
                "No descriptor registered. Call register_descriptor() before "
                "performing vector operations."
            )
        return col

    # ------------------------------------------------------------------ #
    #  Run management
    # ------------------------------------------------------------------ #

    def create_run(
        self,
        name: str,
        study_id: str,
        config: Optional[dict] = None,
        worker_id: Optional[str] = None,
    ) -> str:
        existing = self.get_run_metadata(name)
        if existing is not None:
            raise ValueError(
                f"Run '{name}' already exists. Use a different name or resume it."
            )

        study = self.get_study(study_id)
        if study is None:
            raise ValueError(f"Study '{study_id}' not found.")
        
        # Merge batch config with study config
        batch_cfg = config or {}
        run_elements = set(batch_cfg.get("formula", {}).keys())
        study_elements = set(study["system"].split("-"))
        if not run_elements <= study_elements:
            extra = run_elements - study_elements
            raise ValueError(f"Elements {extra} not in study '{study['system']}'.")

        self._db.query(
            f"CREATE {_record_id('run', name)} CONTENT $data",
            {
                "data": {
                    "name": name,
                    "batch_config_json": json.dumps(batch_cfg),
                    "timestamp": datetime.now().isoformat(),
                    "study": RecordID("study", study_id),
                    "run_status": "generating",
                    "worker_id": worker_id,
                    "heartbeat": datetime.now().isoformat() if worker_id else None,
                },
            },
        )
        return name

    def get_run_metadata(self, name: str) -> Optional[dict]:
        rows = _as_rows(self._db.query(f"SELECT * FROM {_record_id('run', name)}"))
        if not rows:
            return None
        row = rows[0]
        study_val = row.get("study")
        study_id = _extract_id(study_val) if study_val else None
        
        batch_cfg = json.loads(row.get("batch_config_json", "{}"))
        study_cfg = {}
        domain = ""
        
        if study_id:
            study = self.get_study(study_id)
            if study:
                study_cfg = study.get("config", {})
                domain = study.get("domain", "")
                system = study.get("system", "")
                calculator = study.get("calculator", "")
                
        # Merge study config with batch config (batch overrides study)
        merged_config = {**study_cfg, **batch_cfg}
        if domain:
            merged_config["domain"] = domain
        if system:
            merged_config["system"] = system
        if calculator:
            merged_config["calculator"] = calculator

        return {
            "name": row["name"],
            "domain": domain,
            "config": merged_config,
            "timestamp": row["timestamp"],
            "study_id": study_id,
            "run_status": row.get("run_status"),
            "worker_id": row.get("worker_id"),
        }

    def update_run_config(self, name: str, config: dict) -> None:
        self._db.query(
            f"UPDATE {_record_id('run', name)} SET batch_config_json = $cfg",
            {"cfg": json.dumps(config)},
        )

    def list_runs(self) -> List[dict]:
        # For listing, we do a basic fetch without full config merges to be fast
        rows = _as_rows(self._db.query("SELECT * FROM run"))
        results = []
        for row in rows:
            study_val = row.get("study")
            study_id = _extract_id(study_val) if study_val else None
            results.append(
                {
                    "name": row["name"],
                    "domain": row.get("domain", ""),
                    "config": json.loads(row.get("batch_config_json", "{}")),
                    "timestamp": row["timestamp"],
                    "study_id": study_id,
                    "run_status": row.get("run_status"),
                    "worker_id": row.get("worker_id"),
                }
            )
        return results

    def count_by_status(self, run_name: str) -> dict[str, int]:
        rows = _as_rows(
            self._db.query(
                "SELECT status, count() AS cnt FROM structure "
                f"WHERE run = {_record_id('run', run_name)} GROUP BY status"
            )
        )
        return {r["status"]: int(r["cnt"]) for r in rows}

    # ------------------------------------------------------------------ #
    #  Run-level locking
    # ------------------------------------------------------------------ #

    def claim_run(self, run_name: str, worker_id: str) -> bool:
        """Atomically claim a run for processing.

        Succeeds only when ``run_status`` is ``"pending"``,
        ``"generating"``, or ``"failed"`` (or ``NONE`` for legacy runs).
        Returns ``True`` if this worker now owns the run.
        """
        rows = _as_rows(
            self._db.query(
                f"UPDATE {_record_id('run', run_name)} SET "
                "run_status = 'processing', worker_id = $wid, "
                "heartbeat = $ts "
                "WHERE run_status IN ['pending', 'generating', 'failed'] "
                "OR run_status IS NONE "
                "RETURN AFTER",
                {
                    "wid": worker_id,
                    "ts": datetime.now().isoformat(),
                },
            )
        )
        return len(rows) > 0

    def release_run(self, run_name: str, final_status: str = "completed") -> None:
        """Release a claimed run, setting its final status."""
        self._db.query(
            f"UPDATE {_record_id('run', run_name)} SET "
            "run_status = $st, worker_id = NONE, heartbeat = NONE",
            {"st": final_status},
        )

    def update_heartbeat(self, run_name: str, worker_id: str) -> None:
        """Refresh the heartbeat timestamp for a claimed run."""
        self._db.query(
            f"UPDATE {_record_id('run', run_name)} SET heartbeat = $ts "
            "WHERE worker_id = $wid",
            {"wid": worker_id, "ts": datetime.now().isoformat()},
        )

    def set_run_status(self, run_name: str, status: str) -> None:
        """Set the run_status field directly (e.g. generating -> processing)."""
        self._db.query(
            f"UPDATE {_record_id('run', run_name)} SET run_status = $st",
            {"st": status},
        )

    def reclaim_stale_runs(self, timeout_minutes: int = 10) -> list[str]:
        """Reset runs whose heartbeat has expired.

        Returns the names of reclaimed runs.
        """
        cutoff = datetime.now().timestamp() - timeout_minutes * 60
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        rows = _as_rows(
            self._db.query(
                "UPDATE run SET run_status = 'pending', "
                "worker_id = NONE, heartbeat = NONE "
                "WHERE run_status IN ['processing', 'generating'] "
                "AND heartbeat IS NOT NONE AND heartbeat < $cutoff "
                "RETURN AFTER",
                {"cutoff": cutoff_iso},
            )
        )
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------ #
    #  Study management
    # ------------------------------------------------------------------ #

    def create_study(
        self,
        study_id: str,
        system: str,
        domain: str,
        calculator: str,
        config: Optional[dict] = None,
    ) -> str:
        existing = self.get_study(study_id)
        if existing is not None:
            raise ValueError(f"Study '{study_id}' already exists.")
        self._db.query(
            f"CREATE {_record_id('study', study_id)} CONTENT $data",
            {
                "data": {
                    "system": system,
                    "domain": domain,
                    "calculator": calculator,
                    "config_json": json.dumps(config or {}),
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )
        return study_id

    def get_study(self, study_id: str) -> Optional[dict]:
        rows = _as_rows(
            self._db.query(f"SELECT * FROM {_record_id('study', study_id)}")
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "study_id": _extract_id(row.get("id", study_id)),
            "system": row["system"],
            "domain": row["domain"],
            "calculator": row["calculator"],
            "config": json.loads(row.get("config_json", "{}")),
            "timestamp": row["timestamp"],
        }

    def update_study(self, study_id: str, fields: dict) -> None:
        """Update fields on a study record."""
        set_exprs = []
        params = {}
        if "config" in fields:
            set_exprs.append("config_json = $cfg")
            params["cfg"] = json.dumps(fields["config"])

        if not set_exprs:
            return

        set_clause = ", ".join(set_exprs)
        self._db.query(
            f"UPDATE {_record_id('study', study_id)} SET {set_clause}",
            params,
        )

    def list_studies(self) -> List[dict]:
        rows = _as_rows(self._db.query("SELECT * FROM study"))
        return [
            {
                "study_id": _extract_id(row.get("id", "")),
                "system": row["system"],
                "domain": row["domain"],
                "calculator": row["calculator"],
                "config": json.loads(row.get("config_json", "{}")),
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_study_runs(self, study_id: str) -> List[dict]:
        rows = _as_rows(
            self._db.query(
                f"SELECT * FROM run WHERE study = {_record_id('study', study_id)}"
            )
        )
        results = []
        for row in rows:
            study_val = row.get("study")
            sid = _extract_id(study_val) if study_val else None
            results.append(
                {
                    "name": row["name"],
                    "domain": row.get("domain", ""),
                    "config": json.loads(row.get("batch_config_json", "{}")),
                    "timestamp": row["timestamp"],
                    "study_id": sid,
                    "run_status": row.get("run_status"),
                    "worker_id": row.get("worker_id"),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    #  Candidate (generated) structures
    # ------------------------------------------------------------------ #

    def add_candidate(
        self,
        atoms: Atoms,
        vector: np.ndarray,
        run_name: str,
        candidate_id: str,
        metadata: Optional[dict] = None,
        vec_col: Optional[str] = None,
    ) -> str:
        col = self._vec_col(vec_col)
        meta = metadata or {}
        self._db.query(
            f"CREATE {_record_id('structure', candidate_id)} CONTENT $data",
            {
                "data": {
                    "run": RecordID("run", run_name),
                    col: vector.astype(np.float32).tolist(),
                    "status": "generated",
                    "formula": atoms.get_chemical_formula(),
                    "energy_per_atom": 0.0,
                    "energy_total": 0.0,
                    "fmax": 0.0,
                    "converged": False,
                    "thickness": meta.get("thickness"),
                    "enthalpy_per_atom": None,
                    "volume": None,
                    "min_phonon_freq": None,
                    "initial_atoms_json": ase_encode(atoms),
                    "final_atoms_json": "",
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )
        return candidate_id

    def add_candidates(
        self,
        candidates: List[Tuple[Atoms, np.ndarray, str, str, Optional[dict]]],
        vec_col: Optional[str] = None,
    ) -> int:
        if not candidates:
            return 0
        for atoms, vector, run_name, candidate_id, metadata in candidates:
            self.add_candidate(
                atoms, vector, run_name, candidate_id, metadata, vec_col=vec_col
            )
        return len(candidates)

    # -- ABC: add_structures -------------------------------------------

    def add_structures(self, run_name: str, structures: List[dict]) -> int:
        if not structures:
            return 0
        ts = datetime.now().isoformat()
        run_ref = RecordID("run", run_name)

        BATCH_SIZE = 500
        for i in range(0, len(structures), BATCH_SIZE):
            batch = structures[i : i + BATCH_SIZE]
            rows = []
            for s in batch:
                atoms: Atoms | None = s.get("atoms")
                vector: np.ndarray | None = s.get("vector")
                meta: dict = s.get("metadata", {})

                row: dict = {
                    "id": RecordID("structure", s["id"]),
                    "run": run_ref,
                    "status": s.get("status", "generating"),
                    "gen_spg": s.get("gen_spg"),
                    "gen_fu": s.get("gen_fu"),
                    "formula": (
                        atoms.get_chemical_formula() if atoms is not None else ""
                    ),
                    "energy_per_atom": float(meta.get("energy_per_atom", 0.0)),
                    "energy_total": float(meta.get("energy_total", 0.0)),
                    "fmax": float(meta.get("fmax", 0.0)),
                    "converged": bool(meta.get("converged", False)),
                    "thickness": (
                        float(meta["thickness"])
                        if meta.get("thickness") is not None
                        else s.get("thickness")
                    ),
                    "enthalpy_per_atom": None,
                    "volume": None,
                    "min_phonon_freq": None,
                    "initial_atoms_json": (
                        ase_encode(atoms) if atoms is not None else ""
                    ),
                    "final_atoms_json": "",
                    "timestamp": ts,
                }

                if vector is not None:
                    col = self._vec_col()
                    row[col] = vector.astype(np.float32).tolist()

                rows.append(row)

            self._db.query("INSERT INTO structure $data", {"data": rows})

        return len(structures)

    def get_unrelaxed_candidates(self, run_name: str) -> List[dict]:
        rows = _as_rows(
            self._db.query(
                "SELECT * FROM structure WHERE "
                f"run = {_record_id('run', run_name)} AND status = 'generated'"
            )
        )
        return [
            {
                "id": _extract_id(row["id"]),
                "atoms": ase_decode(row["initial_atoms_json"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    #  Generation placeholders
    # ------------------------------------------------------------------ #

    def add_generation_placeholders(
        self,
        run_name: str,
        placeholders: List[Tuple[str, int, int]],
    ) -> int:
        """Bulk-insert placeholder records for resumable generation.

        Parameters
        ----------
        run_name : str
            The run these placeholders belong to.
        placeholders : list of (candidate_id, spg, fu)
            Each tuple gives the deterministic record id, space group
            number, and formula-unit count to generate later.

        Returns
        -------
        int
            Number of placeholders inserted.
        """
        if not placeholders:
            return 0
        ts = datetime.now().isoformat()
        run_ref = RecordID("run", run_name)

        BATCH_SIZE = 500
        for i in range(0, len(placeholders), BATCH_SIZE):
            batch = placeholders[i : i + BATCH_SIZE]
            rows = [
                {
                    "id": RecordID("structure", candidate_id),
                    "run": run_ref,
                    "status": "generating",
                    "gen_spg": spg,
                    "gen_fu": fu,
                    "formula": "",
                    "energy_per_atom": 0.0,
                    "energy_total": 0.0,
                    "fmax": 0.0,
                    "converged": False,
                    "thickness": None,
                    "enthalpy_per_atom": None,
                    "volume": None,
                    "min_phonon_freq": None,
                    "initial_atoms_json": "",
                    "final_atoms_json": "",
                    "timestamp": ts,
                }
                for candidate_id, spg, fu in batch
            ]
            self._db.query("INSERT INTO structure $data", {"data": rows})

        return len(placeholders)

    def get_pending_generation(self, run_name: str) -> List[dict]:
        """Fetch all placeholder records still awaiting generation."""
        rows = _as_rows(
            self._db.query(
                "SELECT * FROM structure WHERE "
                f"run = {_record_id('run', run_name)} AND status = 'generating'"
            )
        )
        return [
            {
                "id": _extract_id(row["id"]),
                "gen_spg": row["gen_spg"],
                "gen_fu": row["gen_fu"],
            }
            for row in rows
        ]

    def update_generated_structure(
        self,
        struct_id: str,
        atoms: Atoms,
        vector: Optional[np.ndarray] = None,
        vec_col: Optional[str] = None,
    ) -> None:
        """Promote a placeholder to ``generated`` with actual atoms data."""
        updates: dict = {
            "status": "generated",
            "formula": atoms.get_chemical_formula(),
            "initial_atoms_json": ase_encode(atoms),
            "timestamp": datetime.now().isoformat(),
        }
        if vector is not None:
            col = self._vec_col(vec_col)
            updates[col] = vector.astype(np.float32).tolist()
        self._db.query(
            f"UPDATE {_record_id('structure', struct_id)} MERGE $data",
            {"data": updates},
        )

    def discard_generation_placeholder(self, struct_id: str) -> None:
        """Mark a placeholder as ``discarded`` (incompatible spg, etc.)."""
        self._db.query(
            f"UPDATE {_record_id('structure', struct_id)} MERGE $data",
            {
                "data": {
                    "status": "discarded",
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    # ------------------------------------------------------------------ #
    #  Structure updates
    # ------------------------------------------------------------------ #

    def update_structure(
        self,
        struct_id: str,
        status: str,
        atoms: Optional[Atoms] = None,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        vec_col: Optional[str] = None,
    ) -> None:
        meta = metadata or {}
        updates: dict = {"status": status, "timestamp": datetime.now().isoformat()}

        if atoms is not None:
            updates["final_atoms_json"] = ase_encode(atoms)
            updates["formula"] = atoms.get_chemical_formula()
        if vector is not None:
            col = self._vec_col(vec_col)
            updates[col] = vector.astype(np.float32).tolist()

        updates["energy_per_atom"] = float(meta.get("energy_per_atom", 0.0))
        updates["energy_total"] = float(meta.get("energy_total", 0.0))
        updates["fmax"] = float(meta.get("fmax", 0.0))
        updates["converged"] = bool(meta.get("converged", False))
        updates["thickness"] = (
            float(meta.get("thickness", 0.0))
            if meta.get("thickness") is not None
            else None
        )
        if "enthalpy_per_atom" in meta:
            updates["enthalpy_per_atom"] = float(meta["enthalpy_per_atom"])
        if "volume" in meta:
            updates["volume"] = float(meta["volume"])

        self._db.query(
            f"UPDATE {_record_id('structure', struct_id)} MERGE $data",
            {"data": updates},
        )

    def update_structure_phonon(self, struct_id: str, min_phonon_freq: float) -> None:
        self._db.query(
            f"UPDATE {_record_id('structure', struct_id)} MERGE $data",
            {
                "data": {
                    "min_phonon_freq": float(min_phonon_freq),
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    def clear_run_phonon_results(self, run_name: str) -> None:
        self._db.query(
            "UPDATE structure SET min_phonon_freq = NONE, "
            f"timestamp = $ts WHERE run = {_record_id('run', run_name)}",
            {"ts": datetime.now().isoformat()},
        )

    def mark_duplicates(
        self,
        dropped_ids: list[str],
        kept_ids: list[str],
    ) -> None:
        """Persist dedup analysis results by setting the ``duplicate`` field.

        Parameters
        ----------
        dropped_ids
            Structure ids to mark as ``duplicate = true``.
        kept_ids
            Structure ids to mark as ``duplicate = false``.
        """
        ts = datetime.now().isoformat()
        BATCH = 500
        for i in range(0, len(dropped_ids), BATCH):
            batch = dropped_ids[i : i + BATCH]
            id_list = ", ".join(_record_id("structure", sid) for sid in batch)
            self._db.query(
                f"UPDATE structure SET duplicate = true, timestamp = $ts "
                f"WHERE id IN [{id_list}]",
                {"ts": ts},
            )
        for i in range(0, len(kept_ids), BATCH):
            batch = kept_ids[i : i + BATCH]
            id_list = ", ".join(_record_id("structure", sid) for sid in batch)
            self._db.query(
                f"UPDATE structure SET duplicate = false, timestamp = $ts "
                f"WHERE id IN [{id_list}]",
                {"ts": ts},
            )

    # ------------------------------------------------------------------ #
    #  Vector search / deduplication
    # ------------------------------------------------------------------ #

    def is_duplicate(
        self,
        vector: np.ndarray,
        threshold: float,
        run_id: Optional[str] = None,
        metric: str = "L2",
        vec_col: Optional[str] = None,
    ) -> bool:
        col = self._vec_col(vec_col)
        where = "status = 'relaxed'"
        if run_id is not None:
            where += f" AND run = {_record_id('run', run_id)}"

        rows = _as_rows(
            self._db.query(
                f"SELECT vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {where} AND {col} IS NOT NONE "
                "ORDER BY dist LIMIT 1",
                {"q": vector.astype(np.float32).tolist()},
            )
        )
        if not rows:
            return False
        return float(rows[0]["dist"]) < threshold

    def get_duplicate_min_energy(
        self,
        vector: np.ndarray,
        threshold: float,
        run_id: Optional[str] = None,
        metric: str = "L2",
        vec_col: Optional[str] = None,
    ) -> Optional[float]:
        col = self._vec_col(vec_col)
        where = "status = 'relaxed'"
        if run_id is not None:
            where += f" AND run = {_record_id('run', run_id)}"

        rows = _as_rows(
            self._db.query(
                f"SELECT energy_per_atom, vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {where} AND {col} IS NOT NONE "
                "ORDER BY dist LIMIT 500",
                {"q": vector.astype(np.float32).tolist()},
            )
        )
        if not rows:
            return None

        within = [r for r in rows if float(r["dist"]) < threshold]
        if not within:
            return None

        return min(float(r["energy_per_atom"]) for r in within)

    def get_nearby_relaxed_structures(
        self,
        vector: np.ndarray,
        threshold: float,
        run_id: Optional[str] = None,
        vec_col: Optional[str] = None,
    ) -> List[dict]:
        col = self._vec_col(vec_col)
        where = "status = 'relaxed'"
        if run_id is not None:
            where += f" AND run = {_record_id('run', run_id)}"

        rows = _as_rows(
            self._db.query(
                f"SELECT *, vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {where} AND {col} IS NOT NONE "
                "ORDER BY dist LIMIT 500",
                {"q": vector.astype(np.float32).tolist()},
            )
        )

        results: List[dict] = []
        for row in rows:
            dist = float(row["dist"])
            if dist >= threshold:
                continue
            final_raw = row.get("final_atoms_json", "")
            atoms = ase_decode(final_raw) if final_raw else None
            if atoms is None:
                continue
            results.append(
                {
                    "id": _extract_id(row["id"]),
                    "atoms": atoms,
                    "energy_per_atom": float(row["energy_per_atom"]),
                    "distance": dist,
                }
            )
        return results

    def get_nearby_structures(
        self,
        vector: np.ndarray,
        threshold: float,
        run_id: Optional[str] = None,
        statuses: tuple = ("relaxed",),
        exclude_ids: Optional[List[str]] = None,
        limit: int = 100,
        vec_col: Optional[str] = None,
    ) -> List[dict]:
        col = self._vec_col(vec_col)
        status_list = ", ".join(f"'{s}'" for s in statuses)
        where = f"status IN [{status_list}]"
        if run_id is not None:
            where += f" AND run = {_record_id('run', run_id)}"
        if exclude_ids:
            excluded = ", ".join(_record_id("structure", eid) for eid in exclude_ids)
            where += f" AND id NOT IN [{excluded}]"

        rows = _as_rows(
            self._db.query(
                "SELECT id, status, energy_per_atom, "
                "initial_atoms_json, final_atoms_json, "
                f"vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {where} AND {col} IS NOT NONE "
                f"ORDER BY dist LIMIT {limit}",
                {"q": vector.astype(np.float32).tolist()},
            )
        )

        results: List[dict] = []
        for row in rows:
            dist = float(row["dist"])
            if dist >= threshold:
                break

            status = row["status"]
            if status == "relaxed":
                raw = row.get("final_atoms_json", "")
            else:
                raw = row.get("initial_atoms_json", "")
            atoms = ase_decode(raw) if raw else None
            if atoms is None:
                continue

            forces = atoms.info.get("initial_forces")
            if forces is not None:
                forces = np.asarray(forces)

            results.append(
                {
                    "id": _extract_id(row["id"]),
                    "atoms": atoms,
                    "energy_per_atom": float(row["energy_per_atom"]),
                    "distance": dist,
                    "forces": forces,
                }
            )
        return results

    def find_similar(
        self,
        vector: np.ndarray,
        k: int,
        threshold: Optional[float] = None,
        symprec: float = 1e-3,
        vec_col: Optional[str] = None,
    ) -> List[dict]:
        col = self._vec_col(vec_col)
        if self.count() == 0:
            return []
        rows = _as_rows(
            self._db.query(
                f"SELECT *, vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {col} IS NOT NONE "
                "ORDER BY dist LIMIT $k",
                {"q": vector.astype(np.float32).tolist(), "k": k},
            )
        )
        results: List[dict] = []
        for row in rows:
            dist = float(row["dist"])
            if threshold is not None and dist > threshold:
                continue
            final_raw = row["final_atoms_json"]
            final_atoms = ase_decode(final_raw) if final_raw else None
            results.append(
                {
                    "distance": dist,
                    "formula": row["formula"],
                    "id": _extract_id(row["id"]),
                    "final_spg": format_spg(final_atoms, symprec=symprec),
                    "energy_per_atom": row["energy_per_atom"],
                    "final_atoms_json": final_raw,
                }
            )
        return results

    def add_if_unique(
        self,
        atoms: Atoms,
        vector: np.ndarray,
        metadata: dict,
        threshold: float,
        vec_col: Optional[str] = None,
    ) -> bool:
        col = self._vec_col(vec_col)
        if self.count() > 0:
            rows = _as_rows(
                self._db.query(
                    f"SELECT vector::distance::euclidean({col}, $q) AS dist "
                    f"FROM structure WHERE {col} IS NOT NONE "
                    "ORDER BY dist LIMIT 1",
                    {"q": vector.astype(np.float32).tolist()},
                )
            )
            if rows and float(rows[0]["dist"]) < threshold:
                return False

        record_id_str = metadata.get("id", f"legacy-{self.count()}")
        encoded = ase_encode(atoms)
        self._db.query(
            f"CREATE {_record_id('structure', record_id_str)} CONTENT $data",
            {
                "data": {
                    "run": RecordID("run", metadata.get("run_id", "")),
                    col: vector.astype(np.float32).tolist(),
                    "status": metadata.get("status", "relaxed"),
                    "formula": atoms.get_chemical_formula(),
                    "energy_per_atom": float(metadata.get("energy_per_atom", 0.0)),
                    "energy_total": float(metadata.get("energy_total", 0.0)),
                    "fmax": float(metadata.get("fmax", 0.0)),
                    "converged": bool(metadata.get("converged", False)),
                    "thickness": (
                        float(metadata.get("thickness", 0.0))
                        if metadata.get("thickness") is not None
                        else None
                    ),
                    "enthalpy_per_atom": (
                        float(metadata["enthalpy_per_atom"])
                        if "enthalpy_per_atom" in metadata
                        else None
                    ),
                    "volume": (
                        float(metadata["volume"]) if "volume" in metadata else None
                    ),
                    "min_phonon_freq": None,
                    "initial_atoms_json": encoded,
                    "final_atoms_json": encoded,
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )
        return True

    # -- ABC: find_neighbors --------------------------------------------

    def find_neighbors(
        self,
        vector: np.ndarray,
        threshold: float,
        *,
        k: int = 500,
        run_id: Optional[str] = None,
        statuses: tuple[str, ...] = ("relaxed",),
        exclude_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        col = self._vec_col()
        status_list = ", ".join(f"'{s}'" for s in statuses)
        where = f"status IN [{status_list}]"
        if run_id is not None:
            where += f" AND run = {_record_id('run', run_id)}"
        if exclude_ids:
            excluded = ", ".join(_record_id("structure", eid) for eid in exclude_ids)
            where += f" AND id NOT IN [{excluded}]"

        rows = _as_rows(
            self._db.query(
                "SELECT id, status, energy_per_atom, "
                "initial_atoms_json, final_atoms_json, "
                f"vector::distance::euclidean({col}, $q) AS dist "
                f"FROM structure WHERE {where} AND {col} IS NOT NONE "
                f"ORDER BY dist LIMIT {k}",
                {"q": vector.astype(np.float32).tolist()},
            )
        )

        results: List[dict] = []
        for row in rows:
            dist = float(row["dist"])
            if dist >= threshold:
                break

            status = row["status"]
            if status == "relaxed":
                raw = row.get("final_atoms_json", "")
            else:
                raw = row.get("initial_atoms_json", "")
            atoms = ase_decode(raw) if raw else None
            if atoms is None:
                continue

            forces = atoms.info.get("initial_forces")
            if forces is not None:
                forces = np.asarray(forces)

            results.append(
                {
                    "id": _extract_id(row["id"]),
                    "atoms": atoms,
                    "energy_per_atom": float(row["energy_per_atom"]),
                    "distance": dist,
                    "forces": forces,
                }
            )
        return results

    def get_structures_for_analysis(
        self,
        run_id: str,
        statuses: tuple = ("relaxed",),
    ) -> List[dict]:
        """Load structures for offline analysis (dedup simulation, benchmarking).

        Returns only relational data and decoded ASE Atoms -- no vectors.
        Callers are expected to compute descriptor vectors in-memory using
        whichever descriptor configuration they want to benchmark.
        """
        status_list = ", ".join(f"'{s}'" for s in statuses)
        where = f"run = {_record_id('run', run_id)} " f"AND status IN [{status_list}]"

        rows = _as_rows(
            self._db.query(
                "SELECT id, energy_per_atom, status, "
                "initial_atoms_json, final_atoms_json "
                f"FROM structure WHERE {where}"
            )
        )

        results: List[dict] = []
        for row in rows:
            status = row["status"]
            if status == "relaxed":
                raw = row.get("final_atoms_json", "")
            else:
                raw = row.get("initial_atoms_json", "")
            atoms = ase_decode(raw) if raw else None
            if atoms is None:
                continue

            forces = atoms.info.get("initial_forces")
            if forces is not None:
                forces = np.asarray(forces)

            results.append(
                {
                    "id": _extract_id(row["id"]),
                    "energy_per_atom": float(row["energy_per_atom"]),
                    "atoms": atoms,
                    "forces": forces,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    #  Querying
    # ------------------------------------------------------------------ #

    def get_run_structures(
        self,
        run_name: str,
        status: Optional[str] = None,
        symprec: float = 1e-3,
    ) -> List[dict]:
        where = f"run = {_record_id('run', run_name)}"
        if status is not None:
            where += f" AND status = '{status}'"

        rows = _as_rows(self._db.query(f"SELECT * FROM structure WHERE {where}"))

        results: List[dict] = []
        for row in rows:
            initial_atoms = ase_decode(row["initial_atoms_json"])
            final_raw = row["final_atoms_json"]
            final_atoms = ase_decode(final_raw) if final_raw else None

            results.append(
                {
                    "id": _extract_id(row["id"]),
                    "formula": row["formula"],
                    "initial_spg": format_spg(initial_atoms, symprec=symprec),
                    "final_spg": format_spg(final_atoms, symprec=symprec),
                    "energy_per_atom": row["energy_per_atom"],
                    "energy_total": row["energy_total"],
                    "enthalpy_per_atom": row.get("enthalpy_per_atom"),
                    "volume": row.get("volume"),
                    "fmax": row["fmax"],
                    "converged": row["converged"],
                    "thickness": row.get("thickness"),
                    "min_phonon_freq": row.get("min_phonon_freq"),
                    "duplicate": row.get("duplicate"),
                    "initial_atoms": initial_atoms,
                    "final_atoms": final_atoms,
                    "atoms": final_atoms if final_atoms is not None else initial_atoms,
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                }
            )
        return results

    # -- ABC: get_structures --------------------------------------------

    def get_structures(
        self,
        run_name: str,
        *,
        status: Optional[str] = None,
        statuses: Optional[tuple[str, ...]] = None,
        fields: Optional[list[str]] = None,
        symprec: float = 1e-3,
    ) -> List[dict]:
        effective = statuses or ((status,) if status else None)

        where = f"run = {_record_id('run', run_name)}"
        if effective:
            if len(effective) == 1:
                where += f" AND status = '{effective[0]}'"
            else:
                status_list = ", ".join(f"'{s}'" for s in effective)
                where += f" AND status IN [{status_list}]"

        rows = _as_rows(self._db.query(f"SELECT * FROM structure WHERE {where}"))

        results: List[dict] = []
        for row in rows:
            initial_raw = row.get("initial_atoms_json", "")
            final_raw = row.get("final_atoms_json", "")
            initial_atoms = ase_decode(initial_raw) if initial_raw else None
            final_atoms = ase_decode(final_raw) if final_raw else None

            d: dict = {
                "id": _extract_id(row["id"]),
                "formula": row["formula"],
                "energy_per_atom": row["energy_per_atom"],
                "energy_total": row.get("energy_total", 0.0),
                "enthalpy_per_atom": row.get("enthalpy_per_atom"),
                "volume": row.get("volume"),
                "fmax": row["fmax"],
                "converged": row["converged"],
                "thickness": row.get("thickness"),
                "min_phonon_freq": row.get("min_phonon_freq"),
                "initial_atoms": initial_atoms,
                "final_atoms": final_atoms,
                "atoms": (final_atoms if final_atoms is not None else initial_atoms),
                "status": row["status"],
                "timestamp": row.get("timestamp", ""),
                "gen_spg": row.get("gen_spg"),
                "gen_fu": row.get("gen_fu"),
            }

            if initial_atoms is not None or final_atoms is not None:
                d["initial_spg"] = format_spg(initial_atoms, symprec=symprec)
                d["final_spg"] = format_spg(final_atoms, symprec=symprec)

            results.append(d)
        return results

    def count(self) -> int:
        rows = _as_rows(
            self._db.query("SELECT count() AS cnt FROM structure GROUP ALL")
        )
        if not rows:
            return 0
        return int(rows[0]["cnt"])

    # ------------------------------------------------------------------ #
    #  Evaluation records
    # ------------------------------------------------------------------ #

    def add_evaluation(
        self,
        structure_id: str,
        run_name: str,
        calculator: str,
        config_json: str,
        energy_per_atom: float,
        energy_total: float,
        min_phonon_freq: Optional[float] = None,
    ) -> str:
        """Insert or update an evaluation result for a structure.

        The record id is derived from the structure id, calculator name,
        and config hash so that re-running with the same settings is an
        upsert (supports resume).
        """
        import hashlib

        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
        eval_id = f"{structure_id}_{calculator}_{config_hash}"

        self._db.query(
            f"UPSERT {_record_id('evaluation', eval_id)} CONTENT $data",
            {
                "data": {
                    "structure": RecordID("structure", structure_id),
                    "run": RecordID("run", run_name),
                    "calculator": calculator,
                    "config_json": config_json,
                    "energy_per_atom": float(energy_per_atom),
                    "energy_total": float(energy_total),
                    "min_phonon_freq": (
                        float(min_phonon_freq) if min_phonon_freq is not None else None
                    ),
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )
        return eval_id

    def has_evaluation(
        self, structure_id: str, calculator: str, config_json: str
    ) -> bool:
        """Check whether an evaluation record already exists (for resume)."""
        import hashlib

        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:12]
        eval_id = f"{structure_id}_{calculator}_{config_hash}"
        rows = _as_rows(
            self._db.query(f"SELECT id FROM {_record_id('evaluation', eval_id)}")
        )
        return len(rows) > 0

    def get_evaluations(
        self,
        run_name: str,
        calculator: Optional[str] = None,
    ) -> List[dict]:
        """Return evaluation records for a run, optionally filtered by calculator."""
        where = f"run = {_record_id('run', run_name)}"
        if calculator is not None:
            where += f" AND calculator = '{calculator}'"

        rows = _as_rows(self._db.query(f"SELECT * FROM evaluation WHERE {where}"))
        return [
            {
                "id": _extract_id(row["id"]),
                "structure_id": _extract_id(row["structure"]),
                "calculator": row["calculator"],
                "config_json": row["config_json"],
                "energy_per_atom": row["energy_per_atom"],
                "energy_total": row["energy_total"],
                "min_phonon_freq": row.get("min_phonon_freq"),
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    #  Maintenance
    # ------------------------------------------------------------------ #

    def compact_files_if_needed(self, threshold: int = 100) -> None:
        """No-op — SurrealDB manages storage compaction internally."""

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _extract_id(record_id) -> str:
    """Extract the bare id string from a SurrealDB record-id.

    The SDK returns ``RecordID`` objects with ``.table_name`` and ``.id``
    attributes.  Also handles plain strings like ``"run:⟨al-run⟩"``.
    """
    if record_id is None:
        return ""
    if isinstance(record_id, RecordID):
        return str(record_id.id)
    s = str(record_id)
    if ":" in s:
        _, _, key = s.partition(":")
        return key.strip("⟨⟩")
    return s
