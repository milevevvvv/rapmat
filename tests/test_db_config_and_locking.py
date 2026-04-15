"""Tests for DB config resolution, store factory, and run-level locking."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk
from rapmat.config import DbMode, DbParams
from rapmat.db_config import (
    _DB_CONFIG_FILE,
    clear_db_config,
    load_db_config,
    resolve_store,
    save_db_config,
)
from rapmat.storage import SurrealDBStore

# ------------------------------------------------------------------ #
#  load_db_config / clear_db_config
# ------------------------------------------------------------------ #


class TestDbConfig:
    def test_load_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", tmp_path / "nope.toml")
        monkeypatch.delenv("RAPMAT_DB_URL", raising=False)
        monkeypatch.delenv("RAPMAT_DB_NS", raising=False)
        monkeypatch.delenv("RAPMAT_DB_NAME", raising=False)
        monkeypatch.delenv("RAPMAT_DB_USER", raising=False)
        monkeypatch.delenv("RAPMAT_DB_PASSWORD", raising=False)
        assert load_db_config() is None

    def test_env_vars_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        cfg_file.write_text(
            '[server]\nurl = "ws://old/rpc"\nnamespace = "ns"\n'
            'database = "db"\nusername = "u"\npassword = "p"\n'
        )
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        monkeypatch.setenv("RAPMAT_DB_URL", "ws://new/rpc")
        full = load_db_config()
        assert full is not None
        assert full["server"]["url"] == "ws://new/rpc"
        assert full["server"]["namespace"] == "ns"

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        monkeypatch.setattr("rapmat.db_config.APP_CONFIG_DIR", tmp_path)
        monkeypatch.delenv("RAPMAT_DB_URL", raising=False)
        monkeypatch.delenv("RAPMAT_DB_NS", raising=False)
        monkeypatch.delenv("RAPMAT_DB_NAME", raising=False)
        monkeypatch.delenv("RAPMAT_DB_USER", raising=False)
        monkeypatch.delenv("RAPMAT_DB_PASSWORD", raising=False)

        save_db_config(
            general={"mode": "remote"},
            server={
                "url": "ws://localhost:8000/rpc",
                "namespace": "rapmat",
                "database": "main",
                "username": "root",
                "password": "test",
            },
        )
        full = load_db_config()
        assert full is not None
        assert full["general"]["mode"] == "remote"
        assert full["server"]["url"] == "ws://localhost:8000/rpc"
        assert full["server"]["password"] == "test"

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "db.toml"
        cfg_file.write_text("[server]\n")
        monkeypatch.setattr("rapmat.db_config._DB_CONFIG_FILE", cfg_file)
        assert clear_db_config() is True
        assert not cfg_file.exists()
        assert clear_db_config() is False


# ------------------------------------------------------------------ #
#  Run-level locking
# ------------------------------------------------------------------ #


class TestRunLocking:
    @pytest.fixture
    def store(self, tmp_path):
        s = SurrealDBStore.from_path(tmp_path / "lock_db")
        yield s
        s.close()

    def test_claim_and_release(self, store):
        store.create_study(
            study_id="lock-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="lock-run", worker_id="w1", study_id="lock-run")
        assert store.claim_run("lock-run", "w1")

        meta = store.get_run_metadata("lock-run")
        assert meta["run_status"] == "processing"
        assert meta["worker_id"] == "w1"

        store.release_run("lock-run", "completed")
        meta = store.get_run_metadata("lock-run")
        assert meta["run_status"] == "completed"
        assert meta["worker_id"] is None

    def test_double_claim_fails(self, store):
        store.create_study(
            study_id="dc-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="dc-run", worker_id="w1", study_id="dc-run")
        assert store.claim_run("dc-run", "w1")

        # Second claim should fail (status is now "processing")
        assert store.claim_run("dc-run", "w2") is False

    def test_claim_after_release(self, store):
        store.create_study(
            study_id="re-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="re-run", worker_id="w1", study_id="re-run")
        store.claim_run("re-run", "w1")
        store.release_run("re-run", "pending")

        assert store.claim_run("re-run", "w2")
        meta = store.get_run_metadata("re-run")
        assert meta["worker_id"] == "w2"

    def test_heartbeat_update(self, store):
        store.create_study(
            study_id="hb-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="hb-run", worker_id="w1", study_id="hb-run")
        store.claim_run("hb-run", "w1")
        store.update_heartbeat("hb-run", "w1")

        meta = store.get_run_metadata("hb-run")
        assert meta["run_status"] == "processing"

    def test_reclaim_stale_runs(self, store):
        store.create_study(
            study_id="stale-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="stale-run", worker_id="old-w", study_id="stale-run")
        store.claim_run("stale-run", "old-w")

        # Manually set heartbeat to the past
        past_ts = (datetime.now() - timedelta(minutes=20)).isoformat()
        store._db.query(
            "UPDATE run:⟨stale-run⟩ SET heartbeat = $ts",
            {"ts": past_ts},
        )

        reclaimed = store.reclaim_stale_runs(timeout_minutes=10)
        assert "stale-run" in reclaimed

        meta = store.get_run_metadata("stale-run")
        assert meta["run_status"] == "pending"
        assert meta["worker_id"] is None

    def test_reclaim_ignores_active_runs(self, store):
        store.create_study(
            study_id="active-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="active-run", worker_id="w1", study_id="active-run")
        store.claim_run("active-run", "w1")
        store.update_heartbeat("active-run", "w1")

        reclaimed = store.reclaim_stale_runs(timeout_minutes=10)
        assert "active-run" not in reclaimed

    def test_create_run_sets_initial_status(self, store):
        store.create_study(
            study_id="init-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="init-run", worker_id="w1", study_id="init-run")
        meta = store.get_run_metadata("init-run")
        assert meta["run_status"] == "generating"
        assert meta["worker_id"] == "w1"

    def test_create_run_without_worker(self, store):
        store.create_study(
            study_id="no-w-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="no-w-run", study_id="no-w-run")
        meta = store.get_run_metadata("no-w-run")
        assert meta["run_status"] == "generating"
        assert meta["worker_id"] is None


# ------------------------------------------------------------------ #
#  SurrealDBStore auth (embedded only — server tests need live DB)
# ------------------------------------------------------------------ #


class TestStoreAuth:
    def test_init_without_auth(self, tmp_path):
        store = SurrealDBStore.from_path(tmp_path / "no_auth_db")
        store.create_study(
            study_id="auth-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="auth-run", study_id="auth-run")
        assert store.get_run_metadata("auth-run") is not None
        store.close()

    def test_init_with_none_credentials(self, tmp_path):
        store = SurrealDBStore(
            db_url=f"file://{(tmp_path / 'none_auth_db').as_posix()}",
            username=None,
            password=None,
        )
        store.create_study(
            study_id="none-auth-run",
            system="Test",
            domain="bulk",
            calculator="MATTERSIM",
            config={},
        )
        store.create_run(name="none-auth-run", study_id="none-auth-run")
        assert store.get_run_metadata("none-auth-run") is not None
        store.close()
