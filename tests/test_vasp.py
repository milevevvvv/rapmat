"""Tests for VASP calculator initialization, config parsing, and CLI integration.

All tests run without invoking VASP -- the ASE ``Vasp()`` constructor only
stores parameters and does not launch any external process.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase.calculators.vasp import Vasp
from rapmat.calculators import Calculators
from rapmat.calculators.factory import load_calculator
from rapmat.calculators.vasp import build_calculator_vasp
from rapmat.config import (
    CalculatorParams,
    _parse_calc_opt_value,
    resolve_calculator_config,
)

# ------------------------------------------------------------------ #
#  1. _parse_calc_opt_value  (value parser unit tests)
# ------------------------------------------------------------------ #


class TestParseCalcOptValue:
    def test_int(self):
        assert _parse_calc_opt_value("500") == 500

    def test_negative_int(self):
        assert _parse_calc_opt_value("-1") == -1

    def test_float_scientific(self):
        result = _parse_calc_opt_value("1e-5")
        assert result == pytest.approx(1e-5)

    def test_float_decimal(self):
        result = _parse_calc_opt_value("0.05")
        assert result == pytest.approx(0.05)

    def test_bool_true(self):
        assert _parse_calc_opt_value("true") is True

    def test_bool_false(self):
        assert _parse_calc_opt_value("false") is False

    def test_json_list(self):
        assert _parse_calc_opt_value("[4,4,4]") == [4, 4, 4]

    def test_json_list_with_spaces(self):
        assert _parse_calc_opt_value("[4, 4, 4]") == [4, 4, 4]

    def test_json_dict(self):
        result = _parse_calc_opt_value('{"L": 2, "U": 4.0}')
        assert result == {"L": 2, "U": 4.0}

    def test_plain_string_fallback(self):
        assert _parse_calc_opt_value("Accurate") == "Accurate"

    def test_plain_string_with_spaces(self):
        assert _parse_calc_opt_value("PBE") == "PBE"

    def test_null_json(self):
        assert _parse_calc_opt_value("null") is None

    def test_empty_string(self):
        assert _parse_calc_opt_value("") == ""


# ------------------------------------------------------------------ #
#  2. resolve_calculator_config  (TOML + CLI merge logic)
# ------------------------------------------------------------------ #


class TestResolveCalculatorConfig:
    def test_empty_no_toml_no_opts(self):
        calc = CalculatorParams()
        assert resolve_calculator_config(calc) == {}

    def test_toml_only(self, tmp_path):
        toml_file = tmp_path / "vasp.toml"
        toml_file.write_text(
            'xc = "PBE"\nencut = 500\nediff = 1e-5\nkpts = [4, 4, 4]\n'
        )
        calc = CalculatorParams(calculator_config=str(toml_file))
        result = resolve_calculator_config(calc)

        assert result["xc"] == "PBE"
        assert result["encut"] == 500
        assert result["ediff"] == pytest.approx(1e-5)
        assert result["kpts"] == [4, 4, 4]

    def test_calc_opt_only(self):
        calc = CalculatorParams(calc_opt=("encut=500", "prec=Accurate"))
        result = resolve_calculator_config(calc)

        assert result["encut"] == 500
        assert result["prec"] == "Accurate"

    def test_calc_opt_float(self):
        calc = CalculatorParams(calc_opt=("sigma=0.05",))
        result = resolve_calculator_config(calc)
        assert result["sigma"] == pytest.approx(0.05)

    def test_calc_opt_bool(self):
        calc = CalculatorParams(calc_opt=("lreal=false",))
        result = resolve_calculator_config(calc)
        assert result["lreal"] is False

    def test_calc_opt_list(self):
        calc = CalculatorParams(calc_opt=("kpts=[6,6,6]",))
        result = resolve_calculator_config(calc)
        assert result["kpts"] == [6, 6, 6]

    def test_override_priority_calc_opt_wins(self, tmp_path):
        toml_file = tmp_path / "base.toml"
        toml_file.write_text('encut = 400\nprec = "Normal"\n')
        calc = CalculatorParams(
            calculator_config=str(toml_file),
            calc_opt=("encut=600",),
        )
        result = resolve_calculator_config(calc)

        assert result["encut"] == 600
        assert result["prec"] == "Normal"

    def test_nested_toml_ldau(self, tmp_path):
        toml_file = tmp_path / "ldau.toml"
        toml_file.write_text(
            "[ldau_luj.Fe]\nL = 2\nU = 4.0\nJ = 0.0\n\n"
            "[ldau_luj.O]\nL = 1\nU = 0.0\nJ = 0.0\n"
        )
        calc = CalculatorParams(calculator_config=str(toml_file))
        result = resolve_calculator_config(calc)

        assert "ldau_luj" in result
        assert result["ldau_luj"]["Fe"] == {"L": 2, "U": 4.0, "J": 0.0}
        assert result["ldau_luj"]["O"] == {"L": 1, "U": 0.0, "J": 0.0}

    def test_complex_toml(self, tmp_path):
        toml_file = tmp_path / "complex.toml"
        toml_file.write_text(
            'xc = "PBE"\n'
            "encut = 500\n"
            "ediff = 1e-5\n"
            'prec = "Accurate"\n'
            "kpts = [4, 4, 4]\n"
            "ismear = 0\n"
            "sigma = 0.05\n"
            'command = "mpirun -np 4 vasp_std"\n'
            "\n"
            "[ldau_luj.Fe]\n"
            "L = 2\n"
            "U = 4.0\n"
            "J = 0.0\n"
        )
        calc = CalculatorParams(
            calculator_config=str(toml_file),
            calc_opt=("encut=600", "kpts=[6,6,6]"),
        )
        result = resolve_calculator_config(calc)

        assert result["xc"] == "PBE"
        assert result["encut"] == 600
        assert result["ediff"] == pytest.approx(1e-5)
        assert result["prec"] == "Accurate"
        assert result["kpts"] == [6, 6, 6]
        assert result["ismear"] == 0
        assert result["sigma"] == pytest.approx(0.05)
        assert result["command"] == "mpirun -np 4 vasp_std"
        assert result["ldau_luj"]["Fe"]["U"] == 4.0

    def test_missing_file_raises(self, tmp_path):
        calc = CalculatorParams(calculator_config=str(tmp_path / "nonexistent.toml"))
        with pytest.raises(ValueError, match="not found"):
            resolve_calculator_config(calc)

    def test_invalid_calc_opt_no_equals(self):
        calc = CalculatorParams(calc_opt=("encut",))
        with pytest.raises(ValueError, match="Invalid calc-opt format"):
            resolve_calculator_config(calc)

    def test_invalid_calc_opt_empty_key(self):
        calc = CalculatorParams(calc_opt=("=500",))
        result = resolve_calculator_config(calc)
        assert result[""] == 500

    def test_calc_opt_with_spaces_around_equals(self):
        calc = CalculatorParams(calc_opt=(" encut = 500 ",))
        result = resolve_calculator_config(calc)
        assert result["encut"] == 500

    def test_toml_with_vasp_command(self, tmp_path):
        toml_file = tmp_path / "cmd.toml"
        toml_file.write_text('command = "mpirun -np 8 vasp_std"\n')
        calc = CalculatorParams(calculator_config=str(toml_file))
        result = resolve_calculator_config(calc)
        assert result["command"] == "mpirun -np 8 vasp_std"


# ------------------------------------------------------------------ #
#  3. build_calculator_vasp + factory routing
# ------------------------------------------------------------------ #


class TestBuildCalculatorVasp:
    def test_empty_config_returns_vasp(self):
        calc = build_calculator_vasp({})
        assert isinstance(calc, Vasp)

    def test_params_forwarded_xc(self):
        calc = build_calculator_vasp({"xc": "PBE"})
        assert isinstance(calc, Vasp)
        assert calc.parameters.get("xc") is not None

    def test_params_forwarded_encut(self):
        calc = build_calculator_vasp({"encut": 500})
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500

    def test_params_forwarded_ediff(self):
        calc = build_calculator_vasp({"ediff": 1e-6})
        assert isinstance(calc, Vasp)
        assert calc.parameters["ediff"] == pytest.approx(1e-6)

    def test_params_forwarded_prec(self):
        calc = build_calculator_vasp({"prec": "Accurate"})
        assert isinstance(calc, Vasp)
        assert calc.parameters["prec"] == "Accurate"

    def test_params_forwarded_ismear(self):
        calc = build_calculator_vasp({"ismear": 0})
        assert isinstance(calc, Vasp)
        assert calc.parameters["ismear"] == 0

    def test_directory_from_argument(self, tmp_path):
        target = tmp_path / "vasp_work"
        calc = build_calculator_vasp({}, directory=target)
        assert isinstance(calc, Vasp)
        assert calc.directory == str(target)

    def test_directory_from_config_takes_precedence(self, tmp_path):
        config_dir = str(tmp_path / "from_config")
        arg_dir = tmp_path / "from_arg"
        calc = build_calculator_vasp({"directory": config_dir}, directory=arg_dir)
        assert calc.directory == config_dir

    def test_directory_not_set_when_none(self):
        calc = build_calculator_vasp({})
        assert isinstance(calc, Vasp)
        # Default ASE directory is "."
        assert calc.directory == "."

    def test_multiple_params_combined(self, tmp_path):
        calc = build_calculator_vasp(
            {
                "xc": "PBE",
                "encut": 500,
                "ediff": 1e-5,
                "prec": "Accurate",
                "ismear": 0,
                "sigma": 0.05,
            },
            directory=tmp_path / "work",
        )
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500
        assert calc.parameters["ediff"] == pytest.approx(1e-5)
        assert calc.parameters["prec"] == "Accurate"
        assert calc.parameters["sigma"] == pytest.approx(0.05)
        assert calc.directory == str(tmp_path / "work")

    def test_config_dict_not_mutated(self):
        config = {"encut": 500}
        build_calculator_vasp(config, directory=Path("/tmp/x"))
        assert "directory" not in config


class TestFactoryVasp:
    def test_factory_routes_to_vasp(self):
        calc = load_calculator(Calculators.VASP, config={"encut": 500})
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 500

    def test_factory_vasp_no_config(self):
        calc = load_calculator(Calculators.VASP)
        assert isinstance(calc, Vasp)

    def test_factory_vasp_with_directory(self, tmp_path):
        calc = load_calculator(
            Calculators.VASP,
            output_dir_path=tmp_path / "out",
            config={"xc": "PBE"},
        )
        assert isinstance(calc, Vasp)
        assert calc.directory == str(tmp_path / "out")

    def test_factory_vasp_config_none_gives_empty(self):
        calc = load_calculator(Calculators.VASP, config=None)
        assert isinstance(calc, Vasp)

    def test_factory_vasp_complex_config(self):
        config = {
            "xc": "PBE",
            "encut": 600,
            "ediff": 1e-6,
            "prec": "Accurate",
            "kpts": [4, 4, 4],
            "ismear": 0,
            "sigma": 0.05,
        }
        calc = load_calculator(Calculators.VASP, config=config)
        assert isinstance(calc, Vasp)
        assert calc.parameters["encut"] == 600
