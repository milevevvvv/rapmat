from pathlib import Path

from rapmat.calculators import Calculators, get_install_hint, is_calculator_available


def load_calculator(
    calculator_name: Calculators,
    output_dir_path: Path | None = None,
    config: dict | None = None,
):
    """Instantiate an ASE calculator by name.

    *config* is an optional dict of calculator-specific settings
    (e.g. VASP INCAR parameters resolved from a TOML file).
    It is only used by calculators that accept arbitrary keyword config.
    """
    try:
        match calculator_name.value:
            case Calculators.MATTERSIM.value:
                from rapmat.calculators.mattersim import build_calculator_mattersim

                return build_calculator_mattersim()
            case Calculators.NEQUIP_OAML.value:
                from rapmat.calculators.nequip import build_calculator_nequip_oaml

                return build_calculator_nequip_oaml()
            case Calculators.UPET.value:
                from rapmat.calculators.upet import build_calculator_upet

                return build_calculator_upet(config)
            case Calculators.VASP.value:
                from rapmat.calculators.vasp import build_calculator_vasp

                return build_calculator_vasp(config or {}, output_dir_path)
            case _:
                raise NotImplementedError(
                    f"Calculator {calculator_name.value} is not implemented"
                )
    except ImportError as ie:
        hint = get_install_hint(calculator_name)
        installed = [c.value for c in Calculators if is_calculator_available(c)]
        msg = f"Calculator {calculator_name.value} is not installed."
        if hint:
            msg += f"\n  Install with: {hint}"
        if installed:
            msg += f"\n  Currently available: {', '.join(installed)}"
        raise ImportError(msg) from ie
    except RuntimeError as re:
        raise RuntimeError(
            f"An error occurred while attempting to initialize {calculator_name.value} calculator."
        ) from re
