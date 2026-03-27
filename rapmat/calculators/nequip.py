import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path

from rapmat.calculators import CalculatorCallback, ensure_asset


def build_calculator_nequip_oaml(
    callback: CalculatorCallback | None = None,
):
    import torch
    import torch._inductor.codecache
    from nequip.ase import NequIPCalculator
    from rapmat.config import APP_DATA_DIR

    app_dir = APP_DATA_DIR.resolve()
    app_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = app_dir / Path(f"nequipoaml_{device}.nequip.pt2")

    is_windows = sys.platform.startswith("win")

    if is_windows:
        raise RuntimeError(
            "Windows is not supported for nequip due to triton lacking windows support."
        )

    def _install() -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                "nequip-compile",
                "nequip.net:mir-group/NequIP-OAM-L:0.1",
                str(checkpoint),
                "--mode",
                "aotinductor",
                "--device",
                device,
                "--target",
                "ase",
            ],
            capture_output=True,
            check=False,
        )

    ensure_asset(
        name="NequIP-OAM-L checkpoint",
        path=checkpoint,
        install_fn=_install,
        callback=callback,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = NequIPCalculator.from_compiled_model(
            compile_path=str(checkpoint.resolve()),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return calculator
