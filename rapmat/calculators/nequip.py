import subprocess
import sys
import warnings
from pathlib import Path
from time import sleep


def build_calculator_nequip_oaml():
    import torch
    import torch._inductor.codecache
    from nequip.ase import NequIPCalculator
    from rapmat.config import APP_DATA_DIR
    from rapmat.utils.console import console

    app_dir = APP_DATA_DIR.resolve()
    app_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = app_dir / Path(f"nequipoaml_{device}.nequip.pt2")

    is_windows = sys.platform.startswith("win")

    if is_windows:
        raise RuntimeError(
            "Windows is not supported for nequip due to triton lacking windows support."
        )

    if not checkpoint.exists():
        with console.status(
            f"Nequip-OAM-L checkpoint not found at {checkpoint}, trying to install for {device}..."
        ):
            sleep(3)

            command = [
                "nequip-compile",
                "nequip.net:mir-group/NequIP-OAM-L:0.1",
                str(checkpoint),
                "--mode",
                "aotinductor",
                "--device",
                device,
                "--target",
                "ase",
            ]

            try:
                result = subprocess.run(command, capture_output=True, check=False)
            except Exception as e:
                raise RuntimeError(f"Failed to install the checkpoint: {e}") from e

            if result.returncode == 0:
                if not checkpoint.exists():
                    raise RuntimeError(
                        f"Tried to install, but {checkpoint} still does not exist."
                    )

                console.print("[green]Install ok")
            else:
                raise RuntimeError(
                    f"Failed to install the checkpoint (code {result.returncode}):\n{result.stderr.decode() if result.stderr is not None else ''}"
                )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = NequIPCalculator.from_compiled_model(
            compile_path=str(checkpoint.resolve()),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return calculator
