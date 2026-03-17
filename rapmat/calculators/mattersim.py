import io
import sys
import warnings


def build_calculator_mattersim():
    import torch

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from mattersim.forcefield import MatterSimCalculator

            calculator = MatterSimCalculator(
                load_path="MatterSim-v1.0.0-5M.pth",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return calculator
