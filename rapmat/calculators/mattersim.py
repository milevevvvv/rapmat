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
            
            # MatterSim's underlying PyTorch model expects float32 tensors but ASE
            # often natively passes float64 numpy arrays, causing crash:
            # "expected scalar type Float but found Double"
            # We wrap the calculate method to enforce float32 casting.
            # Will it work though? Idk
            import numpy as np
            orig_calculate = calculator.calculate
            
            def _safe_calculate(*args, **kwargs):
                atoms = kwargs.get("atoms")
                if atoms is None and len(args) > 0:
                    atoms = args[0]
                if atoms is None:
                    atoms = getattr(calculator, "atoms", None)
                    
                if atoms is not None:
                    atoms.set_positions(atoms.get_positions().astype(np.float32))
                    atoms.set_cell(atoms.get_cell().astype(np.float32))
                    
                return orig_calculate(*args, **kwargs)
                
            calculator.calculate = _safe_calculate
            
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return calculator
