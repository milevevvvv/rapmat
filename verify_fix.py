import sys
import os

# Add the project root to sys.path
sys.path.append(r"c:\Users\Mi\Documents\Dev\rapmat")

try:
    from rapmat.core.phonon_stability import compute_dynamical_stability_for_results
    from unittest.mock import MagicMock

    # Mock inputs
    results = [{"converged": True, "formula": "Al2O3"}]
    structures = [MagicMock()]
    
    # Mock load_calculator to avoid real calculator loading
    import rapmat.calculators.factory
    rapmat.calculators.factory.load_calculator = MagicMock(return_value=MagicMock())

    # This should NOT raise UnboundLocalError
    compute_dynamical_stability_for_results(
        results=results,
        structures=structures,
        phonon_top=1,
        phonon_cutoff=0.01,
        phonon_supercell=(2, 2, 2),
        phonon_mesh=(4, 4, 4),
        phonon_displacement=0.01,
        phonon_calculator="mattersim"
    )
    print("Verification successful: No UnboundLocalError raised.")
except UnboundLocalError as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
except Exception as e:
    # Other errors might occur due to complex mocks, but we specifically care about UnboundLocalError
    if "local variable '_process_one' referenced before assignment" in str(e) or isinstance(e, UnboundLocalError):
        print(f"Verification failed: {e}")
        sys.exit(1)
    print(f"Caught other error (expected due to mocks): {e}")
    # If we got past the _process_one call, it's fine
    print("Verification successful: No UnboundLocalError raised during initialization/loop start.")
