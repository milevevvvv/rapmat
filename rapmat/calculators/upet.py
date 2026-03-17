import warnings


def build_calculator_upet(config: dict | None = None):
    import torch

    config = config or {}

    model = config.pop("model", "pet-oam-xl")

    version = config.pop("version", "1.0.0")
    device = config.pop("device", "cuda" if torch.cuda.is_available() else "cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from upet.calculator import UPETCalculator

        calculator = UPETCalculator(
            model=model,
            version=version,
            device=device,
            **config,
        )

    return calculator
