## Installation

Install `pytorch<2.10.0` with CUDA support if you have an NVIDIA GPU otherwise skip this step:

```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install rapmat:

```bash
# Basic install
pip install rapmat

# MatterSim support
pip install rapmat[mattersim]

# NequIP support
pip install rapmat[nequip]

# uPET support
pip install rapmat[upet]

# All calculator backends at once
pip install rapmat[all-calculators]
```

## Usage

Just run it's TUI:

```bash
rapmat
```
