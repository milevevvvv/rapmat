# rapmat

Rapid materials discovery using machine learning interatomic potentials (MLIPs) and random crystal structure generation - all from a terminal UI.

## Features

- **Random crystal structure search** - generate candidate structures with [PyXTal](https://github.com/qzhu2017/PyXtal) and relax them with ML potentials
- **Multiple MLIP backends** - MatterSim, NequIP out of the box, more coming soon
- **Phonon analysis** - evaluate dynamical stability and thermal properties via Phonopy
- **Terminal UI** - manage studies and runs, launch calculations, and browse results without leaving the terminal

## Installation

Linux systems recommended.

Install `pytorch<2.10.0` with CUDA support if you have an NVIDIA GPU; otherwise skip this step:

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

Run its TUI:

```bash
rapmat
```

## Usage

### Basic concepts

A Study defines the system (e.g. Al-O) you are working on and the calculation settings like fmax.
A Run defines a specific `formula x [formula units range]`: e.g. `Al2O3 x 6..8` constituting the unit cell being calculated.

Each run is assigned to its study. One study may have multiple runs, but not vice versa. Runs in one study may overlap, but you can view and perform actions such as deduplication or thickness filtering for only one run at a time. If the endpoint runs (e.g. Al and O) are present (at least one for each element), you can build the convex hull if the compound is binary.
