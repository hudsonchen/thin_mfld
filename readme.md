# Thinned Mean-Field Dynamics

This repository contains the official code for the paper [arXiv:2605.28589](https://arxiv.org/pdf/2605.28589), accepted at **ICML 2026**.

The code studies thinning strategies for particle-based mean-field dynamics in several settings:

- Mean-field games
- Student-teacher neural network
- Lotka-Volterra model
- MMD gradient flow

## Repository Structure

- `run_mfld.py`: main entry point for neural-network, VLM, and MMD-flow experiments.
- `run_mfg.py`: entry point for mean-field game experiments.
- `mfld.py`: core mean-field dynamics simulation classes.
- `utils/`: kernels, datasets, evaluation, problem definitions, and Lotka-Volterra simulators.
- `scripts/`: local and cluster batch scripts.
- `notebooks/` and `plotting_notebooks/`: experiment analysis and plotting notebooks.
- `results*/`: experiment outputs.

## Requirements

The code is written in Python and primarily uses JAX.

Core dependencies:

- `jax`, `jaxlib`
- `numpy`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `diffrax`
- `jaxtyping`
- `goodpoints`

## Setup

### 1. Create and activate an environment

Using conda:

```bash
conda create -n thin_mfld python=3.10 -y
conda activate thin_mfld
```

### 2. Install dependencies

```bash
pip install numpy matplotlib tqdm scikit-learn diffrax jaxtyping
pip install --upgrade "jax[cpu]"
pip install goodpoints
```

If you want GPU support, install JAX/JAXLIB using the official instructions for your CUDA version:

- https://docs.jax.dev/en/latest/installation.html

## Running Experiments

Run all commands from the repository root.

### A. Student-teacher (neural network)

```bash
python run_mfld.py \
	--dataset student_teacher \
	--seed 0 \
	--particle_num 1024 \
	--step_size 0.01 \
	--noise_scale 0.0 \
	--bandwidth 1.0 \
	--step_num 1000 \
	--thinning kt \
	--kernel gaussian \
	--zeta 0.0001 \
	--d 10 \
	--teacher_num 100 \
	--g 1 \
	--kt_function compress_kt
```

### B. Lotka-Volterra model inference (VLM)

```bash
python run_mfld.py \
	--dataset vlm \
	--seed 0 \
	--particle_num 1024 \
	--step_size 0.0001 \
	--noise_scale 0.001 \
	--bandwidth 1.0 \
	--step_num 150 \
	--thinning kt \
	--kernel sobolev \
	--zeta 0.1 \
	--g 1 \
	--kt_function compress_kt
```

### C. MMD flow

```bash
python run_mfld.py \
	--dataset mmd_flow \
	--seed 0 \
	--particle_num 1024 \
	--step_size 1.0 \
	--noise_scale 3e-4 \
	--step_num 15000 \
	--thinning kt \
	--kernel gaussian \
	--zeta 0.0 \
	--g 1 \
	--kt_function compress_kt
```

### D. Mean-field game

```bash
python run_mfg.py \
	--seed 0 \
	--thinning kt \
	--relax 0.05 \
	--bandwidth 1.0 \
	--step_num 100 \
	--particle_num 4096 \
	--kernel gaussian \
	--dt 0.01 \
	--g 1 \
	--kt_function compress_kt
```

## Thinning Options

Most experiments support:

- `--thinning false`: no thinning.
- `--thinning random`: random subset of particles.
- `--thinning kt`: kernel thinning via `goodpoints`.

For KT thinning, supported variants include:

- `--kt_function compress_kt`
- `--kt_function compresspp_kt`

Use `--g` to control KT depth/level, and optionally `--skip_swap`.

## Outputs

Outputs are written into directory trees under `results/`.
Completed runs are renamed with a `__complete` suffix.


## Notes

- Some runner scripts include user-specific path logic (`os.chdir(...)`) for specific machines. If needed, remove or adapt those sections for your environment.
- For reproducibility, keep `--seed` fixed and record `configs` from each run.

## Citation

If you use this code, please cite the corresponding paper:

- https://arxiv.org/pdf/2605.28589

