#!/usr/bin/env python3
import os
import re

# Root of your results
dataset = 'vlm'
# dataset = 'covertype'
ROOT = f"/home/zongchen/thinned_mfld/results_server/neural_network_{dataset}"

# Where to save the configs
OUTPUT_CONFIG = f"/home/zongchen/thinned_mfld/scripts/bad_configs_{dataset}.txt"


def parse_experiment_dir(dirpath: str) -> str | None:
    """
    Parse a path like:
    /.../neural_network_vlm/thinning_kt/kernel_sobolev__step_size_0.0001__...__seed_6

    into a config line:
    --seed 6 --dataset vlm --g 0 --particle_num 1024 --step_size 0.0001
    --noise_scale 0.001 --bandwidth 1.0 --step_num 200 --thinning kt
    --kernel sobolev --zeta 1.0
    """
    base = os.path.basename(dirpath)

    # Only handle dirs that look like experiment dirs
    if not base.startswith("kernel_"):
        return None
    if base.endswith("__complete"):
        return None

    # Get dataset from 'neural_network_<dataset>' in the path
    dataset = None
    for part in dirpath.split(os.sep):
        m = re.match(r"neural_network_(.+)", part)
        if m:
            dataset = m.group(1)
            break

    # Get thinning from parent 'thinning_<method>'
    parent = os.path.basename(os.path.dirname(dirpath))
    thinning = None
    if parent.startswith("thinning_"):
        thinning = parent.split("_", 1)[1]

    # Parse key/value pairs from the experiment directory name
    # e.g. kernel_sobolev__step_size_0.0001 -> {"kernel": "sobolev", "step_size": "0.0001"}
    params: dict[str, str] = {}
    for chunk in base.split("__"):
        if "_" not in chunk:
            continue
        key, value = chunk.rsplit("_", 1)
        params[key] = value

    required_keys = [
        "seed",
        "g",
        "particle_num",
        "step_size",
        "noise_scale",
        "bandwidth",
        "step_num",
        "kernel",
        "zeta",
    ]
    for k in required_keys:
        if k not in params:
            print(f"[WARN] Missing key '{k}' in directory name: {dirpath}")
            return None

    if dataset is None:
        print(f"[WARN] Could not infer dataset from path: {dirpath}")
        return None
    if thinning is None:
        print(f"[WARN] Could not infer thinning method from path: {dirpath}")
        return None

    # Build the config line in your desired order
    config = (
        f"--seed {params['seed']} "
        f"--dataset {dataset} "
        f"--g {params['g']} "
        f"--particle_num {params['particle_num']} "
        f"--step_size {params['step_size']} "
        f"--noise_scale {params['noise_scale']} "
        f"--bandwidth {params['bandwidth']} "
        f"--step_num {params['step_num']} "
        f"--thinning {thinning} "
        f"--kernel {params['kernel']} "
        f"--zeta {params['zeta']}"
    )
    return config


def main():
    configs = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        base = os.path.basename(dirpath)

        # We only care about experiment dirs that don't end with __complete
        if base.endswith("__complete"):
            continue

        config = parse_experiment_dir(dirpath)
        if config is not None:
            configs.append(config)

    # Deduplicate, just in case
    configs = sorted(set(configs))

    with open(OUTPUT_CONFIG, "w") as f:
        for line in configs:
            f.write(line + "\n")

    print(f"Wrote {len(configs)} configs to {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
