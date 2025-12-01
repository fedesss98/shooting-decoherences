import argparse
import tomllib
from pathlib import Path

from killit import ROOT


def load_and_override_config(config_path, verbose=False):
    """
    Parses CLI arguments, loads the TOML config, applies overrides,
    and returns a pure dictionary.
    """
    # 1. Define CLI Arguments
    parser = argparse.ArgumentParser(description="BB84 Quantum Protocol Simulation")
    
    # Overrides
    parser.add_argument("-e", "--experiment", type=str, help="Override name of experiment")
    parser.add_argument("-n", "--qubits", type=int, help="Override number of qubits")
    parser.add_argument("-s", "--shots", type=int, help="Override number of shots")
    parser.add_argument("-p", "--prob", type=float, help="Override noise probability (0.0 - 1.0)")
    parser.add_argument("--method", type=str, choices=["automatic", "statevector", "density_matrix"], help="Override simulation method")

    args = parser.parse_args()

    # Load Base TOML Config
    if not config_path.exists():
        raise ValueError(f"Error: Config file '{config_path}' not found.")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # 3. Apply Overrides (Modifying the dict directly)
    # Ensure 'general' section exists
    if 'general' not in config: config['general'] = {}
    
    if args.experiment is not None:
        config['experiment'] = args.experiment
    experiment_root = ROOT / config['experiment']
    
    if args.qubits is not None:
        config['general']['n_qubits'] = args.qubits
    
    if args.shots is not None:
        config['general']['shots'] = args.shots

    if args.method is not None:
        config['general']['sim_method'] = args.method

    # Noise Overrides
    if args.prob is not None:
        if 'noise' not in config: config['noise'] = {}
        config['noise']['depolarizing_prob'] = args.prob
        config['noise']['enabled'] = True

    if verbose:
        print(
            f"Running simulation '{config['experiment']}'\n"
            f"\tAlice qubits: {config['general']['n_qubits']}\n" 
            f"\t(noise: {config['noise']['depolarizing_prob']})"
        )

    return experiment_root, config

def create_experiment_dir(root:Path, verbose=False):
    print(verbose)
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "visualization").mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Created folder {root}!")
    return


def setup_experiment(config_path:str|Path, verbose=False):
    root, config = load_and_override_config(config_path, verbose)
    create_experiment_dir(root, verbose)
    return root, config
