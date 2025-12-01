import numpy as np
import json

from killit import ROOT
from killit.protocols.controller import BB84Controller
from killit.utils.setup import setup_experiment


def save_result(root, metrics, verbose=False):
    if verbose:
        print(f"--- Simulation Results ---")
        print(f"Sifted Key Length: {metrics['sifted_key_length']}")
        print(f"Errors Found:      {metrics['error_count']}")
        print(f"QBER:              {metrics['QBER']:.4f}")

        print()  # Newline
        if metrics['QBER'] > 0.11:
            print("SECURITY WARNING: QBER > 11%. Channel may be eavesdropped.")
        else:
            print("Key Exchange Successful.")

    # Save results
    try:
        with open(root / "data/results.json", "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"Error while saving the results: {e}")
    else:
        print(f"Results saved in {root / 'data/results.json'}")

    return


def main():
    root, config = setup_experiment(ROOT / "configs/config.toml", verbose=True)
    ctrl = BB84Controller(config)
    n = ctrl.n
    # Print metadata


    # Generate random data
    alice_bits = np.random.randint(2, size=n)
    alice_bases = np.random.randint(2, size=n)
    bob_bases = np.random.randint(2, size=n)

    # Run Protocol
    ctrl.run_alice(alice_bits, alice_bases)
    ctrl.run_bob(bob_bases)
    
    # Execute
    bob_results_bits = ctrl.execute_circuit()
    # Calculate metrics
    metrics = ctrl.calculate_metrics(
        alice_bits, 
        alice_bases, 
        bob_bases, 
        bob_results_bits
    )
    save_result(root, metrics, verbose=True)


if __name__ == "__main__":
    main()


