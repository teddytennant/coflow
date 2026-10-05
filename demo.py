"""Train NeuralUCB on the synthetic velocity field and print mean steps."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from coflow.synthetic import run_synthetic_experiment
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from coflow.synthetic import run_synthetic_experiment


def main():
    result = run_synthetic_experiment()
    print(f"seed {result['seed']} rounds {result['rounds']}")
    print(f"easy mean steps: {result['easy_mean']:.2f}")
    print(f"hard mean steps: {result['hard_mean']:.2f}")


if __name__ == "__main__":
    main()
