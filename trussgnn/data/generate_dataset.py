"""Command-line entry point for deterministic TrussGNN dataset generation."""

import argparse
from pathlib import Path

from trussgnn.data import GenerationConfig, generate_dataset, save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--iid-test", type=int, default=200)
    parser.add_argument("--geometry-ood", type=int, default=200)
    parser.add_argument("--topology-size-ood", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "train": args.train,
        "validation": args.validation,
        "iid_test": args.iid_test,
        "geometry_ood": args.geometry_ood,
        "topology_size_ood": args.topology_size_ood,
    }
    bundle = generate_dataset(GenerationConfig(seed=args.seed, split_counts=counts))
    save_dataset(bundle, args.output)
    print(f"Saved {sum(counts.values())} graphs to {args.output}")


if __name__ == "__main__":
    main()
