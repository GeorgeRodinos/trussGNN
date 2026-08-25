"""Run a minimal write check against the configured MLflow destination."""

import argparse
from collections.abc import Sequence

from .tracking import redact_tracking_uri, resolve_tracking_config, run_connection_check


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name", default="phase-4a-connection-check")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    config = resolve_tracking_config(args.tracking_uri, args.experiment_name)
    result = run_connection_check(config, args.run_name)
    print(f"Tracking URI: {redact_tracking_uri(config.tracking_uri)}")
    print(f"Experiment: {result.experiment_name}")
    print(f"Run ID: {result.run_id}")


if __name__ == "__main__":
    main()
