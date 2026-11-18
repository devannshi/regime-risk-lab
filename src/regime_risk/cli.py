"""Command-line entry points for data acquisition and reproducible studies."""

import argparse

from .data import fetch_fx
from .experiments import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Freeze public Federal Reserve FX series")
    download.add_argument("--output", default="data")
    run = sub.add_parser("run", help="Execute the simulation and chronological forecast protocol")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--trials", type=int)
    run.add_argument("--output", default=None)
    run.add_argument("--data", default="data/fx_spot.csv")
    run.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing run outputs"
    )
    args = parser.parse_args()
    if args.command == "download":
        print(fetch_fx(args.output))
    else:
        if args.trials is not None and args.trials < 2:
            parser.error("At least two trials are needed for simulation uncertainty")
        run_study(
            args.output or ("results/quick" if args.quick else "results/full"),
            args.data,
            args.quick,
            args.trials,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
