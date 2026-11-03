"""Generate songs from the anonymized lyrics with ACE-Step and record activations (hooks) for every song.

Reads outputs/anonymized/lyrics_*.csv and writes outputs/generated/<config>/. The logic and the output layout are
described in encoder_exorcist.generation; settings are passed through make, see `make generate` in the Makefile.

Usage:
    make generate                          # no-cot, 1 seed per song
    make generate CONFIG=with-cot SEEDS=2
    make generate-check                    # one song, checks the hooks (docs/hook_points.md, section 7)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from encoder_exorcist.generation import CONFIGS, DURATION, AceStepGenerator, GenerationRun, load_jobs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", type=Path, help="lyrics CSV files (default: outputs/anonymized/lyrics_*.csv)")
    parser.add_argument("--config", choices=list(CONFIGS), default="no-cot")
    parser.add_argument("--seeds", type=int, default=1, help="generations per song")
    parser.add_argument("--duration", type=float, default=DURATION, help="song length in seconds")
    parser.add_argument("--limit", type=int, help="generate only the first N songs (for a quick test)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated")
    parser.add_argument("--check", action="store_true", help="verify the hooks on one song and exit, writes nothing")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    csv_paths = args.csv or sorted((PROJECT_ROOT / "outputs" / "anonymized").glob("lyrics_*.csv"))
    if not csv_paths:
        sys.exit("no lyrics CSV files found, run `make anonymize` or pass the files explicitly")

    jobs = load_jobs(csv_paths, n_seeds=args.seeds)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    generator = AceStepGenerator(args.config, duration=args.duration)

    if args.check:
        print(json.dumps(generator.verify_hooks(jobs[0]), indent=2))
        print("hook check passed")
        return

    generated, failed = GenerationRun(args.output_dir, args.config).run(jobs, generator, duration=args.duration)
    print(f"generated {generated}, failed {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
