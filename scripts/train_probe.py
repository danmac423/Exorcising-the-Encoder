"""Train linear probes for artist identity on the activations of one generation run.

Reads outputs/generated/<config>/ (manifest + activations) and the lyrics in outputs/anonymized/lyrics_*.csv (for the
baselines), writes outputs/probe/<config>/results.csv (one row per hook point) and per_artist.csv. The protocol is
described in encoder_exorcist.probing; settings are passed through make, see `make probe` in the Makefile.

Usage:
    make probe                               # no-cot, permutation test for the stream outputs and lyric layers
    make probe CONFIG=with-cot PERMUTATIONS=0
"""

import argparse
import logging
from pathlib import Path

from encoder_exorcist.data.lyrics_dataset import save_records
from encoder_exorcist.probing import (
    DEFAULT_PERMUTED,
    N_PERMUTATIONS,
    PER_ARTIST_FIELDS,
    RESULT_FIELDS,
    load_probe_data,
    run_probes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="no-cot", help="generation run, a folder in --generated-dir")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "probe")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="0 disables the permutation test")
    parser.add_argument("--permute", nargs="*", default=list(DEFAULT_PERMUTED), help="points to permutation-test")
    parser.add_argument("--points", nargs="*", help="probe only these points (default: all)")
    parser.add_argument("--no-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    lyrics_csvs = sorted((PROJECT_ROOT / "outputs" / "anonymized").glob("lyrics_*.csv"))
    data = load_probe_data(args.generated_dir / args.config, lyrics_csvs)
    print(f"{len(data.song_ids)} songs, {len(set(data.artists))} artists, {len(data.features)} hook points")

    rows, per_artist = run_probes(
        data,
        points=args.points,
        permuted=args.permute,
        n_permutations=args.permutations,
        baselines=not args.no_baselines,
    )
    out_dir = args.output_dir / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    save_records(out_dir / "results.csv", RESULT_FIELDS, rows)
    save_records(out_dir / "per_artist.csv", PER_ARTIST_FIELDS, per_artist)

    best = sorted(rows, key=lambda r: float(r["accuracy"]), reverse=True)[:10]
    print(f"chance = {rows[0]['chance']}; top points:")
    for row in best:
        print(
            f"  {row['point']:28s} acc={row['accuracy']}  95% CI [{row['ci_low']}, {row['ci_high']}]  p={row['p_value']}"
        )
    print(f"saved {out_dir / 'results.csv'} and {out_dir / 'per_artist.csv'}")


if __name__ == "__main__":
    main()
