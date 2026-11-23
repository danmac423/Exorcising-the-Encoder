"""Train linear probes for artist identity on LAION-CLAP / MuQ-MuLan song embeddings ("Ghost in the Encoder", Sec. 3.4).

Reads the embeddings made by `make embed` and trains the same classifier as for the activations (5-fold CV, logistic
regression, permutation test) on all artists and within each genre (paper Table 2). Sources:
  generated  songs of a generation run (metadata from outputs/generated/<config>/manifest.csv)
  reference  original recordings (metadata from the lyrics_*.csv files); the "Originals" column of Table 2
Writes <output-dir>/<source>[/<config>]/results.csv and per_artist.csv (one row per model and scope).

Usage (see `make probe-embeddings` in the Makefile):
    make probe-embeddings SOURCE=generated CONFIG=no-cot
    make probe-embeddings SOURCE=reference PERMUTATIONS=0
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from encoder_exorcist.data.audio_reference import audio_path
from encoder_exorcist.data.lyrics_dataset import load_records, save_records
from encoder_exorcist.embeddings import MODELS
from encoder_exorcist.probing import (
    EMBEDDING_PER_ARTIST_FIELDS,
    EMBEDDING_RESULT_FIELDS,
    N_PERMUTATIONS,
    load_embedding_data,
    run_embedding_probes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["generated", "reference"], required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--config", default="no-cot", help="generation run (source generated)")
    parser.add_argument("--embeddings-dir", type=Path, default=PROJECT_ROOT / "outputs" / "embeddings")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated")
    parser.add_argument("--lyrics-dir", type=Path, default=PROJECT_ROOT / "outputs", help="source reference")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "probe_embeddings")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="0 disables the permutation test")
    return parser.parse_args()


def generated_songs(run_dir: Path) -> dict[str, dict]:
    return {row["song_id"]: row for row in load_records(run_dir / "manifest.csv")}


def reference_songs(lyrics_dir: Path) -> dict[str, dict]:
    """song_id (the wav file name stem, see extract_embeddings.py) -> artist, title, genre from the lyrics files."""
    csv.field_size_limit(10**9)  # lyrics columns are long
    songs: dict[str, dict] = {}
    for path in sorted(lyrics_dir.glob("lyrics_*.csv")):
        for row in load_records(path):
            songs.setdefault(audio_path(Path(), row["artist"], row["title"]).stem, row)
    return songs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    sub = Path("generated") / args.config if args.source == "generated" else Path("reference")
    files = {model: args.embeddings_dir / model / sub / "embeddings.npz" for model in args.models}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        sys.exit(f"missing embeddings (run `make embed` first): {missing}")
    if args.source == "generated":
        songs = generated_songs(args.generated_dir / args.config)
    else:
        songs = reference_songs(args.lyrics_dir)

    data = load_embedding_data(files, songs)
    print(f"{len(data.song_ids)} songs, {len(set(data.artists))} artists, models: {', '.join(files)}")
    rows, per_artist = run_embedding_probes(data, n_permutations=args.permutations)
    if not rows:
        sys.exit("nothing was probed: need songs of at least 2 artists")

    out_dir = args.output_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    save_records(out_dir / "results.csv", EMBEDDING_RESULT_FIELDS, rows)
    save_records(out_dir / "per_artist.csv", EMBEDDING_PER_ARTIST_FIELDS, per_artist)
    for row in rows:
        print(
            f"  {row['scope']:10s} {row['point']:6s} acc={row['accuracy']} chance={row['chance']} "
            f"95% CI [{row['ci_low']}, {row['ci_high']}] p={row['p_value']}"
        )
    print(f"saved {out_dir / 'results.csv'} and {out_dir / 'per_artist.csv'}")


if __name__ == "__main__":
    main()
