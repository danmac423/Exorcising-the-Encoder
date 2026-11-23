"""Embed songs with LAION-CLAP or MuQ-MuLan (one L2-normalized vector per song, see encoder_exorcist.embeddings).

Two sources:
  generated  songs of a generation run, outputs/generated/<config>/manifest.csv -> <output-dir>/<model>/generated/<config>/
  reference  original recordings in audio_reference/ (artists and titles from the lyrics CSV files)
                                              -> <output-dir>/<model>/reference/
Each folder gets embeddings.npz (song_ids, embeddings), vectors/<song_id>.npy and failed.csv. Reruns skip finished songs.

Usage (see `make embed` in the Makefile):
    make embed MODEL=clap SOURCE=generated CONFIG=no-cot
    make embed MODEL=mulan SOURCE=reference
"""

import argparse
import logging
import sys
from pathlib import Path

from encoder_exorcist.data.audio_reference import audio_path, read_songs
from encoder_exorcist.data.lyrics_dataset import load_records
from encoder_exorcist.embeddings import EMBEDDERS, MODELS, extract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--source", choices=["generated", "reference"], required=True)
    parser.add_argument("--config", default="no-cot", help="generation run (source generated)")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated")
    parser.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "audio_reference", help="source reference")
    parser.add_argument(
        "--lyrics-dir", type=Path, default=PROJECT_ROOT / "outputs", help="source reference: lyrics_*.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "embeddings")
    parser.add_argument("--limit", type=int, help="only the first N songs, for a quick test")
    return parser.parse_args()


def generated_items(run_dir: Path) -> list[tuple[str, Path]]:
    return [(row["song_id"], run_dir / row["audio_path"]) for row in load_records(run_dir / "manifest.csv")]


def reference_items(audio_dir: Path, lyrics_dir: Path) -> list[tuple[str, Path]]:
    items = []
    for artist, title in read_songs(sorted(lyrics_dir.glob("lyrics_*.csv"))):
        path = audio_path(audio_dir, artist, title)
        if path.exists():
            items.append((path.stem, path))
    return items


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.source == "generated":
        items = generated_items(args.generated_dir / args.config)
        out_dir = args.output_dir / args.model / "generated" / args.config
    else:
        items = reference_items(args.audio_dir, args.lyrics_dir)
        out_dir = args.output_dir / args.model / "reference"
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        sys.exit("no audio files found")

    done, failed = extract(items, out_dir, EMBEDDERS[args.model]())
    print(f"embedded {done}, failed {failed}, saved to {out_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
