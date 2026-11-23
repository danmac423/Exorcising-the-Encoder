"""Download reference audio (wav) for every song in the lyrics CSV files made by data_preparation.py.

The logic lives in encoder_exorcist.data.audio_reference.

Usage:
    make audio
    uv run python scripts/download_audio.py --artist "Sabrina Carpenter" --title Espresso
    uv run python scripts/download_audio.py outputs/lyrics_pop_4artists_20songs.csv --output-dir audio_reference
"""

import argparse
from pathlib import Path

from encoder_exorcist.data.audio_reference import default_output_dir, download_all, download_one
from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="*", type=Path, help="lyrics CSV files (default: outputs/lyrics_*.csv)")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--artist", help="download just one song (together with --title), ignoring the CSV files")
    parser.add_argument("--title")
    parser.add_argument("--url", help="with --artist/--title: download this video instead of the first search result")
    args = parser.parse_args()
    if args.artist or args.title:
        if not (args.artist and args.title):
            parser.error("--artist and --title go together")
        print(f"Saved {download_one(args.artist, args.title, args.output_dir, args.url)}")
        return
    csv_files = args.csv_files or sorted((PROJECT_ROOT / "outputs").glob("lyrics_*.csv"))
    if not csv_files:
        parser.error("no lyrics CSV files found, run `make data` first")
    download_all(csv_files, args.output_dir)


if __name__ == "__main__":
    main()
