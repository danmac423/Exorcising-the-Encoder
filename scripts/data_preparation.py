"""Generate the test and held-out lyrics datasets (Last.fm artists, Genius lyrics).

Requires LASTFM_API_KEY and GENIUS_ACCESS_TOKEN in .env. Settings are passed through make, see `make data` in the
Makefile. The logic lives in encoder_exorcist.data.lyrics_dataset.

Usage:
    make data
    make data GENRES="pop rock" N_ARTISTS=10 TEST_SONGS=5 HELDOUT_SONGS=3
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from encoder_exorcist.data.lyrics_dataset import DatasetBuilder, DatasetConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    DatasetBuilder(
        DatasetConfig.from_env(),
        lastfm_key=os.environ["LASTFM_API_KEY"],
        genius_token=os.environ["GENIUS_ACCESS_TOKEN"],
    ).run()


if __name__ == "__main__":
    main()
