"""Anonymize artist names in the lyrics datasets (MusicBrainz aliases + CMUdict phonetic replacement).

Reads outputs/lyrics_*.csv and writes outputs/anonymized/. Settings are passed through make, see `make anonymize` in
the Makefile. The logic lives in encoder_exorcist.data.anonymization.

Usage:
    make anonymize MUSICBRAINZ_CONTACT=you@example.com
"""

from pathlib import Path

from dotenv import load_dotenv

from encoder_exorcist.data.anonymization import AnonymizationConfig, LyricsAnonymizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    LyricsAnonymizer(AnonymizationConfig.from_env()).run()


if __name__ == "__main__":
    main()
