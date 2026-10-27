"""Collect lyrics of the most popular artists per genre (Last.fm artists, Genius lyrics).

For every genre the script takes the top artists from Last.fm (a pool twice as large as needed), then walks through
them in order. Two sets are built for the same artists:
  - test set: songs released up to --max-release-year (default 2024)
  - held-out set: songs released after --heldout-after (default 2025-01-31, i.e. from February 2025)
An artist is kept only if Genius has enough valid songs for them in BOTH sets. Artists that are not found or have too
few songs in either set are skipped and the next one from the pool is used, until the requested number is reached.
Artists that released little recently (e.g. older bands) are therefore often skipped; raise --pool-multiplier if a
genre ends up short.

Lyrics are saved to one file per genre and set (outputs/lyrics_<genre>.csv for the test set and
outputs/lyrics_heldout_<genre>.csv for the held-out set); skipped artists go to outputs/skipped_artists.csv.
The script is resumable: artists already saved or skipped are not fetched again.

Requires LASTFM_API_KEY and GENIUS_ACCESS_TOKEN in .env.

Usage:
    uv run python scripts/data_preparation.py
    uv run python scripts/data_preparation.py --genres pop rock --n-artists 10 --test-songs 5 --heldout-songs 3
    uv run python scripts/data_preparation.py --max-release-year 2024 --output-dir outputs --pool-multiplier 3
"""

import argparse
import csv
import os
import random
import re
import time
from datetime import date
from pathlib import Path

import lyricsgenius
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"

# Genre name -> Last.fm tag name (R&B is tagged "rnb" there)
GENRES = {
    "rock": "rock",
    "pop": "pop",
    "hip-hop": "hip-hop",
    "R&B": "rnb",
    "country": "country",
}

EXCLUDED_TITLE_WORDS = [
    "Remix",
    "Cover",
    "Edit",
    "Live",
    "Acapella",
    "Instrumental",
    "Karaoke",
    "A cappella",
]
# Excluded in the held-out set only (re-releases of older songs)
HELDOUT_EXCLUDED_TITLE_WORDS = ["Remaster", "Remastered", "Anniversary"]


def title_regex(words: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)


LYRICS_FIELDS = ["genre", "artist", "title", "release_year", "release_date", "url", "lyrics"]
SKIPPED_FIELDS = ["genre", "artist", "reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genres", nargs="+", default=list(GENRES), choices=list(GENRES), help="Genres to process")
    parser.add_argument("--n-artists", type=int, default=10, help="Artists per genre")
    parser.add_argument("--test-songs", type=int, default=5, help="Test set: songs per artist")
    parser.add_argument("--heldout-songs", type=int, default=5, help="Held-out set: songs per artist")
    parser.add_argument("--max-release-year", type=int, default=2024, help="Test set: songs released up to this year")
    parser.add_argument(
        "--heldout-after",
        type=date.fromisoformat,
        default=date(2025, 1, 31),
        help="Held-out set: songs released after this date, YYYY-MM-DD",
    )
    parser.add_argument("--pool-multiplier", type=int, default=2, help="Last.fm pool size = multiplier x --n-artists")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def top_artists_for_tag(api_key: str, tag: str, limit: int) -> list[str]:
    response = requests.get(
        LASTFM_API_URL,
        params={"method": "tag.gettopartists", "tag": tag, "limit": limit, "api_key": api_key, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Last.fm error {data['error']}: {data['message']}")
    return [artist["name"] for artist in data["topartists"]["artist"]]


def release_year(song: dict) -> int | None:
    return (song.get("release_date_components") or {}).get("year")


def release_date(song: dict) -> str | None:
    """Release date as YYYY[-MM[-DD]], as precise as Genius knows it."""
    parts = (song.get("release_date_components") or {}).values()
    return "-".join(f"{part:02d}" if i else str(part) for i, part in enumerate(parts) if part) or None


def released_after(song: dict, cutoff: date) -> bool:
    """True if the song is known to be released after the cutoff; a missing month or day counts as the earliest."""
    components = song.get("release_date_components") or {}
    if not components.get("year"):
        return False
    released = (components["year"], components.get("month") or 0, components.get("day") or 0)
    return released > (cutoff.year, cutoff.month, cutoff.day)


def build_genius(
    access_token: str, after: date | None, max_year: int | None, extra_excluded_words: list[str] | None = None
) -> lyricsgenius.Genius:
    # Throttled on purpose: pause between requests, time out, retry with backoff
    # Not subclassed: lyricsgenius only sends the token when the class is named "Genius"
    genius = lyricsgenius.Genius(
        access_token,
        sleep_time=1.5,
        timeout=15,
        retries=3,
        remove_section_headers=True,
        skip_non_songs=True,
    )

    excluded_title_re = title_regex(EXCLUDED_TITLE_WORDS + (extra_excluded_words or []))
    is_lyrics = genius._result_is_lyrics
    genius._result_is_lyrics = lambda song: (
        is_lyrics(song)
        and not excluded_title_re.search(song["title"])
        # songs without a known release date can't be confirmed, so they are skipped
        and release_year(song) is not None
        and (after is None or released_after(song, after))
        and (max_year is None or release_year(song) <= max_year)
    )
    return genius


def fetch_artist_songs(genius: lyricsgenius.Genius, artist_name: str, genre: str, n_songs: int) -> list[dict] | None:
    """Returns the artist's most popular valid songs, or None if the artist is not found on Genius."""
    artist = genius.search_artist(
        artist_name,
        max_songs=n_songs,
        sort="popularity",
        get_full_info=False,
        allow_name_change=False,
    )
    if artist is None:
        return None
    return [
        {
            "genre": genre,
            "artist": artist_name,
            "title": song.title,
            "release_year": release_year(song._body),
            "release_date": release_date(song._body),
            "url": song.url,
            "lyrics": song.lyrics,
        }
        for song in artist.songs
    ]


def lyrics_path(output_dir: Path, genre: str, heldout: bool = False) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    return output_dir / f"lyrics_{'heldout_' if heldout else ''}{slug}.csv"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_records(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    lastfm_key = os.environ["LASTFM_API_KEY"]
    token = os.environ["GENIUS_ACCESS_TOKEN"]
    test_genius = build_genius(token, after=None, max_year=args.max_release_year)
    heldout_genius = build_genius(
        token, after=args.heldout_after, max_year=None, extra_excluded_words=HELDOUT_EXCLUDED_TITLE_WORDS
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = args.output_dir / "skipped_artists.csv"
    skipped = load_records(skipped_path)
    pool_size = args.pool_multiplier * args.n_artists  # spare artists in case some lack enough songs on Genius

    for genre in args.genres:
        test_path = lyrics_path(args.output_dir, genre)
        heldout_path = lyrics_path(args.output_dir, genre, heldout=True)
        test_rows, heldout_rows = load_records(test_path), load_records(heldout_path)

        # An artist counts only if saved in both sets; drop leftovers of an interrupted run
        accepted = {row["artist"] for row in test_rows} & {row["artist"] for row in heldout_rows}
        test_rows = [row for row in test_rows if row["artist"] in accepted]
        heldout_rows = [row for row in heldout_rows if row["artist"] in accepted]
        skipped_names = {row["artist"] for row in skipped if row["genre"] == genre}

        for artist_name in top_artists_for_tag(lastfm_key, GENRES[genre], pool_size):
            if len(accepted) >= args.n_artists:
                break
            if artist_name in accepted or artist_name in skipped_names:
                continue

            print(f"{genre}: {artist_name}")
            try:
                reason = None
                artist_test = fetch_artist_songs(test_genius, artist_name, genre, args.test_songs)
                if artist_test is None:
                    reason = "not found on Genius"
                elif len(artist_test) < args.test_songs:
                    reason = f"only {len(artist_test)} valid songs up to {args.max_release_year}"
                else:  # fetch held-out songs only for artists that already qualify for the test set
                    time.sleep(random.uniform(2, 5))
                    artist_heldout = fetch_artist_songs(heldout_genius, artist_name, genre, args.heldout_songs)
                    if len(artist_heldout) < args.heldout_songs:
                        reason = f"only {len(artist_heldout)} valid songs after {args.heldout_after}"
            except Exception as error:  # noqa: BLE001 (network error: not skipped, retried on the next run)
                print(f"  failed: {error}")
                continue

            if reason:
                print(f"  skipped: {reason}")
                skipped.append({"genre": genre, "artist": artist_name, "reason": reason})
                save_records(skipped_path, SKIPPED_FIELDS, skipped)
            else:
                test_rows += artist_test
                heldout_rows += artist_heldout
                accepted.add(artist_name)
                save_records(test_path, LYRICS_FIELDS, test_rows)
                save_records(heldout_path, LYRICS_FIELDS, heldout_rows)
            time.sleep(random.uniform(2, 5))

        print(f"{genre}: {len(accepted)}/{args.n_artists} artists -> {test_path.name}, {heldout_path.name}")
        if len(accepted) < args.n_artists:
            print(f"WARNING: {genre} has only {len(accepted)}/{args.n_artists} artists")


if __name__ == "__main__":
    main()
