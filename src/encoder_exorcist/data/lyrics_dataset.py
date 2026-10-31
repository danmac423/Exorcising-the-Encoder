"""Collect lyrics of the most popular artists per genre (artists from Last.fm, lyrics from Genius).

For every genre the top artists are taken from Last.fm (a pool `pool_multiplier` times larger than needed), then
walked through in order. Two sets are built for the same artists:
  - test set: songs released before `test_before`
  - held-out set: songs released after `heldout_after`
An artist is kept only if Genius has enough valid songs for them in BOTH sets. Artists that are not found or have too
few songs in either set are skipped and the next one from the pool is used, until `n_artists` is reached.

Titles are deduplicated with fuzzy matching (difflib), so a song and its remaster/re-release is kept only once:
  - test set: a song is skipped if a similar title is already chosen for that artist
  - held-out set: a song is skipped if a similar title exists anywhere in the artist's earlier discography on Genius
    (released up to `heldout_after`, or with an unknown date) or is already chosen for that artist
Version tags like "(2012 Remaster)", "[From The Vault]" or "- Radio Edit" are ignored when comparing titles.

Lyrics are saved to one file per genre and set, named after the settings, e.g. lyrics_pop_10artists_5songs.csv for the
test set and lyrics_heldout_pop_10artists_3songs.csv for the held-out set. Skipped artists go to
skipped_artists_<test songs>test_<held-out songs>heldout.csv, because whether an artist has enough songs depends on
those numbers. The build is resumable: artists already saved or skipped are not fetched again, so a rerun with the
same settings continues, while different settings start new files.
"""

import csv
import difflib
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lyricsgenius
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

LYRICS_FIELDS = ["genre", "artist", "title", "release_date", "url", "lyrics"]
SKIPPED_FIELDS = ["genre", "artist", "reason"]

VERSION_TAG_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\s-\s.*$")  # "(2012 Remaster)", "[Live]", " - Radio Edit"
SHORT_TITLE_LEN = 5  # titles this short must match exactly, fuzzy matching is too loose for them


@dataclass(frozen=True)
class DatasetConfig:
    genres: list[str] = field(default_factory=lambda: list(GENRES))
    n_artists: int = 10  # artists per genre
    test_songs: int = 5  # songs per artist in the test set
    heldout_songs: int = 5  # songs per artist in the held-out set
    test_before: date = date(2025, 1, 1)  # test set: songs released before this date
    heldout_after: date = date(2025, 1, 31)  # held-out set: songs released after this date
    title_similarity: float = 0.9  # titles at least this similar (0-1, difflib ratio) are the same song
    pool_multiplier: int = 2  # Last.fm pool size = pool_multiplier x n_artists
    output_dir: Path = PROJECT_ROOT / "outputs"

    def __post_init__(self) -> None:
        unknown = set(self.genres) - set(GENRES)
        if unknown:
            raise ValueError(f"Unknown genres {sorted(unknown)}, choose from {list(GENRES)}")

    @classmethod
    def from_env(cls) -> "DatasetConfig":
        """Reads the settings from environment variables (`make data N_ARTISTS=5` sets them); unset ones keep defaults."""
        defaults = cls()
        env = {name: value for name, value in os.environ.items() if value}

        def read(name: str, cast, default):
            return cast(env[name]) if name in env else default

        return cls(
            genres=env["GENRES"].split() if "GENRES" in env else defaults.genres,
            n_artists=read("N_ARTISTS", int, defaults.n_artists),
            test_songs=read("TEST_SONGS", int, defaults.test_songs),
            heldout_songs=read("HELDOUT_SONGS", int, defaults.heldout_songs),
            test_before=read("TEST_BEFORE", date.fromisoformat, defaults.test_before),
            heldout_after=read("HELDOUT_AFTER", date.fromisoformat, defaults.heldout_after),
            title_similarity=read("TITLE_SIMILARITY", float, defaults.title_similarity),
            pool_multiplier=read("POOL_MULTIPLIER", int, defaults.pool_multiplier),
            output_dir=read("OUTPUT_DIR", Path, defaults.output_dir),
        )


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


def title_regex(words: list[str]) -> re.Pattern:
    """Returns a regex that matches any of the words as a whole word, case-insensitive."""
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)


def normalize_title(title: str) -> str:
    """Lowercase title without version tags, accents and punctuation."""

    def clean(text: str) -> str:
        text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
        return re.sub(r"[\W_]+", " ", text.lower()).strip()

    return clean(VERSION_TAG_RE.sub(" ", title)) or clean(title)


def titles_match(first: str, second: str, threshold: float) -> bool:
    """Compares two normalized titles."""
    if first == second:
        return True
    if min(len(first), len(second)) <= SHORT_TITLE_LEN:
        return False
    if re.findall(r"\d+", first) != re.findall(r"\d+", second):  # "Song Pt. 1" vs "Song Pt. 2"
        return False
    return difflib.SequenceMatcher(None, first, second).ratio() >= threshold


class TitleDeduplicator:
    """Remembers the titles of one artist and tells whether a new title is a duplicate of any of them."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.seen: list[str] = []

    def reset(self, known_titles: list[str]) -> None:
        self.seen = [normalize_title(title) for title in known_titles]

    def is_new(self, title: str) -> bool:
        """True (and remembers the title) if no similar title was seen before."""
        normalized = normalize_title(title)
        if any(titles_match(normalized, seen, self.threshold) for seen in self.seen):
            return False
        self.seen.append(normalized)
        return True


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


def released_before(song: dict, cutoff: date) -> bool:
    """True if the song is known to be released before the cutoff; a missing month or day counts as the latest."""
    components = song.get("release_date_components") or {}
    if not components.get("year"):
        return False
    released = (components["year"], components.get("month") or 12, components.get("day") or 31)
    return released < (cutoff.year, cutoff.month, cutoff.day)


class SongFetcher:
    """Genius client that picks one set of songs (e.g. test or held-out) according to the given filters.

    A song is valid if it is a real song (not a tracklist etc.), has a known release date inside the allowed range,
    has no excluded word in the title and is not a duplicate of a title already known for the artist.
    """

    def __init__(
        self,
        access_token: str,
        title_similarity: float,
        after: date | None = None,
        before: date | None = None,
        extra_excluded_words: list[str] | None = None,
    ):
        self.after = after
        self.before = before
        self.excluded_title_re = title_regex(EXCLUDED_TITLE_WORDS + (extra_excluded_words or []))
        self.deduplicator = TitleDeduplicator(title_similarity)

        # Throttled on purpose: pause between requests, time out, retry with backoff
        # Not subclassed: lyricsgenius only sends the token when the class is named "Genius"
        self.genius = lyricsgenius.Genius(
            access_token,
            sleep_time=1.5,
            timeout=15,
            retries=3,
            remove_section_headers=True,
            skip_non_songs=True,
        )
        self._default_is_lyrics = self.genius._result_is_lyrics
        self.genius._result_is_lyrics = self._is_valid_song  # ty: ignore[invalid-assignment]

    def _is_valid_song(self, song: dict) -> bool:
        return bool(
            self._default_is_lyrics(song)
            and not self.excluded_title_re.search(song["title"])
            # songs without a known release date fail both date checks, as they can't be confirmed
            and (self.after is None or released_after(song, self.after))
            and (self.before is None or released_before(song, self.before))
            # last on purpose: a title is remembered only if the song passed every other check
            and self.deduplicator.is_new(song["title"])
        )

    def fetch(
        self,
        artist_name: str,
        genre: str,
        n_songs: int,
        known_titles: list[str] | None = None,
        artist_id: int | None = None,
    ) -> tuple[list[dict], int] | None:
        """Returns the artist's most popular valid songs and the Genius artist id, or None if the artist is not found.

        Songs similar to `known_titles` or to each other are skipped.
        """
        self.deduplicator.reset(known_titles or [])
        artist = self.genius.search_artist(
            artist_name,
            max_songs=n_songs,
            sort="popularity",
            get_full_info=False,
            allow_name_change=False,
            artist_id=artist_id,
        )
        if artist is None:
            return None
        rows = [
            {
                "genre": genre,
                "artist": artist_name,
                "title": song.title,
                "release_date": release_date(song._body),
                "url": song.url,
                "lyrics": song.lyrics,
            }
            for song in artist.songs
        ]
        return rows, artist._body["id"]

    def discography(self, artist_id: int, max_pages: int = 20) -> list[dict]:
        """Returns metadata (no lyrics) of all the artist's songs on Genius, up to max_pages x 50 songs."""
        songs, page = [], 1
        while page and page <= max_pages:
            response = self.genius.artist_songs(artist_id, per_page=50, page=page, sort="popularity")
            songs += response["songs"]
            page = response.get("next_page")
        return songs


def lyrics_path(output_dir: Path, genre: str, n_artists: int, n_songs: int, heldout: bool = False) -> Path:
    """E.g. lyrics_pop_10artists_5songs.csv (lyrics_heldout_... for the held-out set)."""
    slug = re.sub(r"[^a-z0-9]+", "_", GENRES[genre].lower()).strip("_")  # Last.fm tag, e.g. rnb for R&B
    return output_dir / f"lyrics_{'heldout_' if heldout else ''}{slug}_{n_artists}artists_{n_songs}songs.csv"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_records(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )  # drops columns removed since an older run
        writer.writeheader()
        writer.writerows(records)


class DatasetBuilder:
    """Builds the test and held-out lyrics sets for the artists of every configured genre."""

    def __init__(self, config: DatasetConfig, lastfm_key: str, genius_token: str):
        self.config = config
        self.lastfm_key = lastfm_key
        self.test_fetcher = SongFetcher(genius_token, config.title_similarity, before=config.test_before)
        self.heldout_fetcher = SongFetcher(
            genius_token,
            config.title_similarity,
            after=config.heldout_after,
            extra_excluded_words=HELDOUT_EXCLUDED_TITLE_WORDS,
        )
        self.skipped_path = (
            config.output_dir / f"skipped_artists_{config.test_songs}test_{config.heldout_songs}heldout.csv"
        )

    def run(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.skipped = load_records(self.skipped_path)
        for genre in self.config.genres:
            self._build_genre(genre)

    def _build_genre(self, genre: str) -> None:
        config = self.config
        test_path = lyrics_path(config.output_dir, genre, config.n_artists, config.test_songs)
        heldout_path = lyrics_path(config.output_dir, genre, config.n_artists, config.heldout_songs, heldout=True)
        test_rows, heldout_rows = load_records(test_path), load_records(heldout_path)

        # An artist counts only if saved in both sets; drop leftovers of an interrupted run
        accepted = {row["artist"] for row in test_rows} & {row["artist"] for row in heldout_rows}
        test_rows = [row for row in test_rows if row["artist"] in accepted]
        heldout_rows = [row for row in heldout_rows if row["artist"] in accepted]
        skipped_names = {row["artist"] for row in self.skipped if row["genre"] == genre}

        pool_size = config.pool_multiplier * config.n_artists  # spare artists in case some lack enough songs
        for artist_name in top_artists_for_tag(self.lastfm_key, GENRES[genre], pool_size):
            if len(accepted) >= config.n_artists:
                break
            if artist_name in accepted or artist_name in skipped_names:
                continue

            print(f"{genre}: {artist_name}")
            try:
                artist_test, artist_heldout, reason = self._collect_artist(genre, artist_name)
            except Exception as error:  # noqa: BLE001 (network error: not skipped, retried on the next run)
                print(f"  failed: {error}")
                continue

            if reason:
                print(f"  skipped: {reason}")
                self.skipped.append({"genre": genre, "artist": artist_name, "reason": reason})
                save_records(self.skipped_path, SKIPPED_FIELDS, self.skipped)
            else:
                test_rows += artist_test
                heldout_rows += artist_heldout
                accepted.add(artist_name)
                save_records(test_path, LYRICS_FIELDS, test_rows)
                save_records(heldout_path, LYRICS_FIELDS, heldout_rows)
            time.sleep(random.uniform(2, 5))

        print(f"{genre}: {len(accepted)}/{config.n_artists} artists -> {test_path.name}, {heldout_path.name}")
        if len(accepted) < config.n_artists:
            print(f"WARNING: {genre} has only {len(accepted)}/{config.n_artists} artists")

    def _collect_artist(self, genre: str, artist_name: str) -> tuple[list[dict], list[dict], str | None]:
        """Fetches both song sets of an artist. Returns (test rows, held-out rows, reason to skip the artist)."""
        config = self.config
        test_result = self.test_fetcher.fetch(artist_name, genre, config.test_songs)
        if test_result is None:
            return [], [], "not found on Genius"
        test_rows, artist_id = test_result
        if len(test_rows) < config.test_songs:
            return [], [], f"only {len(test_rows)} valid songs before {config.test_before}"

        # Held-out songs are fetched only for artists that already qualify for the test set
        time.sleep(random.uniform(2, 5))
        # They must not repeat anything from the artist's earlier discography
        history = [
            song["title"]
            for song in self.heldout_fetcher.discography(artist_id)
            if not released_after(song, config.heldout_after)
        ]
        heldout_result = self.heldout_fetcher.fetch(artist_name, genre, config.heldout_songs, history, artist_id)
        heldout_rows = heldout_result[0] if heldout_result else []
        if len(heldout_rows) < config.heldout_songs:
            return [], [], f"only {len(heldout_rows)} valid songs after {config.heldout_after}"
        return test_rows, heldout_rows, None
