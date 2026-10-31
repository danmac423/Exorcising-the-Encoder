from datetime import date

import pytest

from encoder_exorcist.data.lyrics_dataset import (
    HELDOUT_EXCLUDED_TITLE_WORDS,
    LYRICS_FIELDS,
    DatasetConfig,
    SongFetcher,
    TitleDeduplicator,
    load_records,
    lyrics_path,
    normalize_title,
    release_date,
    released_after,
    released_before,
    save_records,
)

TEST_CUTOFF = date(2025, 1, 1)
HELDOUT_CUTOFF = date(2025, 1, 31)


def genius_song(title: str, year: int | None = 2024, month: int | None = 5, day: int | None = 1) -> dict:
    """A song as returned by the Genius API list endpoint (only the fields the filters read)."""
    components = {"year": year, "month": month, "day": day} if year else None
    return {"title": title, "lyrics_state": "complete", "release_date_components": components}


def make_test_fetcher() -> SongFetcher:
    return SongFetcher("token", 0.9, before=TEST_CUTOFF)


def make_heldout_fetcher() -> SongFetcher:
    return SongFetcher("token", 0.9, after=HELDOUT_CUTOFF, extra_excluded_words=HELDOUT_EXCLUDED_TITLE_WORDS)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Billie Jean (2012 Remaster)", "billie jean"),
        ("Billie Jean - Single Version", "billie jean"),
        ("Déjà Vu [From The Vault]", "deja vu"),
        ("(Sittin' On) The Dock of the Bay", "the dock of the bay"),
        ("(Interlude)", "interlude"),  # nothing left after tag removal, so the full title is kept
    ],
)
def test_normalize_title(title: str, expected: str) -> None:
    assert normalize_title(title) == expected


def test_deduplicator_catches_remasters_and_typos() -> None:
    dedup = TitleDeduplicator(0.9)
    dedup.reset(["Billie Jean"])
    assert not dedup.is_new("Billie Jean (2012 Remaster)")
    assert not dedup.is_new("Billie Jeans")
    assert dedup.is_new("Beat It")
    assert not dedup.is_new("Beat It (Remastered 2012)")  # duplicate of a title added by the previous call


def test_deduplicator_keeps_different_songs() -> None:
    dedup = TitleDeduplicator(0.9)
    assert dedup.is_new("Song Pt. 1")
    assert dedup.is_new("Song Pt. 2")  # different numbers
    assert dedup.is_new("Love")
    assert dedup.is_new("Lover")  # short titles must match exactly


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ({"year": 2025, "month": 1, "day": 31}, False),
        ({"year": 2025, "month": 2, "day": 1}, True),
        ({"year": 2025, "month": None, "day": None}, False),  # unknown month counts as the earliest
        ({"year": 2026, "month": None, "day": None}, True),
        (None, False),
    ],
)
def test_released_after(components: dict | None, expected: bool) -> None:
    assert released_after({"release_date_components": components}, date(2025, 1, 31)) is expected


@pytest.mark.parametrize(
    ("components", "cutoff", "expected"),
    [
        ({"year": 2024, "month": 12, "day": 31}, date(2025, 1, 1), True),
        ({"year": 2025, "month": 1, "day": 1}, date(2025, 1, 1), False),  # the cutoff day itself is excluded
        ({"year": 2024, "month": None, "day": None}, date(2025, 1, 1), True),  # all of 2024 is before 2025
        ({"year": 2024, "month": None, "day": None}, date(2024, 6, 15), False),  # unknown month could be later
        ({"year": 2024, "month": 3, "day": None}, date(2024, 6, 15), True),
        (None, date(2025, 1, 1), False),
    ],
)
def test_released_before(components: dict | None, cutoff: date, expected: bool) -> None:
    assert released_before({"release_date_components": components}, cutoff) is expected


def test_release_date_precision() -> None:
    assert release_date({"release_date_components": {"year": 2025, "month": 3, "day": 28}}) == "2025-03-28"
    assert release_date({"release_date_components": {"year": 2025, "month": None, "day": None}}) == "2025"
    assert release_date({}) is None


def test_lyrics_path_slug(tmp_path) -> None:
    assert lyrics_path(tmp_path, "R&B", 10, 5).name == "lyrics_rnb_10artists_5songs.csv"
    assert lyrics_path(tmp_path, "hip-hop", 3, 2, heldout=True).name == "lyrics_heldout_hip_hop_3artists_2songs.csv"


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENRES", "pop rock")
    monkeypatch.setenv("N_ARTISTS", "3")
    monkeypatch.setenv("HELDOUT_AFTER", "2025-06-30")
    monkeypatch.setenv("TEST_SONGS", "")  # empty means unset
    config = DatasetConfig.from_env()
    assert config.genres == ["pop", "rock"]
    assert config.n_artists == 3
    assert config.heldout_after == date(2025, 6, 30)
    assert config.test_songs == DatasetConfig().test_songs


def test_config_rejects_unknown_genre() -> None:
    with pytest.raises(ValueError, match="Unknown genres"):
        DatasetConfig(genres=["jazz"])


def test_config_from_env_test_before(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DatasetConfig().test_before == TEST_CUTOFF
    monkeypatch.setenv("TEST_BEFORE", "2024-07-01")
    assert DatasetConfig.from_env().test_before == date(2024, 7, 1)


def test_test_set_accepts_only_songs_before_cutoff() -> None:
    fetcher = make_test_fetcher()
    assert fetcher._is_valid_song(genius_song("A", 2024, 12, 31))
    assert not fetcher._is_valid_song(genius_song("B", 2025, 1, 1))
    assert not fetcher._is_valid_song(genius_song("C", 2025, 3, 28))


def test_heldout_set_accepts_only_songs_after_cutoff() -> None:
    fetcher = make_heldout_fetcher()
    assert not fetcher._is_valid_song(genius_song("A", 2025, 1, 31))
    assert fetcher._is_valid_song(genius_song("B", 2025, 2, 1))
    assert fetcher._is_valid_song(genius_song("C", 2026, None, None))
    assert not fetcher._is_valid_song(genius_song("D", 2024, 5, 1))


@pytest.mark.parametrize("fetcher_factory", [make_test_fetcher, make_heldout_fetcher])
def test_songs_without_release_date_are_rejected(fetcher_factory) -> None:
    fetcher = fetcher_factory()
    assert not fetcher._is_valid_song(genius_song("No date", year=None))
    assert not fetcher._is_valid_song({**genius_song("No components"), "release_date_components": None})


@pytest.mark.parametrize(
    "title", ["Song (Live)", "Song - Remix", "Song (Acapella)", "Song [A cappella]", "Karaoke Song"]
)
def test_excluded_title_words_rejected_in_both_sets(title: str) -> None:
    assert not make_test_fetcher()._is_valid_song(genius_song(title, 2024))
    assert not make_heldout_fetcher()._is_valid_song(genius_song(title, 2026))


def test_excluded_words_match_whole_words_only() -> None:
    assert make_test_fetcher()._is_valid_song(genius_song("Alive", 2024))  # contains "Live"
    assert make_test_fetcher()._is_valid_song(genius_song("Credit", 2024))  # contains "Edit"


@pytest.mark.parametrize("title", ["Song (2011 Remaster)", "Song - Remastered 2009", "Song (25th Anniversary Edition)"])
def test_remaster_and_anniversary_excluded_only_in_heldout(title: str) -> None:
    assert make_test_fetcher()._is_valid_song(genius_song(title, 2024))
    assert not make_heldout_fetcher()._is_valid_song(genius_song(title, 2026))


def test_duplicate_titles_are_skipped() -> None:
    fetcher = make_test_fetcher()
    assert fetcher._is_valid_song(genius_song("Billie Jean", 1982))
    assert not fetcher._is_valid_song(genius_song("Billie Jean (2012 Remaster)", 2012))


def test_rejected_song_does_not_block_its_later_version() -> None:
    fetcher = make_test_fetcher()
    assert not fetcher._is_valid_song(genius_song("Same Title", 2025, 6, 1))  # too new, so not remembered
    assert fetcher._is_valid_song(genius_song("Same Title", 2024, 6, 1))


def test_known_titles_block_duplicates_in_heldout() -> None:
    fetcher = make_heldout_fetcher()
    fetcher.deduplicator.reset(["Old Hit"])  # earlier discography
    assert not fetcher._is_valid_song(genius_song("Old Hit", 2026))
    assert fetcher._is_valid_song(genius_song("New Song", 2026))


def test_save_and_load_records_roundtrip(tmp_path) -> None:
    path = tmp_path / "lyrics.csv"
    rows = [
        {
            "genre": "pop",
            "artist": "Beyoncé",
            "title": "Halo, Pt. 2",
            "release_date": "2008-11-17",
            "url": "https://genius.com/x",
            "lyrics": 'Line one\nLine "two", with comma\n\nVerse ąęł',
        }
    ]
    save_records(path, LYRICS_FIELDS, rows)
    assert load_records(path) == rows


def test_load_records_missing_file(tmp_path) -> None:
    assert load_records(tmp_path / "missing.csv") == []


def test_save_records_drops_columns_removed_since_an_older_run(tmp_path) -> None:
    path = tmp_path / "lyrics.csv"
    path.write_text("genre,artist,title,release_year,release_date,url,lyrics\npop,A,T,2019,2019-01-01,u,la\n")
    save_records(path, LYRICS_FIELDS, load_records(path))
    assert path.read_text().splitlines()[0] == ",".join(LYRICS_FIELDS)
    assert load_records(path)[0]["release_date"] == "2019-01-01"
