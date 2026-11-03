import json

import pytest

from encoder_exorcist.data.anonymization import (
    AnonymizationConfig,
    Anonymizer,
    LyricsAnonymizer,
    MusicBrainzClient,
    PhoneticSubstitutor,
    build_terms,
    load_overrides,
    match_case,
    name_key,
    override_key,
    save_records,
    stress_key,
    strip_accents,
)

CORPUS = ["tired", "break", "lincoln", "candy", "brand"] * 3  # words used in lyrics, each often enough to count


@pytest.fixture(scope="module")
def substitutor() -> PhoneticSubstitutor:
    return PhoneticSubstitutor(CORPUS)  # loading CMUdict takes a moment, so it is shared


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict:
        return self.payload


class FakeSession:
    """Answers MusicBrainz requests from a dict path -> payload (or a list of responses, used one after another)."""

    def __init__(self, routes: dict[str, dict | list[FakeResponse]]):
        self.routes = routes
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
        path = url.split("/ws/2/")[1]
        self.calls.append((path, params))
        route = self.routes[path]
        if isinstance(route, list):
            return route.pop(0)
        return FakeResponse(route)


def found_id(client: MusicBrainzClient, name: str) -> str:
    artist = client.find_artist(name)
    assert artist is not None
    return artist["id"]


def make_client(routes: dict) -> tuple[MusicBrainzClient, FakeSession]:
    client = MusicBrainzClient("test@example.com", interval=0)
    session = FakeSession(routes)
    client.session = session  # ty: ignore[invalid-assignment]
    return client, session


class TestNames:
    def test_strip_accents(self):
        assert strip_accents("Beyoncé ADÉLA") == "Beyonce ADELA"

    def test_name_key_ignores_case_accents_and_punctuation(self):
        assert name_key("P!nk") == "p nk"
        assert name_key("  ADÉLA ") == "adela"

    def test_terms_contain_name_and_its_parts(self):
        assert build_terms(["Ariana Grande"]) == ["Ariana Grande", "Ariana", "Grande"]

    def test_terms_include_aliases_and_deduplicate(self):
        terms = build_terms(["Drake", "Drake", "Aubrey Graham"])
        assert terms == ["Drake", "Aubrey Graham", "Aubrey", "Graham"]

    def test_short_parts_and_keep_words_are_not_used_alone(self):
        assert build_terms(["The Weeknd"]) == ["The Weeknd", "Weeknd"]
        assert build_terms(["Lil Nas X"]) == ["Lil Nas X"]

    def test_parts_ignore_punctuation(self):
        assert build_terms(["Tyler, The Creator"]) == ["Tyler, The Creator", "Tyler", "Creator"]

    def test_accented_name_also_gets_an_accent_free_term(self):
        assert build_terms(["Beyoncé"]) == ["Beyoncé", "Beyonce"]

    def test_non_latin_and_too_short_names_are_dropped(self):
        assert build_terms(["宇多田ヒカル", "AB", "Utada"]) == ["Utada"]


class TestMatchCase:
    def test_keeps_the_case_style_of_the_source(self):
        assert match_case("DRAKE", "break") == "BREAK"
        assert match_case("Drake", "break") == "Break"
        assert match_case("drake", "break") == "break"
        assert match_case("X", "break") == "Break"


class TestPhonetics:
    def test_stress_key_counts_syllables_and_finds_primary_stress(self):
        assert stress_key(["T", "AY1", "ER0", "D"]) == (2, 0)
        assert stress_key(["AH0", "B", "OW1", "V"]) == (2, 1)
        assert stress_key(["HH", "M"]) == (0, -1)

    def test_substitute_sounds_alike_but_is_spelled_differently(self, substitutor):
        result = substitutor.substitute("Drake")
        assert result.word == "break"  # differs from "drake" in one phoneme, and is a word used in the lyrics
        assert not result.approximate
        assert result.score == pytest.approx(0.75)

    def test_substitute_keeps_syllables_and_stress(self, substitutor):
        word = "Taylor"
        result = substitutor.substitute(word)
        pronunciations = substitutor.pronunciations
        assert stress_key(pronunciations[result.word]) == stress_key(pronunciations[word.lower()])

    def test_substitute_never_returns_the_word_or_a_piece_of_it(self, substitutor):
        for word in ["Radio", "Lady", "Parton"]:
            result = substitutor.substitute(word)
            assert result.word != word.lower()
            assert result.word not in word.lower()
            assert word.lower() not in result.word

    def test_word_missing_from_cmudict_is_marked_approximate(self, substitutor):
        result = substitutor.substitute("Linkin")
        assert result.approximate
        assert result.word == "lincoln"

    def test_substitute_is_deterministic_and_ignores_case_and_accents(self, substitutor):
        assert substitutor.substitute("Taylor") == substitutor.substitute("TAYLOR")
        assert substitutor.substitute("Beyoncé").word == substitutor.substitute("Beyonce").word


class TestAnonymizer:
    @pytest.fixture
    def anonymizer(self, substitutor) -> Anonymizer:
        terms = {
            "Drake": ["Drake"],
            "Linkin Park": ["Linkin Park", "Park"],
            "The Weeknd": ["The Weeknd"],
            "Future": ["Future"],
            "Nobody": [],
        }
        return Anonymizer(terms, substitutor)

    def test_name_is_replaced_and_counted(self, anonymizer):
        text, replaced = anonymizer.anonymize("Drake", "Drake got the keys\nI said Drake twice")
        assert text == "Break got the keys\nI said Break twice"
        assert replaced == {("drake", "break", "cmudict"): 2}

    def test_phrase_matches_in_any_case_and_across_a_hyphen_or_space(self, anonymizer):
        text, _ = anonymizer.anonymize("Linkin Park", "LINKIN PARK, linkin-park")
        assert "linkin" not in text.lower()
        assert text.count(",") == 1

    def test_longest_term_wins(self, anonymizer):
        text, replaced = anonymizer.anonymize("Linkin Park", "Linkin Park")
        assert len(replaced) == 1
        assert " " in next(iter(replaced))[0]
        assert len(text.split()) == 2

    def test_single_word_matches_only_when_capitalised(self, anonymizer):
        text, replaced = anonymizer.anonymize("Future", "the future is bright, Future knows")
        assert text.startswith("the future is bright, ")
        assert "Future" not in text
        assert sum(replaced.values()) == 1

    def test_phrase_matches_in_any_case(self, anonymizer):
        text, _ = anonymizer.anonymize("The Weeknd", "the weeknd and THE WEEKND")
        assert "weeknd" not in text.lower()

    def test_keep_words_inside_a_name_stay(self, anonymizer):
        text, _ = anonymizer.anonymize("The Weeknd", "Call The Weeknd now")
        assert text.startswith("Call The ")

    def test_only_whole_words_are_replaced(self, anonymizer):
        text, replaced = anonymizer.anonymize("Drake", "Drakeo and Drakes stay")
        assert text == "Drakeo and Drakes stay"
        assert not replaced

    def test_possessive_keeps_the_suffix(self, anonymizer):
        text, _ = anonymizer.anonymize("Drake", "Drake's song")
        assert text == "Break's song"

    def test_artist_without_terms_or_unknown_artist_is_unchanged(self, anonymizer):
        assert anonymizer.anonymize("Nobody", "Drake")[0] == "Drake"
        assert anonymizer.anonymize("Unknown", "Drake")[0] == "Drake"

    def test_other_artists_names_are_left_alone(self, anonymizer):
        assert anonymizer.anonymize("Linkin Park", "Drake")[0] == "Drake"

    def test_replacement_is_consistent(self, anonymizer):
        first, _ = anonymizer.anonymize("Drake", "Drake")
        second, _ = anonymizer.anonymize("Drake", "Drake again")
        assert second.startswith(first)

    def test_approximate_replacements_are_flagged(self, anonymizer):
        _, replaced = anonymizer.anonymize("Linkin Park", "Linkin Park")
        assert [source for _, _, source in replaced] == ["g2p"]


MB_SEARCH = {
    "artists": [
        {"id": "wrong", "name": "Other Drake", "score": 100},
        {"id": "low", "name": "Drake", "score": 40},
        {"id": "abc", "name": "Drake", "score": 100},
    ]
}
MB_ARTIST = {
    "id": "abc",
    "name": "Drake",
    "disambiguation": "Canadian rapper",
    "aliases": [{"name": "Aubrey Drake Graham"}, {"name": "Champagne Papi"}],
}


class TestMusicBrainz:
    def test_sends_contact_in_user_agent(self):
        client = MusicBrainzClient("me@example.com")
        assert "me@example.com" in client.session.headers["User-Agent"]

    def test_find_artist_takes_the_exact_name_with_a_high_score(self):
        client, _ = make_client({"artist": MB_SEARCH})
        assert found_id(client, "drake") == "abc"

    def test_find_artist_matches_by_alias(self):
        search = {"artists": [{"id": "x", "name": "Madonna", "score": 100, "aliases": [{"name": "Louise Ciccone"}]}]}
        client, _ = make_client({"artist": search})
        assert found_id(client, "Louise Ciccone") == "x"

    def test_find_artist_returns_none_without_a_match(self):
        client, _ = make_client({"artist": {"artists": [{"id": "z", "name": "Someone Else", "score": 100}]}})
        assert client.find_artist("Drake") is None

    def test_artist_names_collects_aliases_into_terms(self):
        client, session = make_client({"artist": MB_SEARCH, "artist/abc": MB_ARTIST})
        entry = client.artist_names("Drake")
        assert entry["mbid"] == "abc"
        assert entry["matched_name"] == "Drake"
        assert entry["disambiguation"] == "Canadian rapper"
        assert entry["terms"] == [
            "Drake",
            "Aubrey Drake Graham",
            "Champagne Papi",
            "Aubrey",
            "Graham",
            "Champagne",
            "Papi",
        ]
        assert session.calls[1] == ("artist/abc", {"inc": "aliases", "fmt": "json"})

    def test_artist_without_match_falls_back_to_the_name(self):
        client, _ = make_client({"artist": {"artists": []}})
        entry = client.artist_names("Ariana Grande")
        assert entry["mbid"] is None
        assert entry["terms"] == ["Ariana Grande", "Ariana", "Grande"]

    def test_rate_limited_request_is_retried(self, monkeypatch):
        monkeypatch.setattr("encoder_exorcist.data.anonymization.time.sleep", lambda _: None)
        client, session = make_client({"artist": [FakeResponse({}, 503), FakeResponse(MB_SEARCH)]})
        assert found_id(client, "Drake") == "abc"
        assert len(session.calls) == 2


class TestConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("OUTPUT_DIR", raising=False)
        monkeypatch.delenv("MUSICBRAINZ_CONTACT", raising=False)
        config = AnonymizationConfig.from_env()
        assert config.output_dir.name == "outputs"
        assert config.musicbrainz_contact is None

    def test_reads_environment_and_treats_empty_as_unset(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("MUSICBRAINZ_CONTACT", "")
        config = AnonymizationConfig.from_env()
        assert config.output_dir == tmp_path
        assert config.musicbrainz_contact is None
        assert config.aliases_path == tmp_path / "artist_aliases.json"
        assert config.anonymized_dir == tmp_path / "anonymized"


LYRICS_FIELDS = ["genre", "artist", "title", "release_date", "url", "lyrics"]


class TestLyricsAnonymizer:
    def write_lyrics(self, directory):
        rows = [
            {
                "genre": "rock",
                "artist": "Drake",
                "title": "A",
                "release_date": "",
                "url": "",
                "lyrics": "Drake said hi",
            },
            {"genre": "rock", "artist": "Drake", "title": "B", "release_date": "", "url": "", "lyrics": "no name"},
        ]
        save_records(directory / "lyrics_rock_1artists_2songs.csv", LYRICS_FIELDS, rows)
        save_records(directory / "skipped_artists_2test_0heldout.csv", ["genre", "artist", "reason"], [])

    def test_writes_anonymized_copies_and_a_report(self, tmp_path):
        self.write_lyrics(tmp_path)
        (tmp_path / "artist_aliases.json").write_text(json.dumps({"Drake": {"terms": ["Drake"]}}))
        LyricsAnonymizer(AnonymizationConfig(output_dir=tmp_path)).run()

        anonymized = (tmp_path / "anonymized" / "lyrics_rock_1artists_2songs.csv").read_text()
        assert "Drake said hi" not in anonymized
        assert "no name" in anonymized
        assert not (tmp_path / "anonymized" / "skipped_artists_2test_0heldout.csv").exists()
        report = (tmp_path / "anonymized" / "replacements.csv").read_text()
        assert report.splitlines()[0] == "artist,original,replacement,count,source"
        assert "Drake,drake," in report
        assert report.splitlines()[1].endswith(",cmudict")

    def test_original_files_are_not_modified(self, tmp_path):
        self.write_lyrics(tmp_path)
        (tmp_path / "artist_aliases.json").write_text(json.dumps({"Drake": {"terms": ["Drake"]}}))
        path = tmp_path / "lyrics_rock_1artists_2songs.csv"
        before = path.read_text()
        LyricsAnonymizer(AnonymizationConfig(output_dir=tmp_path)).run()
        assert path.read_text() == before

    def test_missing_aliases_need_a_contact(self, tmp_path):
        self.write_lyrics(tmp_path)
        with pytest.raises(SystemExit, match="MUSICBRAINZ_CONTACT"):
            LyricsAnonymizer(AnonymizationConfig(output_dir=tmp_path)).run()

    def test_no_lyrics_files(self, tmp_path):
        with pytest.raises(SystemExit, match="make data"):
            LyricsAnonymizer(AnonymizationConfig(output_dir=tmp_path)).run()


class TestOverrides:
    @pytest.fixture
    def terms(self) -> dict[str, list[str]]:
        return {"Kanye West": ["Kanye West", "Kanye", "West"]}

    def make(self, substitutor, terms, overrides) -> Anonymizer:
        return Anonymizer(terms, substitutor, overrides)

    def test_override_key_ignores_case_accents_and_punctuation(self):
        assert override_key("Tyler, The  Creator") == "tyler the creator"
        assert override_key("Beyoncé") == "beyonce"

    def test_word_override_beats_the_automatic_choice(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "Kon-yay"})
        text, replaced = anonymizer.anonymize("Kanye West", "Kanye said hi")
        assert text == "Kon-yay said hi"
        assert replaced == {("kanye", "kon-yay", "override"): 1}

    def test_override_takes_the_case_of_the_name_without_changing_its_inner_capitals(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "McYay"})
        assert anonymizer.anonymize("Kanye West", "Kanye")[0] == "McYay"
        assert anonymizer.anonymize("Kanye West", "KANYE")[0] == "MCYAY"

    def test_keys_are_case_and_accent_insensitive(self, substitutor):
        anonymizer = self.make(substitutor, {"Beyoncé": ["Beyoncé", "Beyonce"]}, {"BEYONCÉ": "bee"})
        assert anonymizer.anonymize("Beyoncé", "Beyoncé and Beyonce")[0] == "Bee and Bee"

    def test_whole_name_override_beats_its_words(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "one", "kanye west": "two three"})
        assert anonymizer.anonymize("Kanye West", "Kanye West and Kanye")[0] == "Two three and One"

    def test_word_override_applies_inside_a_phrase_and_the_rest_is_automatic(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "one"})
        text, replaced = anonymizer.anonymize("Kanye West", "kanye west")
        assert text.startswith("one ")
        assert text != "one west"
        assert [source for _, _, source in replaced] == ["override"]

    def test_override_works_for_words_the_dictionary_does_not_know(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "candy"})
        assert anonymizer.anonymize("Kanye West", "Kanye")[0] == "Candy"

    def test_tracks_which_overrides_were_used(self, substitutor, terms):
        anonymizer = self.make(substitutor, terms, {"kanye": "one", "nobody": "two"})
        anonymizer.anonymize("Kanye West", "Kanye")
        assert anonymizer.used_overrides == {"kanye"}

    def test_override_is_not_applied_to_a_word_outside_the_artists_terms(self, substitutor):
        anonymizer = self.make(substitutor, {"Drake": ["Drake"]}, {"hello": "bye"})
        assert anonymizer.anonymize("Drake", "Hello Drake")[0].startswith("Hello ")

    def test_load_overrides(self, tmp_path):
        path = tmp_path / "overrides.json"
        assert load_overrides(path) == {}
        path.write_text('{"kanye": " candy "}')
        assert load_overrides(path) == {"kanye": "candy"}

    @pytest.mark.parametrize("content", ["{not json", '["kanye"]', '{"kanye": 1}', '{"kanye": "  "}'])
    def test_invalid_overrides_file_is_rejected(self, tmp_path, content):
        path = tmp_path / "overrides.json"
        path.write_text(content)
        with pytest.raises(SystemExit, match="overrides.json"):
            load_overrides(path)

    def test_run_uses_the_overrides_file_and_reports_the_source(self, tmp_path, capsys):
        rows = [
            {"genre": "rock", "artist": "Drake", "title": "A", "release_date": "", "url": "", "lyrics": "Drake said hi"}
        ]
        save_records(tmp_path / "lyrics_rock_1artists_1songs.csv", LYRICS_FIELDS, rows)
        (tmp_path / "artist_aliases.json").write_text(json.dumps({"Drake": {"terms": ["Drake"]}}))
        (tmp_path / "replacement_overrides.json").write_text('{"drake": "Mister X", "ghost": "boo"}')
        LyricsAnonymizer(AnonymizationConfig(output_dir=tmp_path)).run()

        anonymized = (tmp_path / "anonymized" / "lyrics_rock_1artists_1songs.csv").read_text()
        assert "Mister X said hi" in anonymized
        assert "Drake,drake,mister x,1,override" in (tmp_path / "anonymized" / "replacements.csv").read_text()
        assert "['ghost']" in capsys.readouterr().out
