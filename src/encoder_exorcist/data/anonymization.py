"""Anonymize artist names in lyrics: MusicBrainz aliases, then phonetic replacement with CMUdict.

Step 1 (aliases): for every artist in the lyrics files the name, its aliases and the parts of multi-word names
(e.g. "Ariana", "Grande") are collected from the MusicBrainz API and cached in artist_aliases.json. The file is meant
to be edited by hand: remove a term that causes false matches, add one that is missing. Artists already in the cache
are not fetched again.

Step 2 (replacement): every occurrence of those terms in the lyrics is replaced by a different word that sounds alike.
The pronunciation comes from CMUdict (ARPAbet phonemes). Candidates must have the same number of syllables and the same
primary stress position as the original, and are ranked by phoneme similarity (difflib). Candidates are taken from
the words used in the lyrics themselves first, so the replacements read like lyrics, and from the whole of CMUdict
only if nothing similar enough is found. Candidates spelled too much like the original are rejected, otherwise the
name would stay recognisable. A name missing from CMUdict gets its pronunciation predicted by the g2p_en model
(CMUdict plus a small grapheme-to-phoneme network), and such replacements are marked `approximate` in
replacements.csv. g2p_en downloads two small NLTK data packages on first use.

You can choose the replacement yourself in replacement_overrides.json, a JSON object next to artist_aliases.json:
{"kanye": "candy", "kanye west": "candy chest"}. A key is a word or a whole name (case and accents do not matter),
the value is used as it is. A whole-name key wins over the keys of its words, and an override wins over the automatic
choice. A key is only applied to names that are among an artist's terms in artist_aliases.json. replacements.csv shows
where each replacement came from (`source`: cmudict, g2p or override), and unused keys are listed after the run.

A multi-word term matches regardless of case. A single-word term matches only when capitalised, so a name like
"Future" or "Post" does not rewrite the ordinary words in a lyric. Only the lyrics column is changed.
"""

import difflib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cmudict
import requests
from g2p_en import G2p

from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT, load_records, save_records

MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2"
MUSICBRAINZ_INTERVAL = 1.1  # seconds between requests, MusicBrainz allows 1 request per second
MIN_SEARCH_SCORE = 90  # MusicBrainz search score (0-100) needed to accept an artist
MIN_TERM_LEN = 3  # shorter names are too ambiguous to be replaced
MIN_TOKEN_LEN = 4  # parts of a multi-word name shorter than this ("Lil", "Jay") are not used on their own
# Words inside a name that are kept as they are ("The Weeknd" -> "The ...")
KEEP_WORDS = {"the", "and", "of", "a", "an", "de", "la", "le", "el", "von", "van", "da", "di", "du", "feat", "ft"}

LYRICS_FILE_GLOB = "lyrics_*.csv"
REPLACEMENT_FIELDS = ["artist", "original", "replacement", "count", "source"]


def strip_accents(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))


def name_key(text: str) -> str:
    """Lowercase letters and digits only, e.g. "ADÉLA!" -> "adela", for comparing names."""
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(text).lower()).strip()


def build_terms(names: Iterable[str]) -> list[str]:
    """Search terms from an artist's names: each name, its accent-free form and the parts of multi-word names."""
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if len(name_key(term)) >= MIN_TERM_LEN and strip_accents(term).isascii() and term not in terms:
            terms.append(term)

    variants = [variant for name in names for variant in (name, strip_accents(name))]
    for variant in variants:
        add(variant)
    for variant in variants:
        parts = re.findall(r"[^\W\d_]+", variant)  # the words, without punctuation such as the comma in "Tyler, ..."
        if len(parts) < 2:
            continue
        for part in parts:
            if len(part) >= MIN_TOKEN_LEN and part.lower() not in KEEP_WORDS:
                add(part)
    return terms


class MusicBrainzClient:
    """Minimal MusicBrainz client. `contact` (an e-mail or URL) goes into the User-Agent, which the API requires."""

    def __init__(self, contact: str, interval: float = MUSICBRAINZ_INTERVAL, retries: int = 3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"EncoderExorcist/0.1 ( {contact} )"
        self.interval = interval
        self.retries = retries
        self._last_request = 0.0

    def _get(self, path: str, **params) -> dict:
        for attempt in range(1, self.retries + 1):
            wait = self.interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            response = self.session.get(f"{MUSICBRAINZ_URL}/{path}", params={**params, "fmt": "json"}, timeout=15)
            if response.status_code in (429, 503) and attempt < self.retries:
                time.sleep(5 * attempt)  # rate limited, back off
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def find_artist(self, name: str) -> dict | None:
        """The first search result whose name or alias equals `name` (ignoring case and accents), if it scores high."""
        query = name.replace('"', " ")
        results = self._get("artist", query=f'artist:"{query}"', limit=5).get("artists", [])
        wanted = name_key(name)
        for result in results:
            names = [result["name"]] + [alias["name"] for alias in result.get("aliases", [])]
            if result.get("score", 0) >= MIN_SEARCH_SCORE and wanted in {name_key(candidate) for candidate in names}:
                return result
        return None

    def artist_names(self, name: str) -> dict:
        """Cache entry for an artist: MusicBrainz match info and the terms to look for in the lyrics."""
        found = self.find_artist(name)
        if found is None:
            return {"mbid": None, "matched_name": None, "disambiguation": None, "terms": build_terms([name])}
        artist = self._get(f"artist/{found['id']}", inc="aliases")
        names = [name, artist["name"]] + [alias["name"] for alias in artist.get("aliases", [])]
        return {
            "mbid": artist["id"],
            "matched_name": artist["name"],
            "disambiguation": artist.get("disambiguation"),
            "terms": build_terms(names),
        }


def phoneme_core(phones: list[str]) -> list[str]:
    """Phonemes without the stress digit (AH0 -> AH)."""
    return [phone.rstrip("012") for phone in phones]


def stress_key(phones: list[str]) -> tuple[int, int]:
    """(number of syllables, index of the syllable carrying the primary stress or -1)."""
    vowels = [phone for phone in phones if phone[-1].isdigit()]
    primary = next((index for index, vowel in enumerate(vowels) if vowel.endswith("1")), -1)
    return len(vowels), primary


def spelling_similarity(first: str, second: str) -> float:
    return difflib.SequenceMatcher(None, first, second).ratio()


def override_key(text: str) -> str:
    """Words of a name in lowercase without accents or punctuation, e.g. "Tyler, The Creator" -> "tyler the creator"."""
    return " ".join(re.findall(r"[^\W\d_]+", strip_accents(text).lower()))


def apply_case(source: str, text: str) -> str:
    """Gives an override the case style of the name it replaces, without touching its inner capitals."""
    if len(source) > 1 and source.isupper():
        return text.upper()
    return text[:1].upper() + text[1:] if source[:1].isupper() else text


def match_case(source: str, word: str) -> str:
    if len(source) > 1 and source.isupper():
        return word.upper()
    return word.capitalize() if source[:1].isupper() else word


@dataclass(frozen=True)
class Substitution:
    word: str
    score: float  # phoneme similarity to the original (0-1)
    approximate: bool  # the original is not in CMUdict, so its pronunciation was predicted


class PhoneticSubstitutor:
    """Finds, for a word, a differently spelled word that sounds alike."""

    def __init__(
        self,
        corpus_words: Iterable[str] = (),
        min_score: float = 0.7,
        max_spelling_similarity: float = 0.75,
        min_corpus_count: int = 3,
    ):
        self.min_score = min_score
        self.max_spelling_similarity = max_spelling_similarity
        self.pronunciations = {
            word: phones[0] for word, phones in cmudict.dict().items() if word.isalpha() and word.isascii()
        }
        counts = Counter(corpus_words)
        corpus = sorted(
            word
            for word, count in counts.items()
            if count >= min_corpus_count and len(word) >= MIN_TERM_LEN and word in self.pronunciations
        )
        self._corpus = self._index(corpus)
        self._g2p: G2p | None = None  # loaded on first need
        self._full: dict[tuple[int, int], list[str]] | None = None  # built on first need, it is large
        self._cache: dict[str, Substitution] = {}

    def _index(self, words: Iterable[str]) -> dict[tuple[int, int], list[str]]:
        index: dict[tuple[int, int], list[str]] = {}
        for word in words:
            index.setdefault(stress_key(self.pronunciations[word]), []).append(word)
        return index

    def _closest(
        self, word: str, phones: list[str], index: dict[tuple[int, int], list[str]], approximate: bool
    ) -> Substitution | None:
        target = phoneme_core(phones)
        matcher = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(target)
        best: Substitution | None = None
        for candidate in index.get(stress_key(phones), []):  # sorted, so ties resolve the same way every run
            if (
                candidate in word
                or word in candidate
                or spelling_similarity(word, candidate) >= self.max_spelling_similarity
            ):
                continue  # also rejects a part of the name or a word containing it, e.g. "radio" for "radiohead"
            matcher.set_seq1(phoneme_core(self.pronunciations[candidate]))
            if best is not None and matcher.real_quick_ratio() <= best.score:
                continue
            score = matcher.ratio()
            if best is None or score > best.score:
                best = Substitution(candidate, score, approximate)
        return best

    def substitute(self, word: str) -> Substitution:
        key = strip_accents(word).lower()
        if key not in self._cache:
            self._cache[key] = self._find(key)
        return self._cache[key]

    def _pronounce(self, word: str) -> list[str]:
        """CMUdict pronunciation, or one predicted by g2p_en for words CMUdict does not have."""
        if word in self.pronunciations:
            return self.pronunciations[word]
        if self._g2p is None:
            self._g2p = G2p()
        return [phone for phone in self._g2p(word) if phone[0].isupper()]  # drops spaces and punctuation

    def _find(self, word: str) -> Substitution:
        approximate = word not in self.pronunciations
        phones = self._pronounce(word)
        best = self._closest(word, phones, self._corpus, approximate)
        if best is None or best.score < self.min_score:
            if self._full is None:
                self._full = self._index(sorted(word for word in self.pronunciations if len(word) >= MIN_TERM_LEN))
            wider = self._closest(word, phones, self._full, approximate)
            if wider is not None and (best is None or wider.score > best.score):
                best = wider
        return best or Substitution(word, 0.0, approximate)  # nothing fits: the word stays, replacements.csv shows it


class Anonymizer:
    """Replaces the artists' names in lyrics.

    `terms` maps an artist to the terms to look for (see build_terms). `overrides` maps a word or a whole name to the
    text that replaces it instead of the automatic choice.
    """

    def __init__(
        self,
        terms: dict[str, list[str]],
        substitutor: PhoneticSubstitutor,
        overrides: dict[str, str] | None = None,
    ):
        self.terms = terms
        self.substitutor = substitutor
        self.overrides = {override_key(key): value for key, value in (overrides or {}).items() if override_key(key)}
        self.used_overrides: set[str] = set()
        self._patterns: dict[str, re.Pattern | None] = {}

    def _pattern(self, artist: str) -> re.Pattern | None:
        if artist not in self._patterns:
            terms = sorted(set(self.terms.get(artist, [])), key=len, reverse=True)  # longest first: full name wins
            alternatives = [
                r"[\s\-]+".join(re.escape(part) for part in re.findall(r"[^\s\-]+", term)) for term in terms
            ]
            self._patterns[artist] = (
                re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE) if alternatives else None
            )
        return self._patterns[artist]

    def _override(self, text: str) -> str | None:
        key = override_key(text)
        if key not in self.overrides:
            return None
        self.used_overrides.add(key)
        return apply_case(text, self.overrides[key])

    def _replace(self, matched: str) -> tuple[str, str]:
        """Replacement for one matched name and where it came from: "override", "g2p" (a predicted pronunciation
        was used for some word) or "cmudict"."""
        whole = self._override(matched)
        if whole is not None:
            return whole, "override"
        source = "cmudict"

        def substitute(word: re.Match) -> str:
            nonlocal source
            piece = word.group(0)
            chosen = self._override(piece)
            if chosen is not None:
                source = "override"
                return chosen
            if len(piece) < MIN_TERM_LEN or piece.lower() in KEEP_WORDS:
                return piece
            substitution = self.substitutor.substitute(piece)
            if substitution.approximate and source != "override":
                source = "g2p"
            return match_case(piece, substitution.word)

        return re.sub(r"[^\W\d_]+", substitute, matched), source

    def anonymize(self, artist: str, text: str) -> tuple[str, Counter]:
        """The text with the artist's names replaced, and a count per (original, replacement, source)."""
        pattern = self._pattern(artist)
        replaced: Counter = Counter()
        if pattern is None:
            return text, replaced

        def substitute(match: re.Match) -> str:
            matched = match.group(0)
            if not re.search(r"[\s\-]", matched) and not matched[:1].isupper():
                return matched  # a single word counts as a name only when capitalised
            replacement, source = self._replace(matched)
            if replacement != matched:
                replaced[(matched.lower(), replacement.lower(), source)] += 1
            return replacement

        return pattern.sub(substitute, text), replaced


@dataclass(frozen=True)
class AnonymizationConfig:
    output_dir: Path = PROJECT_ROOT / "outputs"  # where lyrics_*.csv are read from
    musicbrainz_contact: str | None = None  # e-mail or URL for the MusicBrainz User-Agent, needed to fetch aliases

    @classmethod
    def from_env(cls) -> "AnonymizationConfig":
        """Reads OUTPUT_DIR and MUSICBRAINZ_CONTACT from the environment (`make anonymize OUTPUT_DIR=...`)."""
        defaults = cls()
        return cls(
            output_dir=Path(os.environ.get("OUTPUT_DIR") or defaults.output_dir),
            musicbrainz_contact=os.environ.get("MUSICBRAINZ_CONTACT") or None,
        )

    @property
    def aliases_path(self) -> Path:
        return self.output_dir / "artist_aliases.json"

    @property
    def overrides_path(self) -> Path:
        return self.output_dir / "replacement_overrides.json"

    @property
    def anonymized_dir(self) -> Path:
        return self.output_dir / "anonymized"


def load_aliases(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_aliases(path: Path, aliases: dict[str, dict]) -> None:
    path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_overrides(path: Path) -> dict[str, str]:
    """The hand-written replacements (see the module docstring); empty if the file does not exist."""
    if not path.exists():
        return {}
    try:
        overrides = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}") from error
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in overrides.items()
    ):
        raise SystemExit(f'{path} must be a JSON object of non-empty strings, e.g. {{"kanye": "candy"}}')
    return {key: value.strip() for key, value in overrides.items()}


def lyric_words(records: Iterable[dict]) -> Iterable[str]:
    for record in records:
        yield from re.findall(r"[a-z]+", strip_accents(record["lyrics"]).lower())


class LyricsAnonymizer:
    """Reads every lyrics_*.csv in the output dir and writes anonymized copies to <output dir>/anonymized/."""

    def __init__(self, config: AnonymizationConfig):
        self.config = config

    def run(self) -> None:
        config = self.config
        files = sorted(config.output_dir.glob(LYRICS_FILE_GLOB))
        if not files:
            raise SystemExit(f"No {LYRICS_FILE_GLOB} files in {config.output_dir}, run `make data` first")
        records = {path: load_records(path) for path in files}
        artists = sorted({row["artist"] for rows in records.values() for row in rows})

        aliases = self._collect_aliases(artists)
        terms = {artist: entry["terms"] for artist, entry in aliases.items()}
        substitutor = PhoneticSubstitutor(lyric_words(row for rows in records.values() for row in rows))
        overrides = load_overrides(config.overrides_path)
        anonymizer = Anonymizer(terms, substitutor, overrides)

        config.anonymized_dir.mkdir(parents=True, exist_ok=True)
        replacements: Counter = Counter()
        for path, rows in records.items():
            for row in rows:
                row["lyrics"], replaced = anonymizer.anonymize(row["artist"], row["lyrics"])
                for (original, replacement, source), count in replaced.items():
                    replacements[(row["artist"], original, replacement, source)] += count
            fields = list(rows[0]) if rows else []
            save_records(config.anonymized_dir / path.name, fields, rows)
            print(f"{path.name}: {len(rows)} songs -> {config.anonymized_dir / path.name}")

        report = [
            dict(zip(REPLACEMENT_FIELDS, (artist, original, replacement, count, source), strict=True))
            for (artist, original, replacement, source), count in sorted(replacements.items())
        ]
        save_records(config.anonymized_dir / "replacements.csv", REPLACEMENT_FIELDS, report)
        by_source = Counter(row["source"] for row in report)
        print(
            f"{sum(replacements.values())} replacements ({len(report)} distinct: {by_source['override']} override, "
            f"{by_source['g2p']} g2p, {by_source['cmudict']} cmudict) -> {config.anonymized_dir / 'replacements.csv'}"
        )
        unused = sorted(set(anonymizer.overrides) - anonymizer.used_overrides)
        if unused:
            print(f"Overrides never applied (not part of an artist's terms, or not in the lyrics): {unused}")

    def _collect_aliases(self, artists: list[str]) -> dict[str, dict]:
        """Aliases for all artists, fetching the missing ones from MusicBrainz (saved after each artist)."""
        path = self.config.aliases_path
        aliases = load_aliases(path)
        missing = [artist for artist in artists if artist not in aliases]
        if not missing:
            return {artist: aliases[artist] for artist in artists}
        if not self.config.musicbrainz_contact:
            raise SystemExit("Set MUSICBRAINZ_CONTACT (an e-mail or URL) to fetch aliases, MusicBrainz requires it")

        client = MusicBrainzClient(self.config.musicbrainz_contact)
        for artist in missing:
            try:
                aliases[artist] = client.artist_names(artist)
            except requests.RequestException as error:  # not cached, so the next run tries again
                print(f"{artist}: MusicBrainz failed ({error}), using the name only")
                aliases[artist] = {"terms": build_terms([artist]), "failed": True}
                continue
            entry = aliases[artist]
            print(f"{artist}: {'matched ' + entry['matched_name'] if entry['mbid'] else 'no match'}, {entry['terms']}")
            save_aliases(path, {name: value for name, value in aliases.items() if not value.get("failed")})
        return {artist: aliases[artist] for artist in artists}
