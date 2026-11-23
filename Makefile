
.PHONY: install fmt lint typecheck test check run data audio anonymize generate generate-check probe embed probe-embeddings clean

install:
	uv sync
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest --cov

check: lint typecheck test

run:
	uv run encoder_exorcist

# Generate the test and held-out lyrics datasets (needs LASTFM_API_KEY and GENIUS_ACCESS_TOKEN in .env).
# Override the defaults on the command line, e.g.: make data GENRES="pop rock" N_ARTISTS=5 HELDOUT_SONGS=3
#   GENRES            genres to process: rock pop hip-hop R&B country (default: all)
#   N_ARTISTS         artists per genre (default: 10)
#   TEST_SONGS        songs per artist in the test set (default: 5)
#   HELDOUT_SONGS     songs per artist in the held-out set (default: 5)
#   TEST_BEFORE       test set: songs released before this date, YYYY-MM-DD (default: 2025-01-01)
#   HELDOUT_AFTER     held-out set: songs released after this date, YYYY-MM-DD (default: 2025-01-31)
#   TITLE_SIMILARITY  titles at least this similar (0-1) count as the same song (default: 0.9)
#   POOL_MULTIPLIER   Last.fm artist pool = POOL_MULTIPLIER x N_ARTISTS (default: 2)
#   OUTPUT_DIR        where the CSV files go (default: outputs)
data:
	uv run python scripts/data_preparation.py

# Replace artist names in the lyrics with phonetically similar words (MusicBrainz aliases + CMUdict).
# Reads OUTPUT_DIR/lyrics_*.csv, writes OUTPUT_DIR/anonymized/ and caches aliases in OUTPUT_DIR/artist_aliases.json.
# Your own replacements go in OUTPUT_DIR/replacement_overrides.json, e.g. {"kanye": "candy"} (see anonymization.py).
#   MUSICBRAINZ_CONTACT  e-mail or URL put in the User-Agent, required by MusicBrainz (can also be set in .env)
#   OUTPUT_DIR           same as for `make data` (default: outputs)
anonymize:
	uv run python scripts/anonymize_lyrics.py

# Download reference audio (wav) for the songs in the lyrics CSV files.
#   CSV         lyrics CSV files, space separated (default: all outputs/lyrics_*.csv)
#   AUDIO_DIR   where the wav files go (default: audio_reference)
#   COOKIES     browser whose YouTube login to use, e.g. chrome (needed when YouTube asks to sign in)
# e.g.: make audio CSV=outputs/lyrics_pop_4artists_20songs.csv
audio:
	uv run python scripts/download_audio.py $(CSV) $(if $(AUDIO_DIR),--output-dir $(AUDIO_DIR)) $(if $(COOKIES),--cookies-from-browser $(COOKIES))

# Generate songs from the anonymized lyrics with ACE-Step and record activations (see encoder_exorcist/generation.py).
# Reads outputs/anonymized/lyrics_*.csv, writes outputs/generated/<CONFIG>/ (audio, activations, manifest.csv).
# Interrupted runs continue where they stopped.
#   CONFIG    no-cot (5Hz LM off, default) or with-cot (LM writes caption/metadata and audio codes)
#   SEEDS     generations per song (default: 1)
#   LIMIT     only the first N songs, for a quick test
#   CSV       lyrics CSV files, space separated (default: all outputs/anonymized/lyrics_*.csv)
#   GENERATED_DIR  where the runs go (default: outputs/generated)
# e.g.: make generate CONFIG=with-cot SEEDS=2
#       make generate CSV="outputs/anonymized_50songs/lyrics_*.csv" GENERATED_DIR=outputs/generated_50songs
GENERATE_ARGS = $(CSV) $(if $(CONFIG),--config $(CONFIG)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(LIMIT),--limit $(LIMIT)) \
	$(if $(GENERATED_DIR),--output-dir $(GENERATED_DIR))
generate:
	uv run python scripts/generate_songs.py $(GENERATE_ARGS)

# Generate one song and check the hooks against docs/hook_points.md, section 7 (writes nothing).
generate-check:
	uv run python scripts/generate_songs.py --check $(GENERATE_ARGS)

# Train linear probes for artist identity on the activations of a generation run (see encoder_exorcist/probing.py).
# Reads GENERATED_DIR/<CONFIG>/, writes PROBE_DIR/<CONFIG>/results.csv and per_artist.csv.
#   CONFIG         generation run: no-cot (default) or with-cot
#   PERMUTATIONS   permutations for the stream outputs (except timbre_out), lyric layers and baselines (default: 1000, 0 disables the test)
#   GENERATED_DIR  generation runs (default: outputs/generated)
#   LYRICS_DIR     lyrics_*.csv the songs were generated from, for the baselines (default: outputs/anonymized)
#   PROBE_DIR      where the results go (default: outputs/probe)
# e.g.: make probe CONFIG=with-cot PERMUTATIONS=0
#       make probe GENERATED_DIR=outputs/generated_50songs LYRICS_DIR=outputs/anonymized_50songs PROBE_DIR=outputs/probe_50songs
probe:
	uv run python scripts/train_probe.py $(if $(CONFIG),--config $(CONFIG)) $(if $(PERMUTATIONS),--permutations $(PERMUTATIONS)) \
		$(if $(GENERATED_DIR),--generated-dir $(GENERATED_DIR)) $(if $(LYRICS_DIR),--lyrics-dir $(LYRICS_DIR)) \
		$(if $(PROBE_DIR),--output-dir $(PROBE_DIR))

# Train linear probes on the CLAP / MuQ-MuLan embeddings from `make embed` (see scripts/train_embedding_probe.py).
# Writes PROBE_EMB_DIR/<SOURCE>[/<CONFIG>]/results.csv and per_artist.csv (all artists and within each genre).
#   SOURCE         generated or reference (original recordings) (required)
#   CONFIG         generation run for SOURCE=generated (default: no-cot)
#   MODELS         space separated: clap mulan (default: both)
#   PERMUTATIONS   permutation test size (default: 1000, 0 disables the test)
#   EMBED_DIR, GENERATED_DIR  as for `make embed`; PROBE_EMB_DIR  where the results go (default: outputs/probe_embeddings)
# e.g.: make probe-embeddings SOURCE=generated CONFIG=no-cot
probe-embeddings:
	uv run python scripts/train_embedding_probe.py --source $(SOURCE) $(if $(CONFIG),--config $(CONFIG)) \
		$(if $(MODELS),--models $(MODELS)) $(if $(PERMUTATIONS),--permutations $(PERMUTATIONS)) \
		$(if $(EMBED_DIR),--embeddings-dir $(EMBED_DIR)) $(if $(GENERATED_DIR),--generated-dir $(GENERATED_DIR)) \
		$(if $(PROBE_EMB_DIR),--output-dir $(PROBE_EMB_DIR))

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info

# Embed songs with LAION-CLAP or MuQ-MuLan (see encoder_exorcist/embeddings.py), writes EMBED_DIR/<MODEL>/<SOURCE>/.
#   MODEL          clap or mulan (required)
#   SOURCE         generated (songs of a generation run) or reference (audio_reference/) (required)
#   CONFIG         generation run for SOURCE=generated (default: no-cot)
#   GENERATED_DIR  generation runs (default: outputs/generated)
#   AUDIO_DIR      reference recordings (default: audio_reference)
#   EMBED_DIR      where the embeddings go (default: outputs/embeddings)
#   LIMIT          only the first N songs, for a quick test
# e.g.: make embed MODEL=clap SOURCE=generated CONFIG=no-cot
embed:
	uv run python scripts/extract_embeddings.py --model $(MODEL) --source $(SOURCE) $(if $(CONFIG),--config $(CONFIG)) \
		$(if $(GENERATED_DIR),--generated-dir $(GENERATED_DIR)) $(if $(AUDIO_DIR),--audio-dir $(AUDIO_DIR)) \
		$(if $(EMBED_DIR),--output-dir $(EMBED_DIR)) $(if $(LIMIT),--limit $(LIMIT))
