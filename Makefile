.PHONY: install fmt lint typecheck test check run data audio clean

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

# Download reference audio (wav) for the songs in the lyrics CSV files.
#   CSV         lyrics CSV files, space separated (default: all outputs/lyrics_*.csv)
#   AUDIO_DIR   where the wav files go (default: audio_reference)
# e.g.: make audio CSV=outputs/lyrics_pop_4artists_20songs.csv
audio:
	uv run python scripts/download_audio.py $(CSV) $(if $(AUDIO_DIR),--output-dir $(AUDIO_DIR))

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
