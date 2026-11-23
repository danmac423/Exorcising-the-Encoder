import numpy as np
import pytest

from encoder_exorcist.probing import load_embedding_data, point_group, run_embedding_probes

ARTISTS = {"A": "rock", "B": "rock", "C": "pop", "D": "pop"}
SONGS_PER_ARTIST = 10


def make_files(tmp_path, signal: float, models=("clap",)):
    """Embeddings where artist k has its centre on axis k (scaled by `signal`) plus noise; returns files, metadata."""
    rng = np.random.default_rng(0)
    songs, ids = {}, []
    centres = np.eye(len(ARTISTS), 16)
    for k, (artist, genre) in enumerate(ARTISTS.items()):
        for i in range(SONGS_PER_ARTIST):
            song_id = f"{artist}_{i}"
            ids.append((song_id, centres[k]))
            songs[song_id] = {"artist": artist, "title": f"t{i}", "genre": genre}
    files = {}
    for model in models:
        path = tmp_path / f"{model}.npz"
        embeddings = np.stack([signal * c + rng.normal(size=16) for _, c in ids])
        np.savez(path, song_ids=np.array([s for s, _ in ids]), embeddings=embeddings)
        files[model] = path
    return files, songs


def test_load_embedding_data_aligns_models_and_metadata(tmp_path):
    files, songs = make_files(tmp_path, 5.0, models=("clap", "mulan"))
    del songs["A_0"]  # no metadata -> skipped
    data = load_embedding_data(files, songs)
    assert len(data.song_ids) == 39
    assert set(data.features) == {"clap", "mulan"}
    assert data.features["clap"].shape == (39, 16)
    assert data.genres is not None and set(data.genres) == {"rock", "pop"}


def test_load_embedding_data_without_overlap_fails(tmp_path):
    files, _ = make_files(tmp_path, 5.0)
    with pytest.raises(ValueError):
        load_embedding_data(files, {})


def test_probe_finds_artist_signal_in_every_scope(tmp_path):
    files, songs = make_files(tmp_path, 6.0)
    rows, per_artist = run_embedding_probes(load_embedding_data(files, songs), n_permutations=20, n_jobs=1)
    by_scope = {r["scope"]: r for r in rows}
    assert set(by_scope) == {"all", "rock", "pop"}
    assert float(by_scope["all"]["accuracy"]) > 0.9
    assert by_scope["all"]["chance"] == "0.2500" and by_scope["rock"]["chance"] == "0.5000"
    assert by_scope["all"]["group"] == "embedding" and point_group("mulan") == "embedding"
    assert float(by_scope["all"]["p_value"]) < 0.1
    assert {r["scope"] for r in per_artist} == {"all", "rock", "pop"}


def test_probe_on_noise_is_near_chance(tmp_path):
    files, songs = make_files(tmp_path, 0.0)
    rows, _ = run_embedding_probes(load_embedding_data(files, songs), n_permutations=0, n_jobs=1)
    assert float(next(r for r in rows if r["scope"] == "all")["accuracy"]) < 0.5
