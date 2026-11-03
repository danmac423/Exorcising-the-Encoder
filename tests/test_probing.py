from pathlib import Path

import numpy as np
import pytest

from encoder_exorcist.activation_recorder import DIT_LAYERS, save_song
from encoder_exorcist.data.lyrics_dataset import save_records
from encoder_exorcist.probing import (
    ProbeData,
    load_probe_data,
    make_folds,
    point_group,
    record_features,
    run_probes,
    wilson_interval,
)

D, KV, STEPS = 16, 8, 2


def fake_record(rng: np.random.Generator, signal: np.ndarray) -> dict[str, np.ndarray]:
    """A record with ActivationRecorder's layout; `lyric_out` carries `signal`, everything else is noise."""
    n = len(DIT_LAYERS)
    noise = lambda *shape: rng.standard_normal(shape).astype(np.float32)
    return {
        "lyric_out": signal + 0.1 * noise(D),
        "text_out": noise(D),
        "timbre_out": noise(D),
        "merged_out": noise(D),
        "cond_embed_out": noise(D),
        "lyric_layers": noise(3, D),
        "kv": noise(n, 2, KV),
        "dit": noise(n, 3, STEPS, D),
        "dit_layers": np.array(DIT_LAYERS),
    }


def write_run(run_dir: Path, n_artists: int = 4, songs: int = 10, seeds: int = 1) -> None:
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((n_artists, D)).astype(np.float32) * 3
    (run_dir / "activations").mkdir(parents=True)
    rows = []
    for a in range(n_artists):
        for s in range(songs):
            for seed in range(seeds):
                song_id = f"artist{a}_song{s}_s{seed}"
                save_song(run_dir / "activations" / f"{song_id}.npz", fake_record(rng, centres[a]), "no-cot")
                rows.append(
                    {
                        "song_id": song_id,
                        "artist": f"Artist {a}",
                        "title": f"Song {s}",
                        "activations_path": f"activations/{song_id}.npz",
                    }
                )
    save_records(run_dir / "manifest.csv", ["song_id", "artist", "title", "activations_path"], rows)


def test_record_features_flattens_all_points():
    record = fake_record(np.random.default_rng(0), np.zeros(D, dtype=np.float32))
    features = record_features(record)
    n = len(DIT_LAYERS)
    assert len(features) == 5 + 3 + 2 * n + n * 3 * (STEPS + 1)
    assert features["k_12"].shape == (KV,) and features["dit_0_xattn_out_t1"].shape == (D,)
    np.testing.assert_allclose(features["dit_0_pre_xattn_mean"], record["dit"][0, 0].mean(axis=0))


def test_point_group():
    assert [point_group(n) for n in ("merged_out", "lyric_L3", "v_7", "dit_0_pre_xattn_t2", "tfidf")] == [
        "stream",
        "lyric",
        "kv",
        "dit",
        "baseline",
    ]


def test_folds_keep_all_seeds_of_a_song_together(tmp_path):
    write_run(tmp_path, seeds=2)
    data = load_probe_data(tmp_path)
    for train, test in make_folds(data.artists, data.groups):
        assert not set(data.groups[train]) & set(data.groups[test])


def test_wilson_interval_contains_estimate():
    low, high = wilson_interval(30, 100)
    assert low < 0.3 < high and 0 <= low and high <= 1


def test_probe_finds_signal_and_not_noise(tmp_path):
    write_run(tmp_path)
    data = load_probe_data(tmp_path)
    rows, per_artist = run_probes(
        data, points=["lyric_out", "timbre_out"], permuted=["lyric_out"], n_permutations=20, baselines=False, n_jobs=1
    )
    by_point = {row["point"]: row for row in rows}
    assert float(by_point["lyric_out"]["accuracy"]) > 0.9
    assert float(by_point["timbre_out"]["accuracy"]) < 0.6
    assert float(by_point["lyric_out"]["p_value"]) < 0.1 and by_point["timbre_out"]["p_value"] == ""
    assert {r["artist"] for r in per_artist if r["point"] == "lyric_out"} == {f"Artist {a}" for a in range(4)}


def test_tfidf_baseline_runs_on_lyrics():
    artists = np.repeat(np.array(["A", "B"]), 10)
    lyrics = np.array([f"{a.lower()}word common text song{i}" for i, a in enumerate(artists)], dtype=object)
    data = ProbeData(
        song_ids=[str(i) for i in range(20)],
        artists=artists,
        groups=np.arange(20).astype(str),
        lyrics=lyrics,
        features={},
    )
    from encoder_exorcist.probing import evaluate_point, make_folds, tfidf_probe

    folds = make_folds(data.artists, data.groups)
    row, _ = evaluate_point("tfidf", tfidf_probe(), data.lyrics, data.artists, data.groups, folds)
    assert float(row["accuracy"]) == 1.0


def test_unknown_point_is_rejected(tmp_path):
    write_run(tmp_path)
    with pytest.raises(ValueError, match="unknown hook points"):
        run_probes(load_probe_data(tmp_path), points=["nope"], baselines=False)


def test_constant_feature_is_not_significant(tmp_path):
    # Regression: with fixed real-label folds the permutation null fell below chance and gave p=0.001 here.
    write_run(tmp_path)
    data = load_probe_data(tmp_path)
    data.features["const"] = np.ones((len(data.artists), D), dtype=np.float32)
    rows, _ = run_probes(data, points=["const"], permuted=["const"], n_permutations=30, baselines=False, n_jobs=1)
    assert float(rows[0]["accuracy"]) == pytest.approx(0.25)
    assert float(rows[0]["p_value"]) > 0.3
