from pathlib import Path

import numpy as np
import pytest

from encoder_exorcist.activation_recorder import DIT_LAYERS
from encoder_exorcist.data.lyrics_dataset import load_records
from encoder_exorcist.generation import CONFIGS, GenerationRun, SongJob, build_params, check_record, load_jobs

CSV_HEADER = "genre,artist,title,release_date,url,lyrics\n"


def make_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(CSV_HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def job(title: str = "Song", seed_index: int = 0) -> SongJob:
    return SongJob("pop", "Ariana Grande", title, "2018-11-03", "some lyrics", seed_index)


def valid_record() -> dict[str, np.ndarray]:
    n = len(DIT_LAYERS)
    return {
        **{key: np.zeros(2048) for key in ("lyric_out", "text_out", "timbre_out", "merged_out", "cond_embed_out")},
        "lyric_layers": np.zeros((8, 2048)),
        "kv": np.zeros((n, 2, 1024)),
        "dit": np.zeros((n, 3, 8, 2048)),
    }


def test_song_id_and_seed_are_deterministic_and_distinct():
    assert job("thank u, next").song_id == "ariana_grande_thank_u_next_s0"
    assert job().seed == job().seed
    assert len({job().seed, job(seed_index=1).seed, job("Other").seed}) == 3
    assert 0 <= job().seed < 2**32


def test_load_jobs_expands_seeds(tmp_path):
    csv_file = make_csv(tmp_path / "lyrics_pop.csv", ["pop,A,One,2020-01-01,u,la", "pop,B,Two,2021-01-01,u,da"])
    jobs = load_jobs([csv_file], n_seeds=2)
    assert [j.song_id for j in jobs] == ["a_one_s0", "a_one_s1", "b_two_s0", "b_two_s1"]
    assert jobs[0].lyrics == "la"


def test_load_jobs_rejects_duplicate_ids_and_bad_seeds(tmp_path):
    csv_file = make_csv(tmp_path / "lyrics_pop.csv", ["pop,A,Song!,d,u,x", "pop,A,Song?,d,u,y"])
    with pytest.raises(ValueError, match="same id"):
        load_jobs([csv_file])
    with pytest.raises(ValueError, match="n_seeds"):
        load_jobs([csv_file], n_seeds=0)


@pytest.mark.parametrize("config_name", list(CONFIGS))
def test_build_params_gives_only_lyrics(config_name):
    params = build_params(job(), config_name, duration=30)
    assert (params.lyrics, params.caption, params.bpm, params.keyscale) == ("some lyrics", "", None, "")
    assert params.duration == 30 and params.seed == job().seed
    assert params.thinking is (config_name == "with-cot")


def test_check_record_accepts_valid_and_rejects_wrong_shape_or_nan():
    check_record(valid_record())
    with pytest.raises(ValueError, match="shapes"):
        check_record({**valid_record(), "dit": np.zeros((len(DIT_LAYERS), 3, 7, 2048))})
    with pytest.raises(ValueError, match="NaN"):
        check_record({**valid_record(), "lyric_out": np.full(2048, np.nan)})


def test_run_writes_manifest_and_resumes(tmp_path):
    run = GenerationRun(tmp_path, "no-cot")
    seen = []

    def fake_generate(song: SongJob, audio_path: Path, activations_path: Path) -> dict:
        seen.append(song.song_id)
        if song.title == "Bad":
            raise RuntimeError("boom")
        audio_path.write_bytes(b"flac")
        activations_path.write_bytes(b"npz")
        return {"lm_caption": ""}

    jobs = [job("One"), job("Bad"), job("Two")]
    assert run.run(jobs, fake_generate) == (2, 1)
    rows = load_records(run.manifest_path)
    assert [row["song_id"] for row in rows] == ["ariana_grande_one_s0", "ariana_grande_two_s0"]
    assert rows[0]["audio_path"] == "audio/ariana_grande_one_s0.flac" and rows[0]["config"] == "no-cot"
    assert [row["song_id"] for row in load_records(run.failed_path)] == ["ariana_grande_bad_s0"]

    # A rerun only retries the failed song; a song whose file disappeared is generated again.
    run.audio_path(job("Two")).unlink()
    seen.clear()
    run.run(jobs, fake_generate)
    assert seen == ["ariana_grande_bad_s0", "ariana_grande_two_s0"]
    # ...and the manifest keeps one row per song.
    assert sorted(row["song_id"] for row in load_records(run.manifest_path)) == [
        "ariana_grande_one_s0",
        "ariana_grande_two_s0",
    ]


def write_files(song: SongJob, audio_path: Path, activations_path: Path) -> dict:
    audio_path.write_bytes(b"flac")
    activations_path.write_bytes(b"npz")
    return {}


def test_run_drops_a_truncated_manifest_row(tmp_path):
    run = GenerationRun(tmp_path, "no-cot")
    run.run([job("One"), job("Two")], write_files)
    lines = run.manifest_path.read_text(encoding="utf-8").splitlines()
    run.manifest_path.write_text("\n".join([*lines[:-1], lines[-1][:20]]) + "\n", encoding="utf-8")  # killed mid-write

    assert [j.song_id for j in run.pending([job("One"), job("Two")])] == ["ariana_grande_two_s0"]
    run.run([job("One"), job("Two")], write_files)
    assert len(load_records(run.manifest_path)) == 2


def test_other_duration_is_not_done(tmp_path):
    GenerationRun(tmp_path, "no-cot", duration=30).run([job()], write_files)
    run = GenerationRun(tmp_path, "no-cot", duration=180)
    assert run.pending([job()]) == [job()]
    run.run([job()], write_files)
    assert [row["duration"] for row in load_records(run.manifest_path)] == ["180"]


def test_run_rejects_unknown_config(tmp_path):
    with pytest.raises(ValueError, match="unknown configuration"):
        GenerationRun(tmp_path, "cot")
