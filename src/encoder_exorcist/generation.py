"""Generate songs from the anonymized lyrics with ACE-Step 1.5 and record activations while generating.

Settings follow "Ghost in the Encoder" (Sec. 3.2): acestep-v15-turbo, duration 180 s, only the lyrics are given
(no caption, bpm, key or timbre reference). Two configurations:
  - no-cot:   the 5Hz LM is off, the DiT sees the lyrics and an empty caption.
  - with-cot: the 5Hz LM writes caption, bpm, key, time signature and language from the lyrics (CoT) and the audio
              codes that the DiT gets as context latents (ACE-Step defaults with thinking=True).

Every song gets a deterministic seed derived from artist, title and seed index, so reruns and the two configurations
use the same noise for the same song. For every song the generator writes
  <output_dir>/<config>/audio/<song_id>.flac         generated audio
  <output_dir>/<config>/activations/<song_id>.npz    mean-pooled activations (ActivationRecorder, docs/hook_points.md)
and appends a row to <output_dir>/<config>/manifest.csv. Songs already in the manifest (with both files on disk) are
skipped, so an interrupted run continues where it stopped. Songs that fail go to failed.csv and are retried next run.
"""

import csv
import logging
import os
import shutil
import tempfile
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from encoder_exorcist.activation_recorder import DIT_LAYERS, DIT_POINTS, ActivationRecorder, save_song
from encoder_exorcist.data.audio_reference import slugify
from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT, load_records, save_records

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DIT_CONFIG = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-1.7B"
DURATION = 180.0  # seconds, as in the paper
N_STEPS = 8  # turbo default
AUDIO_FRAMES_PER_SECOND = 25  # DiT latent rate, halved by patch size 2 inside the DiT

# Configuration name -> whether the 5Hz LM runs. With the LM, every CoT switch is on (ACE-Step defaults with
# thinking=True); without it, all of them are off.
CONFIGS: dict[str, bool] = {"no-cot": False, "with-cot": True}

MANIFEST_FIELDS = [
    "song_id",
    "config",
    "genre",
    "artist",
    "title",
    "release_date",
    "seed_index",
    "seed",
    "duration",
    "audio_path",
    "activations_path",
    "lm_caption",
    "lm_bpm",
    "lm_keyscale",
    "lm_timesignature",
    "lm_language",
    "generation_time_s",
]
FAILED_FIELDS = ["song_id", "artist", "title", "seed_index", "reason"]


@dataclass(frozen=True)
class SongJob:
    """One generation: a song's lyrics and the index of the seed to use for it."""

    genre: str
    artist: str
    title: str
    release_date: str
    lyrics: str
    seed_index: int

    @property
    def song_id(self) -> str:
        """File-name-safe id, e.g. "ariana_grande_thank_u_next_s0"."""
        return f"{slugify(self.artist)}_{slugify(self.title)}_s{self.seed_index}"

    @property
    def seed(self) -> int:
        """Deterministic seed for this song and seed index (crc32, fits in 32 bits)."""
        return zlib.crc32(f"{self.artist}|{self.title}|{self.seed_index}".encode())


def load_jobs(csv_paths: Iterable[Path], n_seeds: int = 1) -> list[SongJob]:
    """Read lyrics CSV files (columns as in lyrics_dataset.LYRICS_FIELDS) and make `n_seeds` jobs per song."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    jobs = []
    for path in csv_paths:
        for record in load_records(path):
            for seed_index in range(n_seeds):
                jobs.append(
                    SongJob(
                        genre=record["genre"],
                        artist=record["artist"],
                        title=record["title"],
                        release_date=record.get("release_date", ""),
                        lyrics=record["lyrics"],
                        seed_index=seed_index,
                    )
                )
    ids = [job.song_id for job in jobs]
    duplicates = sorted({song_id for song_id in ids if ids.count(song_id) > 1})
    if duplicates:
        raise ValueError(f"songs with the same id (artist + title): {duplicates}")
    return jobs


def build_params(job: SongJob, config_name: str, duration: float = DURATION) -> Any:
    """GenerationParams for one song: only the lyrics are given, the rest is left to the model (or the LM)."""
    from acestep.inference import GenerationParams

    return GenerationParams(
        lyrics=job.lyrics,
        caption="",
        duration=duration,
        bpm=None,
        keyscale="",
        timesignature="",
        inference_steps=N_STEPS,
        seed=job.seed,
        thinking=CONFIGS[config_name],
        use_cot_caption=CONFIGS[config_name],
        use_cot_language=CONFIGS[config_name],
        use_cot_metas=CONFIGS[config_name],
    )


def check_record(record: dict[str, np.ndarray], n_lyric_layers: int = 8, d: int = 2048, d_kv: int = 1024) -> None:
    """Raise ValueError if a record from ActivationRecorder.pop_song() does not have the shapes in hook_points.md."""
    expected = {
        "lyric_out": (d,),
        "text_out": (d,),
        "timbre_out": (d,),
        "merged_out": (d,),
        "cond_embed_out": (d,),
        "lyric_layers": (n_lyric_layers, d),
        "kv": (len(DIT_LAYERS), 2, d_kv),
        "dit": (len(DIT_LAYERS), len(DIT_POINTS), N_STEPS, d),
    }
    wrong = {key: record[key].shape for key, shape in expected.items() if record[key].shape != shape}
    if wrong:
        raise ValueError(f"unexpected activation shapes {wrong}, expected {expected}")
    if not all(np.isfinite(record[key]).all() for key in expected):
        raise ValueError("activations contain NaN or inf")


# Called with (job, audio_path, activations_path); writes both files and returns the LM-related manifest columns.
GenerateFn = Callable[[SongJob, Path, Path], dict[str, Any]]


class GenerationRun:
    """Output folder of one configuration: file paths, the manifest and resuming."""

    def __init__(self, output_dir: Path, config_name: str) -> None:
        if config_name not in CONFIGS:
            raise ValueError(f"unknown configuration {config_name!r}, choose from {list(CONFIGS)}")
        self.config_name = config_name
        self.dir = output_dir / config_name
        self.audio_dir = self.dir / "audio"
        self.activations_dir = self.dir / "activations"
        self.manifest_path = self.dir / "manifest.csv"
        self.failed_path = self.dir / "failed.csv"

    def audio_path(self, job: SongJob) -> Path:
        return self.audio_dir / f"{job.song_id}.flac"

    def activations_path(self, job: SongJob) -> Path:
        return self.activations_dir / f"{job.song_id}.npz"

    def done_ids(self) -> set[str]:
        """Songs in the manifest whose audio and activations both exist."""
        return {
            row["song_id"]
            for row in load_records(self.manifest_path)
            if (self.dir / row["audio_path"]).exists() and (self.dir / row["activations_path"]).exists()
        }

    def run(self, jobs: list[SongJob], generate: GenerateFn, duration: float = DURATION) -> tuple[int, int]:
        """Generate every job not done yet; returns (generated, failed). Failures are logged and saved, not raised."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.activations_dir.mkdir(parents=True, exist_ok=True)
        done = self.done_ids()
        todo = [job for job in jobs if job.song_id not in done]
        logger.info("%s: %d songs, %d already done, %d to generate", self.config_name, len(jobs), len(done), len(todo))

        failed: list[dict] = []
        generated = 0
        for i, job in enumerate(todo, start=1):
            start = time.perf_counter()
            try:
                lm_columns = generate(job, self.audio_path(job), self.activations_path(job))
            except Exception as error:  # one bad song must not stop a long run
                logger.exception("failed: %s", job.song_id)
                failed.append({**vars(job), "song_id": job.song_id, "reason": repr(error)})
                continue
            elapsed = time.perf_counter() - start
            self._append_manifest(
                {
                    **vars(job),
                    "song_id": job.song_id,
                    "config": self.config_name,
                    "seed": job.seed,
                    "duration": duration,
                    "audio_path": self.audio_path(job).relative_to(self.dir).as_posix(),
                    "activations_path": self.activations_path(job).relative_to(self.dir).as_posix(),
                    "generation_time_s": f"{elapsed:.2f}",
                    **lm_columns,
                }
            )
            generated += 1
            logger.info("[%d/%d] %s (%.1fs)", i, len(todo), job.song_id, elapsed)

        save_records(self.failed_path, FAILED_FIELDS, failed)
        return generated, len(failed)

    def _append_manifest(self, row: dict) -> None:
        new_file = not self.manifest_path.exists()
        with self.manifest_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


class AceStepGenerator:
    """Loads ACE-Step once and generates one song at a time with the ActivationRecorder attached."""

    def __init__(self, config_name: str, duration: float = DURATION, device: str = "cuda") -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model, ensure_main_model

        self.config_name = config_name
        self.duration = duration
        self.recorder = ActivationRecorder()

        ok, message = ensure_main_model(CHECKPOINTS_DIR)
        if not ok:
            raise RuntimeError(message)
        self.dit_handler = AceStepHandler()
        status, ok = self.dit_handler.initialize_service(
            project_root=str(PROJECT_ROOT), config_path=DIT_CONFIG, device=device
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {status}")

        self.llm_handler = LLMHandler()
        if CONFIGS[config_name]:
            ok, message = ensure_lm_model(model_name=LM_MODEL, checkpoints_dir=CHECKPOINTS_DIR)
            if not ok:
                raise RuntimeError(message)
            status, ok = self.llm_handler.initialize(
                checkpoint_dir=str(CHECKPOINTS_DIR), lm_model_path=LM_MODEL, device=device
            )
            if not ok:
                raise RuntimeError(f"LM init failed: {status}")

    def __call__(self, job: SongJob, audio_path: Path, activations_path: Path) -> dict[str, Any]:
        """Generate `job`, write the flac and the npz atomically (temp file, then rename), return LM columns."""
        from acestep.inference import GenerationConfig, generate_music

        params = build_params(job, self.config_name, self.duration)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[job.seed], audio_format="flac")
        with tempfile.TemporaryDirectory(dir=audio_path.parent) as tmp_dir:
            with self.recorder.attach(self.dit_handler.model):
                result = generate_music(self.dit_handler, self.llm_handler, params, config, save_dir=tmp_dir)
            record = self.recorder.pop_song()  # also resets the recorder after a failed generation
            if not result.success or not result.audios or not result.audios[0]["path"]:
                raise RuntimeError(f"generation failed: {result.error or result.status_message}")
            check_record(record)

            tmp_npz = Path(tmp_dir) / "activations.npz"
            save_song(tmp_npz, record, self.config_name)
            shutil.move(result.audios[0]["path"], audio_path)
            os.replace(tmp_npz, activations_path)

        metadata = result.extra_outputs.get("lm_metadata") or {}
        return {
            "lm_caption": metadata.get("caption", ""),
            "lm_bpm": metadata.get("bpm", ""),
            "lm_keyscale": metadata.get("keyscale", ""),
            "lm_timesignature": metadata.get("timesignature", ""),
            "lm_language": metadata.get("language", ""),
        }

    def verify_hooks(self, job: SongJob) -> dict[str, Any]:
        """Generate one song with extra shape hooks and check docs/hook_points.md, section 7. Writes no files.

        Returns the observed sequence lengths and call counts; raises AssertionError on a mismatch.
        """
        from acestep.inference import GenerationConfig, generate_music

        model: Any = self.dit_handler.model
        calls: dict[str, list[tuple[int, ...]]] = {}

        def shape_hook(name: str, take_input: bool):
            def hook(module: Any, args: tuple, output: Any = None) -> None:
                x = args[0] if take_input else (output[0] if isinstance(output, tuple) else output)
                x = getattr(x, "last_hidden_state", x)
                calls.setdefault(name, []).append(tuple(x.shape))

            return hook

        layer = model.decoder.layers[DIT_LAYERS[0]]
        handles = [
            model.encoder.lyric_encoder.register_forward_hook(shape_hook("lyric", False)),
            model.encoder.text_projector.register_forward_hook(shape_hook("text", False)),
            model.encoder.timbre_encoder.register_forward_hook(shape_hook("timbre", False)),
            model.encoder.register_forward_hook(shape_hook("merged", False)),
            layer.cross_attn.k_proj.register_forward_hook(shape_hook("k", False)),
            layer.cross_attn_norm.register_forward_pre_hook(shape_hook("pre_xattn", True)),
        ]
        params = build_params(job, self.config_name, self.duration)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[job.seed])
        try:
            with tempfile.TemporaryDirectory() as tmp_dir, self.recorder.attach(model):
                result = generate_music(self.dit_handler, self.llm_handler, params, config, save_dir=tmp_dir)
            record = self.recorder.pop_song()
        finally:
            for handle in handles:
                handle.remove()
        assert result.success, f"generation failed: {result.error}"
        check_record(record)

        t_aud = round(self.duration * AUDIO_FRAMES_PER_SECOND / 2)
        assert calls["pre_xattn"] == [(1, t_aud, 2048)] * N_STEPS, f"pre_xattn calls: {calls['pre_xattn']}"
        assert len(calls["k"]) == 1, f"k_proj fired {len(calls['k'])} times, expected once (cross-attention cache)"
        t_lyr, t_timbre, t_txt, t_cond = (calls[name][0][1] for name in ("lyric", "timbre", "text", "merged"))
        assert t_cond == t_lyr + t_timbre + t_txt, f"T_cond={t_cond} != {t_lyr}+{t_timbre}+{t_txt}"
        return {
            "T_lyr": t_lyr,
            "T_timbre": t_timbre,
            "T_txt": t_txt,
            "T_cond": t_cond,
            "T_aud": t_aud,
            "calls": {name: len(shapes) for name, shapes in calls.items()},
        }
