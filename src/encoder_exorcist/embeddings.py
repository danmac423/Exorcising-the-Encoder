"""Song-level audio embeddings from LAION-CLAP and MuQ-MuLan.

A song is loaded as mono audio, resampled to the sample rate of the model and cut into fixed windows (the models see
only a few seconds at a time). Every window is embedded, the window embeddings are averaged and the mean is
L2-normalized, which gives one unit vector per song. Model libraries are imported lazily, so importing this module
does not need them.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torchaudio

from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CLAP_CHECKPOINT = CHECKPOINTS_DIR / "music_audioset_epoch_15_esc_90.14.pt"
MULAN_REPO = "OpenMuQ/MuQ-MuLan-large"
MODELS = ("clap", "mulan")


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """Mono waveform [samples] at `sample_rate`."""
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def split_windows(wav: torch.Tensor, window: int) -> torch.Tensor:
    """Non-overlapping windows [n, window]. A last piece shorter than half a window is dropped, a longer one is
    zero-padded; audio shorter than that still gives one (padded) window."""
    n_full, rest = divmod(wav.shape[0], window)
    windows = list(wav[: n_full * window].reshape(n_full, window))
    if rest >= window // 2 or not windows:
        windows.append(torch.nn.functional.pad(wav[n_full * window :], (0, window - rest)))
    return torch.stack(windows)


def aggregate(window_embeddings: np.ndarray) -> np.ndarray:
    """Mean of the window embeddings [n, d], L2-normalized to a single [d] vector."""
    mean = np.asarray(window_embeddings, dtype=np.float64).mean(axis=0)
    return (mean / max(np.linalg.norm(mean), 1e-12)).astype(np.float32)


class Embedder(ABC):
    name: str
    sample_rate: int
    window_seconds: float
    batch_size: int = 8

    @abstractmethod
    def embed_windows(self, windows: torch.Tensor) -> np.ndarray:
        """Embeddings [n, d] of windows [n, window samples] at `sample_rate`."""

    def embed_wav(self, wav: torch.Tensor) -> np.ndarray:
        windows = split_windows(wav, int(self.window_seconds * self.sample_rate))
        parts = [self.embed_windows(windows[i : i + self.batch_size]) for i in range(0, len(windows), self.batch_size)]
        return aggregate(np.concatenate(parts))

    def embed_file(self, path: Path) -> np.ndarray:
        return self.embed_wav(load_audio(path, self.sample_rate))


class ClapEmbedder(Embedder):
    """LAION-CLAP music checkpoint (HTSAT-base, 512-d), 10 s windows at 48 kHz."""

    name = "clap"
    sample_rate = 48_000
    window_seconds = 10.0
    batch_size = 16

    def __init__(self, checkpoint: Path = CLAP_CHECKPOINT, device: str | None = None) -> None:
        import laion_clap

        if not checkpoint.exists():
            raise FileNotFoundError(f"LAION-CLAP checkpoint not found: {checkpoint} (see docs/embeddings.md)")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=self.device)
        self.model.load_ckpt(str(checkpoint))
        self.model.eval()

    @torch.inference_mode()
    def embed_windows(self, windows: torch.Tensor) -> np.ndarray:
        return self.model.get_audio_embedding_from_data(x=windows.float().numpy(), use_tensor=False)


class MuLanEmbedder(Embedder):
    """MuQ-MuLan audio tower (512-d), 30 s windows at 24 kHz."""

    name = "mulan"
    sample_rate = 24_000
    window_seconds = 30.0
    batch_size = 4

    def __init__(self, repo: str = MULAN_REPO, device: str | None = None) -> None:
        from muq import MuQMuLan

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuQMuLan.from_pretrained(repo).to(self.device).eval()

    @torch.inference_mode()
    def embed_windows(self, windows: torch.Tensor) -> np.ndarray:
        return self.model(wavs=windows.to(self.device)).float().cpu().numpy()


EMBEDDERS: dict[str, Callable[[], Embedder]] = {"clap": ClapEmbedder, "mulan": MuLanEmbedder}


FAILED_FIELDS = ["song_id", "path", "reason"]


class FileEmbedder(Protocol):
    def embed_file(self, path: Path) -> np.ndarray: ...


def extract(
    items: list[tuple[str, Path]], output_dir: Path, embedder: FileEmbedder, collect: bool = True
) -> tuple[int, int]:
    """Embed every (song_id, audio path) into <output_dir>/vectors/<song_id>.npy and rebuild embeddings.npz.

    Songs with an existing vector file are skipped, so an interrupted run continues. Songs that fail are listed in
    failed.csv (the failures of the last run). `collect=False` skips embeddings.npz (for embeddings that are not one
    vector per song, e.g. CLEWS shingle sequences). Returns (newly embedded, failed).
    """
    from encoder_exorcist.data.lyrics_dataset import save_records

    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    failures: list[dict] = []
    for i, (song_id, path) in enumerate(items, 1):
        target = vectors_dir / f"{song_id}.npy"
        if target.exists():
            continue
        try:
            np.save(target, embedder.embed_file(path))
        except Exception as error:  # one broken file should not stop a multi-hour run
            logger.exception("failed %s", song_id)
            failures.append({"song_id": song_id, "path": str(path), "reason": f"{type(error).__name__}: {error}"})
            failed += 1
            continue
        done += 1
        logger.info("[%d/%d] %s", i, len(items), song_id)
    save_records(output_dir / "failed.csv", FAILED_FIELDS, failures)

    ids = [song_id for song_id, _ in items if (vectors_dir / f"{song_id}.npy").exists()]
    if ids and collect:
        np.savez(
            output_dir / "embeddings.npz",
            song_ids=np.array(ids),
            embeddings=np.stack([np.load(vectors_dir / f"{song_id}.npy") for song_id in ids]),
        )
    return done, failed
