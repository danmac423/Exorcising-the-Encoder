"""Song-level similarity between generated songs and the original recordings with CLEWS-DVI ("Ghost in the Encoder",
Sec. 3.4, Table 3).

CLEWS (Serra et al., ICML 2025, https://github.com/sony/clews, MIT) embeds a song as a sequence of shingle embeddings
(20 s windows) and compares two songs by the best matching shingle pairs (`redux`). The model code is not a package, so
it is imported from a clone of the repository (CLEWS_DIR), and the DVI checkpoint (Zenodo record 15045900) goes in a
folder next to a configuration.yaml (see docs/clews.md).

Retrieval@K: every generated song is a query, the gallery holds the original recordings, and the song counts as hit at
K when its own original is among the K closest gallery songs. The chance level is K / N for a gallery of N songs.

Choices the paper leaves open (CLEWS settings, docs/clews.md): shingle hop 5 s and redux "bpwr-10" (the values of the
CLEWS README example for test.py), audio cut at 10 minutes as in test.py, candidates of different lengths padded and
masked. Audio is loaded with torchaudio instead of CLEWS' soxr resampler.
"""

import importlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT
from encoder_exorcist.embeddings import load_audio

logger = logging.getLogger(__name__)

CLEWS_DIR = PROJECT_ROOT / "checkpoints" / "clews-code"
CLEWS_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "clews-dvi" / "checkpoint_best.ckpt"
SHINGLE_HOP = 5.0  # seconds
MAX_SECONDS = 600
REDUX = "bpwr-10"
KS = (10, 25, 50, 100, 200, 400)  # as in Table 3
CANDIDATE_BATCH = 128


def load_model(checkpoint: Path = CLEWS_CHECKPOINT, clews_dir: Path = CLEWS_DIR, device: str | None = None):
    """CLEWS model in eval mode. `checkpoint` must sit next to the configuration.yaml it was trained with."""
    from omegaconf import OmegaConf

    if not clews_dir.exists():
        raise FileNotFoundError(f"CLEWS code not found: {clews_dir} (git clone https://github.com/sony/clews there)")
    if not checkpoint.exists() or not (checkpoint.parent / "configuration.yaml").exists():
        raise FileNotFoundError(f"need {checkpoint} and configuration.yaml next to it (see docs/clews.md)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.load(checkpoint.parent / "configuration.yaml")
    sys.path.insert(0, str(clews_dir))  # the repo uses top-level packages named models, lib and utils
    try:
        module = importlib.import_module(f"models.{conf.model.name}")
    finally:
        sys.path.remove(str(clews_dir))
    model = module.Model(conf.model, sr=conf.data.samplerate)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)  # Fabric checkpoints keep the weights under "model"
    state = {k.removeprefix("_forward_module.").removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def embed_file(model, path: Path, hop: float = SHINGLE_HOP, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Shingle embeddings [S, C] of one song."""
    wav = load_audio(path, model.sr)[: int(max_seconds * model.sr)]
    device = next(model.parameters()).device
    return model(wav[None].to(device), shingle_hop=hop).squeeze(0).float().cpu().numpy()


@torch.inference_mode()
def distance_matrix(
    model,
    queries: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    redux: str = REDUX,
    batch: int = CANDIDATE_BATCH,
) -> np.ndarray:
    """CLEWS distances [n_queries, n_candidates] between songs given as shingle embeddings [S, C].

    Candidates of different lengths are zero-padded to the longest one in their batch and masked (True = padding).
    """
    device = next(model.parameters()).device
    out = np.empty((len(queries), len(candidates)), dtype=np.float32)
    for start in range(0, len(candidates), batch):
        chunk = candidates[start : start + batch]
        longest = max(len(c) for c in chunk)
        z = torch.zeros(len(chunk), longest, chunk[0].shape[1])
        mask = torch.ones(len(chunk), longest, dtype=torch.bool)
        for i, c in enumerate(chunk):
            z[i, : len(c)] = torch.from_numpy(c)
            mask[i, : len(c)] = False
        z, mask = z.to(device), mask.to(device)
        for q, query in enumerate(queries):
            qz = torch.from_numpy(query)[None].to(device)
            qmask = torch.zeros(1, len(query), dtype=torch.bool, device=device)
            dist = model.distances(qz, z, qmask=qmask, cmask=mask, redux_strategy=redux)
            out[q, start : start + len(chunk)] = dist.reshape(-1).cpu().numpy()
    return out


def ranks_of_targets(distances: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each query's own original among the candidates: 1 + the number of strictly closer candidates."""
    true = distances[np.arange(len(targets)), targets]
    return 1 + (distances < true[:, None]).sum(axis=1)


def retrieval_summary(ranks: np.ndarray, n_gallery: int, ks: Sequence[int] = KS) -> list[dict]:
    """Retrieval@K and the chance level K / N for every K (Table 3), plus the mean rank and MAP (one original per
    query, so AP = 1 / rank)."""
    rows = [
        {"metric": f"R@{k}", "value": f"{(ranks <= k).mean():.4f}", "chance": f"{min(k, n_gallery) / n_gallery:.4f}"}
        for k in ks
    ]
    rows.append({"metric": "MAP", "value": f"{(1 / ranks).mean():.4f}", "chance": ""})
    rows.append({"metric": "mean_rank", "value": f"{ranks.mean():.1f}", "chance": f"{(n_gallery + 1) / 2:.1f}"})
    return rows
