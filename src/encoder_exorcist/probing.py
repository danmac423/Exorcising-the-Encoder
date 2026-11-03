"""Linear probes for artist identity on the recorded activations, following "Ghost in the Encoder" (Sec. 3.3, 4).

For every hook point (docs/hook_points.md) a multinomial logistic regression predicts the artist from the
mean-pooled activation of a song. Protocol from the paper: 5-fold cross-validation, held-out accuracy, permutation test
(n=1000, artists shuffled across songs and folds drawn again per permutation) and 95% confidence intervals. Choices the paper leaves open (docs/hook_points.md, section 8) are fixed here:
  - features are standardized inside each fold, L2-regularized logistic regression with C=1;
  - folds are stratified by artist and grouped by song (all seeds of one song stay in the same fold), shuffled with a
    fixed random state, and the same folds are used for every hook point;
  - the 95% CI is the Wilson interval of the pooled out-of-fold accuracy;
  - a PCA rotation onto the training samples' span (all components kept) runs before the regression. For an
    L2-regularized linear model the optimum lies in that span, so the classifier is the same (up to the solver's
    tolerance), only fitted on n_train instead of 2048 dimensions, which makes the permutation test affordable.

Baselines from the paper: TF-IDF bag of words of the lyrics, and the mean Qwen3 token embedding of the lyric encoder
input (the lyrics before any contextual processing).

Feature names: the five stream outputs (lyric_out, text_out, timbre_out, merged_out, cond_embed_out), lyric_L{i},
k_{layer}, v_{layer}, dit_{layer}_{point}_t{step} and dit_{layer}_{point}_mean (mean over denoising steps).
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.stats import norm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from encoder_exorcist.activation_recorder import DIT_POINTS, STREAM_KEYS
from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT, load_records

logger = logging.getLogger(__name__)

N_FOLDS = 5
N_PERMUTATIONS = 1000
RANDOM_STATE = 0
QWEN3_DIR = PROJECT_ROOT / "checkpoints" / "Qwen3-Embedding-0.6B"
LYRICS_TEMPLATE = "# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"  # ACE-Step's lyric encoder input

# Points that get a permutation test by default: the stream outputs and the lyric encoder layers. timbre_out is left
# out, it is the same silence reference for every song (a control that must stay at chance).
DEFAULT_PERMUTED = (*(k for k in STREAM_KEYS if k != "timbre_out"), *[f"lyric_L{i}" for i in range(8)])

RESULT_FIELDS = [
    "point",
    "group",
    "n_songs",
    "n_artists",
    "chance",
    "accuracy",
    "fold_std",
    "ci_low",
    "ci_high",
    "p_value",
    "n_permutations",
]
PER_ARTIST_FIELDS = ["point", "artist", "n_songs", "accuracy"]


@dataclass
class ProbeData:
    """Songs of one generation run: labels, fold groups and one feature matrix per hook point."""

    song_ids: list[str]
    artists: np.ndarray  # [n_songs] artist name per song
    groups: np.ndarray  # [n_songs] song key (artist + title), identical for all seeds of a song
    lyrics: np.ndarray  # [n_songs] lyrics text (object array, "" when not found)
    features: dict[str, np.ndarray]  # point name -> [n_songs, d]


def point_group(name: str) -> str:
    """Coarse group of a feature name, for plotting and filtering: stream, lyric, kv, dit or baseline."""
    if name in STREAM_KEYS:
        return "stream"
    for prefix, group in (("lyric_L", "lyric"), ("k_", "kv"), ("v_", "kv"), ("dit_", "dit")):
        if name.startswith(prefix):
            return group
    return "baseline"


def record_features(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten one song's .npz record into {feature name: vector}."""
    features = {key: record[key] for key in STREAM_KEYS}
    for i, vec in enumerate(record["lyric_layers"]):
        features[f"lyric_L{i}"] = vec
    for j, layer in enumerate(record["dit_layers"]):
        features[f"k_{layer}"] = record["kv"][j, 0]
        features[f"v_{layer}"] = record["kv"][j, 1]
        for p, point in enumerate(DIT_POINTS):
            steps = record["dit"][j, p]
            features[f"dit_{layer}_{point}_mean"] = steps.mean(axis=0)
            for t, vec in enumerate(steps):
                features[f"dit_{layer}_{point}_t{t}"] = vec
    return features


def load_probe_data(run_dir: Path, lyrics_csvs: Sequence[Path] = ()) -> ProbeData:
    """Load every song in <run_dir>/manifest.csv; lyrics are looked up in `lyrics_csvs` by artist and title."""
    rows = load_records(run_dir / "manifest.csv")
    if not rows:
        raise ValueError(f"no songs in {run_dir / 'manifest.csv'}")
    lyrics_by_song = {(r["artist"], r["title"]): r["lyrics"] for path in lyrics_csvs for r in load_records(path)}

    columns: dict[str, list[np.ndarray]] = {}
    for row in rows:
        with np.load(run_dir / row["activations_path"]) as record:
            for name, vec in record_features(dict(record)).items():
                columns.setdefault(name, []).append(vec)
    return ProbeData(
        song_ids=[row["song_id"] for row in rows],
        artists=np.array([row["artist"] for row in rows]),
        groups=np.array([f"{row['artist']}|{row['title']}" for row in rows]),
        lyrics=np.array([lyrics_by_song.get((row["artist"], row["title"]), "") for row in rows], dtype=object),
        features={name: np.stack(vecs).astype(np.float32) for name, vecs in columns.items()},
    )


def qwen3_token_embeddings(lyrics: Sequence[str], model_dir: Path = QWEN3_DIR, language: str = "unknown") -> np.ndarray:
    """Mean Qwen3 token embedding of each song's lyric encoder input (no transformer forward, as in ACE-Step)."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with safe_open(str(model_dir / "model.safetensors"), "pt") as file:
        table = file.get_tensor("embed_tokens.weight").float().numpy()
    means = [
        table[tokenizer(LYRICS_TEMPLATE.format(language=language, lyrics=text))["input_ids"]].mean(axis=0)
        for text in lyrics
    ]
    return np.stack(means)


def activation_probe() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        PCA(svd_solver="full"),  # rotation onto the training span only, see the module docstring
        LogisticRegression(C=1.0, max_iter=2000),
    )


def tfidf_probe() -> Pipeline:
    return make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), LogisticRegression(C=10.0, max_iter=2000))


def make_folds(artists: np.ndarray, groups: np.ndarray, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds stratified by artist and grouped by song (the real-label folds are shared by every probe)."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(artists)), artists, groups))


def wilson_interval(correct: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = norm.ppf(0.5 + level / 2)
    p = correct / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def cross_val_predictions(probe: Pipeline, X: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions for every song and the accuracy of each fold."""
    predictions = np.empty(len(y), dtype=y.dtype)
    fold_accuracies = []
    for train, test in folds:
        with warnings.catch_warnings():
            # Constant features (timbre_out: the same silence reference for every song) make PCA divide by zero
            # variance when it computes explained_variance_ratio_; the projection itself is fine.
            warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
            predictions[test] = clone(probe).fit(X[train], y[train]).predict(X[test])
        fold_accuracies.append(float((predictions[test] == y[test]).mean()))
    return predictions, fold_accuracies


def permuted_accuracy(probe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    """CV accuracy with the artists shuffled across songs (all seeds of a song keep one label) and new folds.

    The folds are drawn again for the permuted labels. Reusing the real-label folds, as sklearn's
    permutation_test_score does, leaves the permuted classes unbalanced across folds, which pushes the null accuracy
    below chance and makes p-values too small (a constant feature at exactly chance came out at p=0.001).
    """
    rng = np.random.default_rng(seed)
    song_keys, song_index = np.unique(groups, return_inverse=True)
    song_labels = np.empty(len(song_keys), dtype=y.dtype)
    song_labels[song_index] = y
    y_perm = rng.permutation(song_labels)[song_index]
    predictions, _ = cross_val_predictions(probe, X, y_perm, make_folds(y_perm, groups))
    return float((predictions == y_perm).mean())


def permutation_p_value(
    probe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, score: float, n_permutations: int, n_jobs: int
) -> float:
    """One-sided permutation p-value of `score`: (1 + #null >= score) / (1 + n_permutations)."""
    null = Parallel(n_jobs=n_jobs)(
        delayed(permuted_accuracy)(probe, X, y, groups, RANDOM_STATE + 1 + i) for i in range(n_permutations)
    )
    return (1 + sum(s >= score for s in null)) / (1 + n_permutations)


def evaluate_point(
    name: str,
    probe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds,
    n_permutations: int = 0,
    n_jobs: int = 1,
) -> tuple[dict, list[dict]]:
    """Probe one feature set: summary row (RESULT_FIELDS) and per-artist accuracy rows (PER_ARTIST_FIELDS)."""
    predictions, fold_accuracies = cross_val_predictions(probe, X, y, folds)
    correct = predictions == y
    ci_low, ci_high = wilson_interval(int(correct.sum()), len(y))
    p_value = ""
    if n_permutations:
        p = permutation_p_value(probe, X, y, groups, float(correct.mean()), n_permutations, n_jobs)
        p_value = f"{p:.4g}"
    artists = np.unique(y)
    row = {
        "point": name,
        "group": point_group(name),
        "n_songs": len(y),
        "n_artists": len(artists),
        "chance": f"{1 / len(artists):.4f}",
        "accuracy": f"{correct.mean():.4f}",
        "fold_std": f"{np.std(fold_accuracies):.4f}",
        "ci_low": f"{ci_low:.4f}",
        "ci_high": f"{ci_high:.4f}",
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
    per_artist = [
        {"point": name, "artist": a, "n_songs": int((y == a).sum()), "accuracy": f"{correct[y == a].mean():.4f}"}
        for a in artists
    ]
    return row, per_artist


def run_probes(
    data: ProbeData,
    points: Sequence[str] | None = None,
    permuted: Sequence[str] = DEFAULT_PERMUTED,
    n_permutations: int = N_PERMUTATIONS,
    baselines: bool = True,
    n_jobs: int = -1,
) -> tuple[list[dict], list[dict]]:
    """Probe every activation point (or `points`) and the baselines; returns (summary rows, per-artist rows).

    Points in `permuted` get a permutation test with `n_permutations` permutations, the others only CV accuracy and CI.
    """
    names = list(points) if points is not None else list(data.features)
    unknown = sorted(set(names) - set(data.features))
    if unknown:
        raise ValueError(f"unknown hook points: {unknown}")
    y, groups = data.artists, data.groups
    folds = make_folds(y, groups)

    # One BLAS thread per worker: every fit is small, and nested threading oversubscribes the CPU (about 70x slower).
    with parallel_config(backend="loky", inner_max_num_threads=1):
        # Points without permutations are cheap: run them in parallel, one job per point.
        cheap = [n for n in names if n not in permuted]
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_point)(n, activation_probe(), data.features[n], y, groups, folds) for n in cheap
        )
        for name in (n for n in names if n in permuted):
            logger.info("permutation test (%d) for %s", n_permutations, name)
            X = data.features[name]
            results.append(evaluate_point(name, activation_probe(), X, y, groups, folds, n_permutations, n_jobs))

        if baselines and any(data.lyrics):
            logger.info("baselines: tfidf and qwen3_tokens")
            results.append(
                evaluate_point("tfidf", tfidf_probe(), data.lyrics, y, groups, folds, n_permutations, n_jobs)
            )
            qwen = qwen3_token_embeddings(list(data.lyrics))
            results.append(
                evaluate_point("qwen3_tokens", activation_probe(), qwen, y, groups, folds, n_permutations, n_jobs)
            )
        elif baselines:
            logger.warning("no lyrics found for the songs, skipping the baselines")

    rows = [row for row, _ in results]
    per_artist = [r for _, rows_ in results for r in rows_]
    order = {name: i for i, name in enumerate([*names, "tfidf", "qwen3_tokens"])}
    rows.sort(key=lambda r: order[r["point"]])
    return rows, per_artist
