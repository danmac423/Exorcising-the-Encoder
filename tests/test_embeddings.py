import numpy as np
import torch

from encoder_exorcist.embeddings import Embedder, aggregate, extract, split_windows


class FakeEmbedder(Embedder):
    name = "fake"
    sample_rate = 10
    window_seconds = 2.0
    batch_size = 2

    def __init__(self):
        self.calls = 0

    def embed_windows(self, windows):
        self.calls += 1
        return np.stack([[w.mean().item(), 1.0] for w in windows])

    def embed_file(self, path):
        return self.embed_wav(torch.ones(50))


def test_split_windows_drops_short_tail_and_pads_long_one():
    assert split_windows(torch.ones(45), 20).shape == (2, 20)  # tail of 5 < half a window
    windows = split_windows(torch.ones(55), 20)  # tail of 15 is padded
    assert windows.shape == (3, 20)
    assert windows[-1].sum() == 15


def test_split_windows_short_audio_gives_one_window():
    assert split_windows(torch.ones(3), 20).shape == (1, 20)


def test_aggregate_is_unit_norm_mean():
    vec = aggregate(np.array([[3.0, 0.0], [3.0, 4.0]]))
    assert np.isclose(np.linalg.norm(vec), 1.0)
    assert np.allclose(vec, [3 / np.hypot(3, 2), 2 / np.hypot(3, 2)])


def test_extract_resumes_and_collects_failures(tmp_path):
    class Flaky(FakeEmbedder):
        def embed_file(self, path):
            if path.name == "bad":
                raise RuntimeError("broken")
            return super().embed_file(path)

    items = [("a", tmp_path / "a"), ("bad", tmp_path / "bad"), ("b", tmp_path / "b")]
    embedder = Flaky()
    assert extract(items, tmp_path / "out", embedder) == (2, 1)
    with np.load(tmp_path / "out" / "embeddings.npz") as data:
        assert list(data["song_ids"]) == ["a", "b"]
        assert data["embeddings"].shape == (2, 2)
    assert "broken" in (tmp_path / "out" / "failed.csv").read_text()

    calls = embedder.calls
    assert extract(items[:1] + items[2:], tmp_path / "out", embedder) == (0, 0)
    assert embedder.calls == calls
