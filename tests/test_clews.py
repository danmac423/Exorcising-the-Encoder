import numpy as np
import torch

from encoder_exorcist.clews import distance_matrix, ranks_of_targets, retrieval_summary


class FakeModel(torch.nn.Module):
    """Same `distances` contract as CLEWS: mean over valid shingle pairs of the normalized squared distance."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def distances(self, q, c, qmask: torch.Tensor, cmask: torch.Tensor, redux_strategy=None):
        d = ((q[:, None, :, None, :] - c[None, :, None, :, :]) ** 2).mean(-1)  # (bq, bc, sq, sc)
        valid = ~(qmask[:, None, :, None] | cmask[None, :, None, :])
        return (d * valid).sum((-1, -2)) / valid.sum((-1, -2))


def test_distance_matrix_ignores_padding_of_shorter_candidates():
    rng = np.random.default_rng(0)
    queries = [rng.normal(size=(3, 8)).astype(np.float32)]
    candidates = [rng.normal(size=(2, 8)).astype(np.float32), rng.normal(size=(5, 8)).astype(np.float32)]
    together = distance_matrix(FakeModel(), queries, candidates, batch=8)
    alone = np.concatenate([distance_matrix(FakeModel(), queries, [c]) for c in candidates], axis=1)
    assert together.shape == (1, 2)
    assert np.allclose(together, alone, atol=1e-5)  # padding in the batch changes nothing


def test_batches_give_the_same_matrix():
    rng = np.random.default_rng(1)
    queries = [rng.normal(size=(4, 8)).astype(np.float32) for _ in range(2)]
    candidates = [rng.normal(size=(n, 8)).astype(np.float32) for n in (2, 4, 3, 5, 4)]
    full = distance_matrix(FakeModel(), queries, candidates, batch=10)
    assert np.allclose(full, distance_matrix(FakeModel(), queries, candidates, batch=2), atol=1e-5)


def test_ranks_and_summary():
    distances = np.array([[0.1, 0.5, 0.9], [0.4, 0.2, 0.3], [0.5, 0.5, 0.5]])
    ranks = ranks_of_targets(distances, np.array([0, 2, 1]))
    assert ranks.tolist() == [1, 2, 1]  # a perfect tie counts as rank 1
    rows = {r["metric"]: r for r in retrieval_summary(ranks, n_gallery=3, ks=(1, 2))}
    assert rows["R@1"]["value"] == "0.6667" and rows["R@2"]["value"] == "1.0000"
    assert rows["R@1"]["chance"] == "0.3333"
    assert rows["MAP"]["value"] == f"{(1 + 0.5 + 1) / 3:.4f}"
    assert rows["mean_rank"]["chance"] == "2.0"
