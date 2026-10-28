import numpy as np
import pytest
import torch
from torch import nn

from encoder_exorcist.activation_recorder import DIT_LAYERS, ActivationRecorder, save_song, token_mean

D = 8
KV = 4
N_STEPS = 3
T_LYR, T_TXT, T_AUD = 5, 3, 6
N_LYRIC_LAYERS = 2
N_DIT_LAYERS = 24


class FakeLyricEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(D, D) for _ in range(N_LYRIC_LAYERS)])

    def forward(self, inputs_embeds, attention_mask=None):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return type("Out", (), {"last_hidden_state": x})()


class FakeTimbreEncoder(nn.Module):
    def forward(self, packed, order_mask):
        return torch.zeros(1, 1, D), torch.ones(1, 1, dtype=torch.long)


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_projector = nn.Linear(D, D, bias=False)
        self.lyric_encoder = FakeLyricEncoder()
        self.timbre_encoder = FakeTimbreEncoder()

    def forward(self, text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask):
        text = self.text_projector(text_hidden_states)
        lyric = self.lyric_encoder(inputs_embeds=lyric_hidden_states, attention_mask=lyric_attention_mask)
        timbre, timbre_mask = self.timbre_encoder(None, None)
        states = torch.cat([lyric.last_hidden_state, timbre, text], dim=1)
        mask = torch.cat([lyric_attention_mask, timbre_mask, text_attention_mask], dim=1)
        return states, mask


class FakeCrossAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, KV)
        self.v_proj = nn.Linear(D, KV)
        self.o = nn.Linear(KV, D)

    def forward(self, hidden_states, encoder_hidden_states):
        k, v = self.k_proj(encoder_hidden_states), self.v_proj(encoder_hidden_states)
        return self.o(k.mean(1, keepdim=True) + v.mean(1, keepdim=True)).expand_as(hidden_states) + self.q(
            hidden_states
        ), None


class FakeDiTLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(D)
        self.cross_attn = FakeCrossAttn()
        self.mlp_norm = nn.LayerNorm(D)

    def forward(self, hidden_states, encoder_hidden_states):
        out, _ = self.cross_attn(self.cross_attn_norm(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + out
        return hidden_states + self.mlp_norm(hidden_states)


class FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition_embedder = nn.Linear(D, D)
        self.layers = nn.ModuleList([FakeDiTLayer() for _ in range(N_DIT_LAYERS)])

    def forward(self, hidden_states, encoder_hidden_states):
        for _ in range(N_STEPS):
            cond = self.condition_embedder(encoder_hidden_states)
            x = hidden_states
            for layer in self.layers:
                x = layer(x, cond)
        return x


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.decoder = FakeDecoder()

    def generate(self, lyric_mask: torch.Tensor):
        text = torch.randn(1, T_TXT, D)
        lyric = torch.randn(1, T_LYR, D)
        text_mask = torch.ones(1, T_TXT, dtype=torch.long)
        # Called with kwargs, like AceStepConditionGenerationModel.prepare_condition.
        states, _ = self.encoder(
            text_hidden_states=text,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric,
            lyric_attention_mask=lyric_mask,
        )
        return self.decoder(torch.randn(1, T_AUD, D), states)


LYRIC_MASK = torch.ones(1, T_LYR, dtype=torch.long)
LAYERS = DIT_LAYERS


@pytest.fixture
def model() -> FakeModel:
    torch.manual_seed(0)
    return FakeModel().eval()


def record_song(model: FakeModel, recorder: ActivationRecorder) -> dict[str, np.ndarray]:
    with torch.no_grad(), recorder.attach(model):
        model.generate(LYRIC_MASK)
        return recorder.pop_song()


def test_token_mean_is_the_papers_formula() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]]])
    np.testing.assert_allclose(token_mean(x), [3.0, 5.0])


def test_token_mean_is_float32_and_rejects_batches() -> None:
    assert token_mean(torch.ones(1, 2, 3, dtype=torch.bfloat16)).dtype == np.float32
    with pytest.raises(ValueError):
        token_mean(torch.ones(2, 2, 3))


def test_record_shapes(model: FakeModel) -> None:
    record = record_song(model, ActivationRecorder())
    for key in ("lyric_out", "text_out", "timbre_out", "merged_out", "cond_embed_out"):
        assert record[key].shape == (D,)
    assert record["lyric_layers"].shape == (N_LYRIC_LAYERS, D)
    assert record["kv"].shape == (len(LAYERS), 2, KV)
    assert record["dit"].shape == (len(LAYERS), 3, N_STEPS, D)
    assert record["dit_layers"].tolist() == list(LAYERS)


def test_lyric_stream_is_mean_over_all_tokens(model: FakeModel) -> None:
    captured = {}
    model.encoder.lyric_encoder.register_forward_hook(lambda m, a, out: captured.update(x=out.last_hidden_state))
    record = record_song(model, ActivationRecorder())
    np.testing.assert_allclose(record["lyric_out"], captured["x"][0].mean(0).numpy(), rtol=1e-5)


def test_merged_stream_and_kv_are_means_over_all_tokens(model: FakeModel) -> None:
    captured = {}
    model.encoder.register_forward_hook(lambda m, a, out: captured.update(states=out[0]))
    record = record_song(model, ActivationRecorder())
    states = captured["states"]
    np.testing.assert_allclose(record["merged_out"], states[0].mean(0).numpy(), rtol=1e-5)

    cond = model.decoder.condition_embedder(states)
    k = model.decoder.layers[LAYERS[0]].cross_attn.k_proj(cond)[0].mean(0)
    np.testing.assert_allclose(record["kv"][0, 0], k.detach().numpy(), rtol=1e-4, atol=1e-6)


def test_padded_conditioning_sequence_is_rejected(model: FakeModel) -> None:
    padded = torch.tensor([[1, 1, 1, 0, 0]])
    with pytest.raises(ValueError, match="padded"), torch.no_grad(), ActivationRecorder().attach(model):
        model.generate(padded)


def test_cond_embed_is_affine_image_of_merged(model: FakeModel) -> None:
    record = record_song(model, ActivationRecorder())
    embedder = model.decoder.condition_embedder
    expected = embedder(torch.from_numpy(record["merged_out"])).detach().numpy()
    np.testing.assert_allclose(record["cond_embed_out"], expected, rtol=1e-4, atol=1e-5)


def test_dit_points_match_layer_tensors(model: FakeModel) -> None:
    captured = {"pre": [], "post": []}
    layer = model.decoder.layers[LAYERS[2]]
    layer.cross_attn_norm.register_forward_pre_hook(lambda m, a: captured["pre"].append(a[0].mean(1)[0]))
    layer.mlp_norm.register_forward_pre_hook(lambda m, a: captured["post"].append(a[0].mean(1)[0]))
    record = record_song(model, ActivationRecorder())
    layer_idx = 2
    np.testing.assert_allclose(record["dit"][layer_idx, 0], torch.stack(captured["pre"]).numpy(), rtol=1e-5)
    np.testing.assert_allclose(record["dit"][layer_idx, 2], torch.stack(captured["post"]).numpy(), rtol=1e-5)


def test_pop_song_resets_state(model: FakeModel) -> None:
    recorder = ActivationRecorder()
    with torch.no_grad(), recorder.attach(model):
        model.generate(LYRIC_MASK)
        first = recorder.pop_song()
        model.generate(LYRIC_MASK)
        second = recorder.pop_song()
    assert first["dit"].shape == second["dit"].shape == (len(LAYERS), 3, N_STEPS, D)
    assert not np.allclose(first["merged_out"], second["merged_out"])


def test_hooks_are_removed_on_exit(model: FakeModel) -> None:
    recorder = ActivationRecorder()
    with torch.no_grad(), recorder.attach(model):
        pass
    with torch.no_grad():
        model.generate(LYRIC_MASK)
    with pytest.raises(RuntimeError, match="never fired"):
        recorder.pop_song()


def test_pop_song_without_generation_raises(model: FakeModel) -> None:
    recorder = ActivationRecorder()
    with recorder.attach(model), pytest.raises(RuntimeError, match="never fired"):
        recorder.pop_song()


def test_no_nested_attach(model: FakeModel) -> None:
    recorder = ActivationRecorder()
    with recorder.attach(model), pytest.raises(RuntimeError), recorder.attach(model):
        pass


def test_model_with_too_few_dit_layers(model: FakeModel) -> None:
    model.decoder.layers = model.decoder.layers[:4]
    with pytest.raises(ValueError, match="out of range"), ActivationRecorder().attach(model):
        pass


def test_repeated_encoder_call_keeps_first(model: FakeModel, caplog: pytest.LogCaptureFixture) -> None:
    recorder = ActivationRecorder()
    with torch.no_grad(), recorder.attach(model):
        model.generate(LYRIC_MASK)
        first_lyric = recorder._once["lyric_out"].copy()
        model.generate(LYRIC_MASK)
        assert np.array_equal(recorder._once["lyric_out"], first_lyric)
        with caplog.at_level("WARNING"):
            recorder.pop_song()
    assert "different value" in caplog.text


def test_identical_repeated_encoder_call_is_silent(model: FakeModel, caplog: pytest.LogCaptureFixture) -> None:
    recorder = ActivationRecorder()
    with torch.no_grad(), recorder.attach(model):
        torch.manual_seed(1)
        model.generate(LYRIC_MASK)
        torch.manual_seed(1)
        model.generate(LYRIC_MASK)
        with caplog.at_level("WARNING"):
            recorder.pop_song()
    assert caplog.text == ""


def test_save_song_roundtrip(model: FakeModel, tmp_path) -> None:
    record = record_song(model, ActivationRecorder())
    path = tmp_path / "song.npz"
    save_song(path, record, "no-cot")
    with np.load(path) as data:
        assert str(data["config_name"]) == "no-cot"
        np.testing.assert_array_equal(data["dit"], record["dit"])
