# Hook points for activation extraction (ACE-Step 1.5 turbo)

Goal: reproduce the activation extraction of *Ghost in the Decoder* (Sec. 3.2 and 3.3) with PyTorch hooks.
Sources: the paper, `checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py` and `config.json`.
Shapes below are derived from code and config, **not yet measured** (see "Verification").

Notation: `m = dit_handler.model`, `B` = batch (1), `T_lyr` = lyric tokens, `T_txt` = caption tokens,
`T_cond = T_lyr + T_timbre + T_txt` (merged sequence), `T_aud = 2250` (180 s at 25 Hz, patch size 2; the 25 Hz rate is inferred from `lm_hints_25Hz`).

## 1. Averaging rule (paper, Sec. 3.3)

For every hook point `h`: `a_i = mean(H_h(x_i)) in R^{d_h}`, the mean over **all axes except the last**.
For `H in R^{T x d}`: `a_i = (1/T) * sum_t H_t`.

Implemented in the hook as a plain float32 mean over the token axis (`token_mean`), exactly the paper's formula. No masks:

- With one song per call every token is valid: the tokenizer pads to the longest sequence (`padding="longest"`), and with no reference audio the handler feeds 750 frames of `silence_latent`, which the timbre encoder turns into one valid token (`refer_audio_order_mask = [0]`). So the lyric, text, timbre and merged masks are all ones.
- `AceStepDiTModel.forward` overwrites its `attention_mask` and `encoder_attention_mask` arguments with `None` (lines 1388-1389), so cross-attention attends to every conditioning token regardless of any mask.
- Guard: the `m.encoder` hook raises `ValueError` if the merged mask contains a zero (it is all ones iff every lyric, timbre and text token is valid). This would signal a code path with padding, e.g. an unexpected with-CoT behaviour.
- Audio-latent states: all `T_aud` tokens are used (4500 latent frames is even, no patch padding).
- DiT audio-latent hooks fire once per denoising step. The step index is the per-hook call counter (8 steps for turbo). Reset all counters at the start of every song.
- K/V hooks fire only at step 0: `AceStepAttention` fills the cross-attention cache once (`is_updated`). Do not expect 8 calls.
- Save per song, float32, one `.npz`. **Only averages are stored** (decision): about 2 MB/song instead of about 2.3 GB/song in bf16. Changing the averaging variant later means regenerating.

## 2. Masks

Not used for averaging (see section 1). The only mask handling is the guard on `merged_out`.

## 3. Hook list (62 hooks)

### A. Conditioning stream outputs (4, forward hook)

| Name | Module | Take | Shape |
|---|---|---|---|
| `lyric_out` | `m.encoder.lyric_encoder` | `out.last_hidden_state` (after final `norm`) | `[B,T_lyr,2048]` |
| `text_out` | `m.encoder.text_projector` | `out` | `[B,T_txt,2048]` |
| `timbre_out` | `m.encoder.timbre_encoder` | `out[0]` (unused stream, expect near-chance; it is the silence reference) | `[B,1,2048]` |
| `merged_out` | `m.encoder` | `out[0]` | `[B,T_cond,2048]` |

### B. Lyric encoder layers (8, forward hook)

| Name | Module | Take | Shape |
|---|---|---|---|
| `lyric_L{i}`, i=0..7 | `m.encoder.lyric_encoder.layers[i]` | `out[0]` | `[B,T_lyr,2048]` |

These are layer **outputs**. HF `output_hidden_states` would instead give layer inputs plus the normed final state, so do not use it. `lyric_L7` is pre-`norm`, hence differs from `lyric_out`.

### C. DiT cross-attention (5 per selected layer, 10 layers = 50)

For `L = m.decoder.layers[i]`:

| Name | Module | Hook | Take | Shape | Calls/song |
|---|---|---|---|---|---|
| `k` | `L.cross_attn.k_proj` | forward | `out` | `[B,T_cond,1024]` | 1 |
| `v` | `L.cross_attn.v_proj` | forward | `out` | `[B,T_cond,1024]` | 1 |
| `pre_xattn` | `L.cross_attn_norm` | forward **pre** | `args[0]` | `[B,T_aud,2048]` | 8 |
| `xattn_out` | `L.cross_attn` | forward | `out[0]` | `[B,T_aud,2048]` | 8 |
| `post_xattn` | `L.mlp_norm` | forward **pre** | `args[0]` | `[B,T_aud,2048]` | 8 |

- `k`/`v` dim 1024 = `num_key_value_heads * head_dim` = 8 x 128. Hooking `k_proj` gives the value before `k_norm` and the head reshape.
- `pre_xattn` is the state after self-attention, before `cross_attn_norm`.
- `post_xattn` is `hidden_states + cross_attn_out`, i.e. exactly the input of `mlp_norm` (no module outputs this tensor).

### Which 10 layers (fixed constant)

The paper says "5 equally-spaced pairs of DiT layers (10 layers total)" without indices. Layers alternate sliding/full attention (`layer_types`), so a pair is `(2k, 2k+1)` over 12 pairs. Working guess: `linspace` over pairs, i.e. pairs 0, 3, 6, 8, 11 = layers **0-1, 6-7, 12-13, 16-17, 22-23** (consistent with Figure 3 mentioning layers 0-1 and 12-13). **Decision:** the layers are a fixed constant (`DIT_LAYERS` in `activation_recorder.py`), not a parameter. If the paper's indices differ from this guess, change that one constant.

## 4. Encoder output vs. decoder input

Question: save the output of the condition encoder, or the input of the decoder?

- The decoder does not take the encoder output directly. `AceStepDiTModel.forward` first applies `condition_embedder` (`nn.Linear(2048, 2048, bias=True)`) to `encoder_hidden_states`, and K/V are computed from that projected sequence.
- The paper's "merged conditioning sequence passed to the DiT" is read as the **encoder output** (`merged_out`), before `condition_embedder`.
- Since `condition_embedder` is affine, `mean(Wx + b) = W mean(x) + b`. The decoder-input mean is therefore recoverable from the encoder-output mean, but a probe with standardization or regularization is not invariant to that map.

Decision: use `merged_out` (matches the paper, keeps results comparable) and additionally record `cond_embed_out` (forward hook on `m.decoder.condition_embedder`, shape `[B,T_cond,2048]`). It costs 2048 floats per song and lets us compare both views.

## 5. Per-song output layout

| Key | Shape |
|---|---|
| `lyric_out`, `text_out`, `timbre_out`, `merged_out`, `cond_embed_out` | `[2048]` |
| `lyric_layers` | `[8, 2048]` |
| `kv` | `[10, 2, 1024]` (layer, k/v) |
| `dit` | `[10, 3, 8, 2048]` (layer, pre/xattn/post, step) |

About 0.54 M float32 values, ~2 MB per song; ~4.3 GB for 2000 songs per configuration (no-CoT, with-CoT).

## 6. Scope and code structure (decisions)

- **Configurations:** both no-CoT and with-CoT from the first version (toggle the 5Hz LM; `text_out` is only meaningful with CoT). Store the configuration name in the output file.
- **API:** class `ActivationRecorder` in `src/encoder_exorcist/`. Used as a context manager: `with recorder.attach(model): generate(...)`, then `recorder.pop_song()` returns the dict from section 5 and resets all counters. Must be testable without GPU on a tiny model with the same module names.
- **Out of scope for v1:** saving full tensors (see section 1).

## 7. Verification checklist (run once before the big run)

Register shape-recording hooks, generate one 180 s song with `bpm=None, key=None, text_caption=None, timbre_reference=None`, and assert:

- last dim is 2048 everywhere except `k`/`v` (1024);
- `k`/`v` fire once per layer; `pre_xattn`/`xattn_out`/`post_xattn` fire 8 times per layer;
- `T_aud == 2250`; if not, the 25 Hz assumption is wrong;
- `lyric_out`, `merged_out` fire twice on the real model (the handler pre-computes the conditioning, then `generate_audio` recomputes it) with identical output; the recorder keeps the first call and only warns if a repeat differs;
- `T_cond == T_lyr + T_timbre + T_txt`, and the merged mask is all ones (the guard in `merged_out` enforces it; check it in with-CoT too);
- `AceStepHandler` generation reaches the same denoising loop as `generate_audio` (not verified).

## 8. Not specified by the paper

Number of CV folds, feature standardization, regularization of the logistic regression, and similar choices. We fix these ourselves and document them with the results.
