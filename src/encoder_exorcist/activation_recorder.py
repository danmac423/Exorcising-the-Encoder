"""Mean-pooled activation recording for ACE-Step 1.5, following "Ghost in the Decoder" (Sec. 3.2, 3.3).

Hook points, shapes and decisions are described in docs/hook_points.md.

Usage:
    recorder = ActivationRecorder()
    with recorder.attach(dit_handler.model):
        generate(...)
    record = recorder.pop_song()
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)

# DiT layers hooked, as in the paper: 5 equally-spaced (sliding, full) layer pairs out of 12.
DIT_LAYERS = (0, 1, 6, 7, 12, 13, 16, 17, 22, 23)

# Order of the second axis of record["dit"].
DIT_POINTS = ("pre_xattn", "xattn_out", "post_xattn")

STREAM_KEYS = ("lyric_out", "text_out", "timbre_out", "merged_out", "cond_embed_out")


def token_mean(x: torch.Tensor) -> np.ndarray:
    """Mean over the token axis of a [1, T, d] tensor, in float32: `a = (1/T) * sum_t x_t` (paper, Sec. 3.3)."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"expected activations of shape [1, T, d], got {tuple(x.shape)}")
    return x.detach().float()[0].mean(dim=0).cpu().numpy()


def _first(output: Any) -> torch.Tensor:
    """Tensor out of a module output: plain tensor, tuple (hidden states first) or HF ModelOutput."""
    if isinstance(output, tuple):
        return output[0]
    return getattr(output, "last_hidden_state", output)


class ActivationRecorder:
    """Registers hooks on an ACE-Step model and keeps one mean-pooled vector per hook point and call."""

    def __init__(self) -> None:
        """Create an empty recorder. Nothing is hooked until `attach`; the hooked DiT layers are DIT_LAYERS."""
        self._handles: list[RemovableHandle] = []
        self._attached = False
        self.reset()

    def reset(self) -> None:
        """Discard everything recorded so far: once-per-song vectors and per-step DiT vectors."""
        # Hook points that fire once per song (or whose repeats are identical): the first call wins.
        self._once: dict[str, np.ndarray] = {}
        self._repeated: set[str] = set()
        # DiT audio-latent points fire once per denoising step.
        self._dit: dict[tuple[int, str], list[np.ndarray]] = {}

    @contextmanager
    def attach(self, model: Any) -> Generator["ActivationRecorder"]:
        """Context manager: hook `model` (needs `.encoder` and `.decoder`) for the duration of the `with` block.

        Starts from a clean state and removes every hook on exit, also when the block raises.
        Raises RuntimeError if the recorder is already attached.
        """
        if self._attached:
            raise RuntimeError("recorder is already attached")
        self.reset()
        try:
            self._register(model)
            self._attached = True
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._attached = False

    def pop_song(self) -> dict[str, np.ndarray]:
        """Return the record of the song generated since attach/last pop and reset the state."""
        try:
            return self._build_record()
        finally:
            self.reset()

    def _record_once(self, key: str, vec: np.ndarray, warn_repeat: bool = True) -> None:
        """Store `vec` under `key` unless that key already has one (the first call wins).

        A repeated call is remembered for a warning in `_build_record` when `warn_repeat` is set; hook points whose
        repeats are expected and identical (K/V, condition_embedder) pass False.
        """
        if key in self._once:
            if warn_repeat:
                self._repeated.add(key)
            return
        self._once[key] = vec

    def _register(self, model: Any) -> None:
        """Register every hook listed in docs/hook_points.md and keep their handles for removal.

        Raises ValueError if a requested DiT layer does not exist or has no cross-attention.
        """
        encoder: Any = model.encoder
        decoder: Any = model.decoder
        lyric_encoder: Any = encoder.lyric_encoder
        n_dit = len(decoder.layers)
        for i in DIT_LAYERS:
            if not 0 <= i < n_dit:
                raise ValueError(f"DiT layer {i} out of range (model has {n_dit} layers)")
            if not hasattr(decoder.layers[i], "cross_attn"):
                raise ValueError(f"DiT layer {i} has no cross-attention")

        def add(handle: RemovableHandle) -> None:
            """Remember a hook handle so `attach` can remove it."""
            self._handles.append(handle)

        def encoder_post(module: Any, args: tuple, output: tuple) -> None:
            """Record `merged_out`, the conditioning sequence handed to the DiT.

            Also checks that the sequence has no padding. With one song per call every token is valid (the tokenizer
            pads to the longest, the silence reference counts as one timbre token), and the DiT ignores its attention
            masks anyway, so averaging over all tokens is exactly the paper's formula. If a mask has zeros, that
            assumption is broken and a plain mean would no longer match what we intend to measure.
            """
            states, mask = output
            if not bool(mask.all()):
                raise ValueError("conditioning sequence contains padded tokens; token_mean assumes none")
            self._record_once("merged_out", token_mean(states))

        def lyric_post(module: Any, args: tuple, output: Any) -> None:
            """Record `lyric_out`, the lyric encoder output (after its final norm)."""
            self._record_once("lyric_out", token_mean(_first(output)))

        def text_post(module: Any, args: tuple, output: torch.Tensor) -> None:
            """Record `text_out`, the projected caption embeddings."""
            self._record_once("text_out", token_mean(output))

        def timbre_post(module: Any, args: tuple, output: tuple) -> None:
            """Record `timbre_out`, the timbre stream: one token, the embedding of the silence reference."""
            self._record_once("timbre_out", token_mean(output[0]))

        # condition_embedder runs at every denoising step on the same input; keep the first output.
        def cond_embed_post(module: Any, args: tuple, output: torch.Tensor) -> None:
            """Record `cond_embed_out`, the conditioning sequence as the DiT sees it (after condition_embedder)."""
            self._record_once("cond_embed_out", token_mean(output), warn_repeat=False)

        add(encoder.register_forward_hook(encoder_post))
        add(lyric_encoder.register_forward_hook(lyric_post))
        add(encoder.text_projector.register_forward_hook(text_post))
        add(encoder.timbre_encoder.register_forward_hook(timbre_post))
        add(decoder.condition_embedder.register_forward_hook(cond_embed_post))

        for i, layer in enumerate(lyric_encoder.layers):
            add(layer.register_forward_hook(self._lyric_layer_hook(i)))

        for i in DIT_LAYERS:
            layer = decoder.layers[i]
            add(layer.cross_attn.k_proj.register_forward_hook(self._kv_hook(f"k_{i}")))
            add(layer.cross_attn.v_proj.register_forward_hook(self._kv_hook(f"v_{i}")))
            add(layer.cross_attn_norm.register_forward_pre_hook(self._dit_pre_hook(i, "pre_xattn")))
            add(layer.cross_attn.register_forward_hook(self._dit_post_hook(i, "xattn_out")))
            add(layer.mlp_norm.register_forward_pre_hook(self._dit_pre_hook(i, "post_xattn")))

    def _lyric_layer_hook(self, i: int):
        """Build the hook for lyric encoder layer `i` (a factory, so each hook keeps its own `i`)."""

        def hook(module: Any, args: tuple, output: Any) -> None:
            """Record `lyric_L{i}`: the layer output (before the encoder's final norm)."""
            self._record_once(f"lyric_L{i}", token_mean(_first(output)))

        return hook

    def _kv_hook(self, key: str):
        """Build the hook for a `k_proj` or `v_proj` output, stored as `key` (e.g. "k_12")."""

        # With the cross-attention cache K/V fire once per layer; without it they repeat with identical values.
        def hook(module: Any, args: tuple, output: torch.Tensor) -> None:
            """Record the projection before k_norm and head reshape."""
            self._record_once(key, token_mean(output), warn_repeat=False)

        return hook

    def _dit_pre_hook(self, layer: int, point: str):
        """Build a forward pre-hook that records the module's input for DiT `layer` under `point`."""

        def hook(module: Any, args: tuple) -> None:
            """Append the mean of the audio-latent input; fires once per denoising step, so one vector per step."""
            self._dit.setdefault((layer, point), []).append(token_mean(args[0]))

        return hook

    def _dit_post_hook(self, layer: int, point: str):
        """Build a forward hook that records the module's output for DiT `layer` under `point`."""

        def hook(module: Any, args: tuple, output: Any) -> None:
            """Append the mean of the audio-latent output; fires once per denoising step, so one vector per step."""
            self._dit.setdefault((layer, point), []).append(token_mean(_first(output)))

        return hook

    def _build_record(self) -> dict[str, np.ndarray]:
        """Assemble the per-song record described in docs/hook_points.md, section 5.

        Keys: the five stream vectors [d], lyric_layers [n_layers, d], kv [n_dit, 2, d_kv], dit [n_dit, 3, steps, d]
        (points in DIT_POINTS order) and dit_layers. Raises RuntimeError if a hook point never fired or the DiT points
        fired a different number of times; logs a warning if a once-per-song point fired more than once.
        """
        lyric_keys = sorted((k for k in self._once if k.startswith("lyric_L")), key=lambda k: int(k[len("lyric_L") :]))
        expected = [*STREAM_KEYS, *[f"{kv}_{i}" for i in DIT_LAYERS for kv in "kv"]]
        missing = [k for k in expected if k not in self._once]
        missing += [f"{i}:{p}" for i in DIT_LAYERS for p in DIT_POINTS if (i, p) not in self._dit]
        if not lyric_keys:
            missing.append("lyric_L*")
        if missing:
            raise RuntimeError(f"hook points never fired: {missing}")

        n_steps = {len(v) for v in self._dit.values()}
        if len(n_steps) != 1:
            counts = {f"{i}:{p}": len(v) for (i, p), v in self._dit.items()}
            raise RuntimeError(f"DiT hook points fired a different number of times: {counts}")
        if self._repeated:
            logger.warning("hook points fired more than once, only the first call was kept: %s", sorted(self._repeated))

        record = {key: self._once[key] for key in STREAM_KEYS}
        record["lyric_layers"] = np.stack([self._once[k] for k in lyric_keys])
        record["kv"] = np.stack([[self._once[f"k_{i}"], self._once[f"v_{i}"]] for i in DIT_LAYERS])
        record["dit"] = np.stack([[np.stack(self._dit[(i, p)]) for p in DIT_POINTS] for i in DIT_LAYERS])
        record["dit_layers"] = np.array(DIT_LAYERS)
        return record


def save_song(path: str | Path, record: dict[str, np.ndarray], config_name: str) -> None:
    """Write a record from pop_song() as .npz; config_name is e.g. "no-cot" or "with-cot"."""
    arrays: dict[str, Any] = {**record, "config_name": np.array(config_name)}
    np.savez(path, **arrays)
