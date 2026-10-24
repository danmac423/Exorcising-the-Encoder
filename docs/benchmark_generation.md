# ACE-Step 1.5 generation benchmark (RTX 5070 Ti)

*Date: 2026-10-24 · Author: Maksymilian Baj · Week 2 task: "Uruchomienie ACE-Step z udostępnionymi wagami, pomiar czasu generacji jednego utworu na naszym GPU"*

## TL;DR

- Generating one song on the RTX 5070 Ti takes **1.5–5 s without the 5Hz LM** and **7–35 s with it** (for songs of 30 s to 3 min).
- The full generation run (**2,500 songs**) takes **1–3.3 h without the LM** and **4.7–24 h with it**, depending on song length.
- When the LM is enabled it uses **about 85% of the time**. The diffusion model itself is cheap at roughly 0.04–0.16 s per step.
- Peak VRAM is **6.4 GB without the LM** and **11.1 GB with it**, out of 16 GB, so both fit without CPU offloading.
- **Decision needed:** should experiments run with the LM on or off? See [Implications](#implications-for-the-project).

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, sm_120) |
| Host | WSL2 (Linux 5.15), 31 GB RAM |
| Software | PyTorch 2.10.0+cu128, ACE-Step 1.5 @ `ca1e85f` (pinned in `pyproject.toml`) |
| DiT model | `acestep-v15-turbo`, 8 inference steps, bf16, SDPA attention, no `torch.compile`, no quantization |
| LM | `acestep-5Hz-lm-1.7B`, vLLM backend (only in the "with LM" run) |
| Input | Fixed pop caption + 2-verse/2-chorus English lyrics; BPM, key and time signature given explicitly |
| Protocol | Batch size 1, one warm-up run, then 3 timed runs per length (2 with LM); the table reports the median |

"Wall time" covers the whole `generate_music()` call: LM, text encoder, diffusion, VAE decode, normalization and saving the FLAC file.

## Results

### Time per song

| Song length | Without LM | With LM | Real-time factor (no LM / LM) |
|---:|---:|---:|---:|
| 30 s | 1.45 s | 6.78 s | 0.048 / 0.226 |
| 60 s | 2.02 s | – | 0.034 / – |
| 120 s | 3.22 s | 23.65 s | 0.027 / 0.197 |
| 180 s | 4.80 s | 34.83 s | 0.027 / 0.194 |

A real-time factor below 1 means faster than real time; 0.027 is about 37× faster than real time. The 60 s length was not measured with the LM.

### Where the time goes (median, seconds)

| Song length | LM (audio codes) | DiT diffusion (8 steps) | VAE decode | Total |
|---:|---:|---:|---:|---:|
| 30 s, with LM | 4.75 | 0.35 | 0.75 | 6.78 |
| 120 s, with LM | 19.67 | 0.75 | 2.87 | 23.65 |
| 180 s, with LM | 30.13 | 1.27 | 2.83 | 34.83 |
| 180 s, without LM | – | 1.29 | 2.62 | 4.80 |

The LM writes 5 audio-code tokens per second of audio, one token at a time, so its cost grows linearly with song length (about 30 tokens/s). The diffusion and VAE stages run in parallel over the whole song and stay at a few seconds.

### Memory and loading

| | Without LM | With LM |
|---|---:|---:|
| Peak VRAM | 6.4 GB | 11.1 GB |
| Model load time | 7 s | 40 s |
| Weights on disk | ~6 GB | ~9.5 GB |

## Compute budget for the planned experiments

| Run | Songs | Without LM (30 s / 120 s songs) | With LM (30 s / 120 s songs) |
|---|---:|---:|---:|
| Week 3: 20 artists × 50 | 1,000 | 0.4 h / 0.9 h | 1.9 h / 6.6 h |
| Week 5: 30 artists × 50 | 1,500 | 0.6 h / 1.3 h | 2.8 h / 9.9 h |
| All 50 artists × 50 | 2,500 | 1.0 h / 2.2 h | 4.7 h / 16.4 h |

The schedule assumes about 4 h, 6 h and 8 h for these runs. Those figures hold for every option except **full-length songs with the LM**, which need overnight runs. The LEACE re-generation in week 8 costs the same as the corresponding row above.

## How ACE-Step 1.5 generates a song

ACE-Step 1.5 has two stages. An optional **planner**, the 5Hz LM, sketches the song. A **renderer**, the diffusion model (DiT) together with its encoders and the VAE, turns that sketch plus the text into audio. Everything below was checked against the code at the pinned commit (`acestep/inference.py`, `checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py`).

```
 caption ("upbeat pop, female vocals…")        lyrics ("[Verse 1] Walking down…")
      │                                              │
      ├──────────────────────┐                       ├───────────────────────┐
      │                      ▼                       │                       ▼
      │   ┌─────────────────────────────────────────────────┐                │
      │   │ STAGE 1 (optional): 5Hz LM, 1.7B, autoregressive│                │ 
      │   │  phase 1: <think> metadata: BPM, key, duration, │                │
      │   │           language, rewritten caption           │                │
      │   │  phase 2: audio codes, 5 tokens per second      │                │
      │   └─────────────────────────────────────────────────┘                │
      │                      │ audio codes (e.g. 900 tokens for 3 min)       │
      ▼                      ▼                                               ▼
 Qwen3-Embedding-0.6B    Detokenizer: 5 Hz → 25 Hz          Qwen3 token embeddings only
 (full model forward)    latent "sketch"                    (lookup table, no forward)
      │                      │                                               │
      ▼                      │                                               ▼
 text projector              │                              Lyric encoder (8 transformer
      │                      │                              layers, part of the DiT)
      │                      │                                               │
      └────────┬─────────────┼───────────────────────────────────────────────┘
               │             │
               ▼             ▼
      condition sequence   context latents
      [lyrics | timbre |   (LM sketch, or silence
       caption]            if no LM)
               │             │
               ▼             ▼
      ┌───────────────────────────────────────┐
      │ STAGE 2: DiT (24 layers, turbo)       │  starts from random noise,
      │ flow matching, 8 denoising steps      │  cross-attends to the condition
      └───────────────────────────────────────┘  sequence at every step
                       │ latent audio: 64 channels × 25 frames per second
                       ▼
                 VAE decoder  →  48 kHz stereo waveform
```

### What each part does

| Component | Input → output | Role | Time for a 3-min song |
|---|---|---|---:|
| **5Hz LM**, phase 1 ("thinking") | caption + lyrics → text in a `<think>` block | Fills in missing metadata: BPM, key, time signature, duration, vocal language, and an expanded caption. Skipped when these are given, as in our benchmark. | 0 s (skipped) |
| **5Hz LM**, phase 2 (audio codes) | caption + lyrics + metadata → discrete tokens | Writes a coarse **plan of the song**: 5 tokens per second, each an index into a quantized audio vocabulary (FSQ). It works like a sketch of melody, rhythm and structure. Tokens are produced one at a time with classifier-free guidance (`lm_cfg_scale=2.0`, two forward passes per token), which is why this stage is slow. | ~30 s |
| **Detokenizer** (in the DiT model) | 5 Hz codes → 25 Hz latents | Expands each code into 5 latent frames, so the plan has the same time resolution as the audio latents. | negligible |
| **Qwen3-Embedding-0.6B** (caption) | caption tokens → 1024-d vectors | Runs the full text model on the **caption** (style, genre, instruments, mood), then a linear projector maps it to the DiT width (2048). | ~0.02 s |
| **Qwen3 token embeddings** (lyrics) | lyric tokens → 1024-d vectors | **Lyrics do not go through the Qwen3 transformer.** Only its token-embedding lookup table is used, so at this point each token carries no context. | negligible |
| **Lyric encoder** (8 transformer layers, in the DiT checkpoint) | lyric token vectors → contextual lyric vectors | The only place where lyrics are processed in context. It is trained together with the DiT. | negligible |
| **Timbre encoder** | reference audio → timbre vectors | Optional voice/timbre reference. With no reference audio, it gets a silent placeholder. | negligible |
| **Condition sequence** | lyric + timbre + caption vectors | The three are concatenated into one sequence that the DiT reads through cross-attention. | – |
| **Context latents** | LM sketch, or silence latents | With the LM, the detokenized codes give the DiT a time-aligned outline of the song. Without it, the context is a silence latent, and all structure must come from the text. | – |
| **DiT** (24 layers, hidden size 2048, turbo) | noise + context + conditions → audio latents | A diffusion transformer trained with flow matching. The turbo variant is distilled to **8 steps**. Each step refines the whole song at once (in parallel over time), which is why it is fast. | ~1.3 s |
| **VAE decoder** | 64 × 25 Hz latents → 48 kHz stereo | Turns the compressed latents back into a waveform. It is tiled into chunks to save memory and dominates the non-LM time. | ~2.8 s |

### With and without the LM

| | Without LM (`thinking=False`) | With LM (`thinking=True`, default) |
|---|---|---|
| Where song structure comes from | DiT only, from the caption and lyrics | LM plan (codes) + DiT |
| What the DiT receives as context | silence latents | detokenized LM codes |
| Speed for a 3-min song | ~5 s | ~35 s |
| Models in VRAM | DiT + encoders + VAE (6.4 GB) | + 1.7B LM (11.1 GB) |

### Relevance for Experiment 1 (where to put hooks)

The design proposal says "the model's text encoder". In ACE-Step 1.5, the representation that sees the **lyrics in context** is the **8-layer lyric encoder** (`AceStepConditionEncoder.lyric_encoder`). Qwen3-Embedding only processes the caption, and for lyrics it provides the token lookup table. The lyric encoder layers are therefore the natural place for activation hooks and LEACE. If artist names are kept out of the caption, the caption path should carry no artist information. The 5Hz LM, by contrast, reads the lyrics itself, so it holds its own, separate representation of them.

## Implications for the project

1. **The LM is a second possible source of copying.** Autoregressive LMs are known to memorize and repeat training sequences. If generations use the LM, a copy detected in Experiment 2 may come from the LM's audio codes rather than from the lyrics-encoder representation we probe (Exp. 1) and erase with LEACE (Exp. 4). LEACE on the encoder does not touch the LM.
2. **Options:**
   - *LM off:* about 7× faster, and the encoder → DiT path is the only route for artist identity, so RQ1 and RQ2 are cleaner.
   - *LM on:* matches the default ACE-Step pipeline and possibly the setup in the baseline papers (to be checked).
   - *Both (suggested):* run the main experiments with the LM off, plus a smaller LM-on control to see how much copying the LM adds.
3. **Song length** affects cost about as much as the LM choice. Lyrics-based copying needs enough of the song to contain the lyrics, so we should pick a length that covers at least a verse and a chorus.

## Not yet measured

- Batch sizes above 1. The script supports `--batch-size N`, which should raise throughput, especially for the LM.
- `torch.compile` and flash-attention: the DiT is already fast, so these would gain little.
- Quality of the generated audio. The FLAC samples are in `outputs/benchmark/audio/` for a quick listen.

## How to reproduce

```bash
uv sync
uv run python scripts/benchmark_generation.py                 # without LM
uv run python scripts/benchmark_generation.py --with-lm       # with 5Hz LM
# options: --durations 30 120 --repeats 3 --batch-size 2 --inference-steps 8
```

Weights download automatically to `checkpoints/` on the first run, about 9.5 GB. Results are saved as JSON in `outputs/benchmark/benchmark_<model>[_lm]_bs<N>.json`, with a per-stage time breakdown for each run.
