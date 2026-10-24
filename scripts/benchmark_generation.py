"""Benchmark ACE-Step 1.5 generation speed on the local GPU.

Estimates the compute budget for the generation experiments (e.g. 50 artists x 50 songs).

Usage:
    uv run python scripts/benchmark_generation.py                     # DiT only (no LM)
    uv run python scripts/benchmark_generation.py --with-lm           # DiT + 5Hz LM "thinking"
    uv run python scripts/benchmark_generation.py --durations 30 120 --repeats 3 --batch-size 2
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import ensure_lm_model, ensure_main_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CAPTION = "upbeat pop song, female vocals, catchy chorus, synths, drums, bass"
LYRICS = """[Verse 1]
Walking down the empty street tonight
Neon signs are flickering with light
Every step I take I hear your name
Nothing in this city feels the same

[Chorus]
Hold on, hold on, don't let it go
Turn up the radio, let the whole world know
Hold on, hold on, we're burning bright
We're gonna dance until the morning light

[Verse 2]
Pictures on the wall are fading fast
Trying to forget about the past
But every song reminds me of the night
When you and I were standing in the light

[Chorus]
Hold on, hold on, don't let it go
Turn up the radio, let the whole world know
Hold on, hold on, we're burning bright
We're gonna dance until the morning light
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-path", default="acestep-v15-turbo", help="DiT model variant")
    parser.add_argument("--durations", type=float, nargs="+", default=[30, 60, 120, 180], help="Song lengths [s]")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per duration")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=8, help="Turbo default is 8")
    parser.add_argument("--with-lm", action="store_true", help="Also run the 5Hz LM (thinking/CoT) stage")
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--lm-backend", default="vllm", choices=["vllm", "pt"])
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark")
    return parser.parse_args()


def init_models(args: argparse.Namespace) -> tuple[AceStepHandler, LLMHandler]:
    ok, msg = ensure_main_model(CHECKPOINTS_DIR)
    if not ok:
        raise RuntimeError(msg)

    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path=args.config_path,
        device="cuda",
        use_flash_attention=dit_handler.is_flash_attention_available("cuda"),
        offload_to_cpu=args.offload_to_cpu,
    )
    if not ok:
        raise RuntimeError(f"DiT init failed: {status}")
    print(status)

    llm_handler = LLMHandler()
    if args.with_lm:
        ok, msg = ensure_lm_model(model_name=args.lm_model, checkpoints_dir=CHECKPOINTS_DIR)
        if not ok:
            raise RuntimeError(msg)
        status, ok = llm_handler.initialize(
            checkpoint_dir=str(CHECKPOINTS_DIR),
            lm_model_path=args.lm_model,
            backend=args.lm_backend,
            device="cuda",
            offload_to_cpu=args.offload_to_cpu,
        )
        if not ok:
            raise RuntimeError(f"LM init failed: {status}")
        print(status)
    return dit_handler, llm_handler


def run_once(dit_handler, llm_handler, args, duration: float, save_dir: Path, seed: int) -> dict:
    params = GenerationParams(
        caption=CAPTION,
        lyrics=LYRICS,
        vocal_language="en",
        duration=duration,
        bpm=120,
        keyscale="C major",
        timesignature="4",
        inference_steps=args.inference_steps,
        seed=seed,
        thinking=args.with_lm,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(batch_size=args.batch_size, use_random_seed=False, seeds=[seed], audio_format="flac")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = generate_music(dit_handler, llm_handler, params, config, save_dir=str(save_dir))
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    if not result.success:
        raise RuntimeError(f"Generation failed: {result.error}")

    return {
        "duration_s": duration,
        "wall_s": wall,
        "per_song_s": wall / args.batch_size,
        "rtf": wall / (duration * args.batch_size),  # < 1 means faster than real time
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "time_costs": result.extra_outputs.get("time_costs", {}),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out_dir / "audio"

    t0 = time.perf_counter()
    dit_handler, llm_handler = init_models(args)
    load_s = time.perf_counter() - t0
    print(f"Model load: {load_s:.1f}s")

    print("Warm-up run...")
    run_once(dit_handler, llm_handler, args, duration=min(args.durations), save_dir=audio_dir, seed=0)

    runs = []
    for duration in args.durations:
        for i in range(args.repeats):
            run = run_once(dit_handler, llm_handler, args, duration, save_dir=audio_dir, seed=i + 1)
            runs.append(run)
            print(
                f"duration={duration:>5.0f}s  run={i}  wall={run['wall_s']:6.2f}s  "
                f"per_song={run['per_song_s']:6.2f}s  RTF={run['rtf']:.3f}  peak_vram={run['peak_vram_gb']:.1f}GB"
            )

    print("\n=== Summary (median over repeats) ===")
    summary = []
    for duration in args.durations:
        per_song = statistics.median(r["per_song_s"] for r in runs if r["duration_s"] == duration)
        vram = max(r["peak_vram_gb"] for r in runs if r["duration_s"] == duration)
        summary.append({"duration_s": duration, "median_per_song_s": per_song, "max_peak_vram_gb": vram})
        print(
            f"{duration:>5.0f}s song: {per_song:6.2f}s/song  | 2500 songs ≈ {2500 * per_song / 3600:5.2f}h"
            f"  | peak VRAM {vram:.1f}GB"
        )

    tag = f"{args.config_path}{'_lm' if args.with_lm else ''}_bs{args.batch_size}"
    out_file = args.out_dir / f"benchmark_{tag}.json"
    out_file.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "args": {k: str(v) for k, v in vars(args).items()},
                "model_load_s": load_s,
                "summary": summary,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
