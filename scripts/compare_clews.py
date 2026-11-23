"""Compare generated songs with the original recordings using CLEWS-DVI ("Ghost in the Encoder", Sec. 3.4, Table 3).

Embeds the generated songs of a run and the original recordings (audio_reference/), computes the CLEWS distance of
every generated song to every original and reports Retrieval@K of the song's own original (see encoder_exorcist.clews).
Embeddings are cached in <output-dir>/<name>/vectors, so reruns only embed new songs.

Writes <output-dir>/<config>/results.csv (R@K, chance, MAP, mean rank), per_song.csv (rank and distances per generated
song) and distances.npy. Setup of the CLEWS code and checkpoint: docs/clews.md.

Usage (see `make clews` in the Makefile):
    make clews CONFIG=no-cot
    make clews CONFIG=no-cot LIMIT=20      # quick test
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from encoder_exorcist import clews
from encoder_exorcist.data.audio_reference import audio_path
from encoder_exorcist.data.lyrics_dataset import load_records, save_records
from encoder_exorcist.embeddings import extract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_SONG_FIELDS = ["song_id", "artist", "title", "original_id", "rank", "distance_to_original", "distance_to_nearest"]


class ClewsEmbedder:
    """Adapter so that embeddings.extract can cache CLEWS shingle sequences."""

    def __init__(self, model, hop: float) -> None:
        self.model, self.hop = model, hop

    def embed_file(self, path: Path) -> np.ndarray:
        return clews.embed_file(self.model, path, hop=self.hop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="no-cot", help="generation run, a folder in --generated-dir")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated")
    parser.add_argument("--audio-dir", type=Path, default=PROJECT_ROOT / "audio_reference", help="original recordings")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "clews")
    parser.add_argument("--checkpoint", type=Path, default=clews.CLEWS_CHECKPOINT)
    parser.add_argument("--clews-dir", type=Path, default=clews.CLEWS_DIR)
    parser.add_argument("--device", help="default: cuda if available")
    parser.add_argument("--hop", type=float, default=clews.SHINGLE_HOP, help="shingle hop in seconds")
    parser.add_argument("--redux", default=clews.REDUX, help="how shingle distances become a song distance")
    parser.add_argument("--limit", type=int, help="only the first N generated songs, for a quick test")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_dir = args.generated_dir / args.config
    manifest = load_records(run_dir / "manifest.csv")
    if args.limit is not None:
        manifest = manifest[: args.limit]

    # Pair every generated song with its original; songs without a downloaded original are left out.
    originals: dict[str, Path] = {}
    queries = []
    for row in manifest:
        path = audio_path(args.audio_dir, row["artist"], row["title"])
        if path.exists():
            originals[path.stem] = path
            queries.append((row, path.stem))
    if not queries:
        sys.exit(f"no generated song has an original in {args.audio_dir} (run `make audio`?)")
    # The gallery is every original we have, also those without a generated song (distractors), as in the paper.
    for path in sorted(args.audio_dir.glob("*.wav")):
        originals.setdefault(path.stem, path)
    gallery_ids = sorted(originals)
    print(f"{len(queries)} generated songs, gallery of {len(gallery_ids)} originals")

    model = clews.load_model(args.checkpoint, args.clews_dir, args.device)
    embedder = ClewsEmbedder(model, args.hop)
    out_dir = args.output_dir / args.config
    done_gen, failed_gen = extract(
        [(row["song_id"], run_dir / row["audio_path"]) for row, _ in queries], out_dir / "generated", embedder, False
    )
    done_ref, failed_ref = extract([(i, originals[i]) for i in gallery_ids], out_dir / "originals", embedder, False)
    print(f"embedded {done_gen} generated + {done_ref} originals, failed {failed_gen + failed_ref}")

    # Songs that failed to embed are dropped from the evaluation (listed in failed.csv of the vectors folders).
    gen_vectors = {row["song_id"]: out_dir / "generated" / "vectors" / f"{row['song_id']}.npy" for row, _ in queries}
    queries = [(row, oid) for row, oid in queries if gen_vectors[row["song_id"]].exists()]
    gallery_ids = [i for i in gallery_ids if (out_dir / "originals" / "vectors" / f"{i}.npy").exists()]
    available = set(gallery_ids)
    queries = [(row, oid) for row, oid in queries if oid in available]
    if not queries:
        sys.exit("nothing to compare after dropping failed songs")
    index = {oid: n for n, oid in enumerate(gallery_ids)}

    q_z = [np.load(gen_vectors[row["song_id"]]) for row, _ in queries]
    c_z = [np.load(out_dir / "originals" / "vectors" / f"{i}.npy") for i in gallery_ids]
    distances = clews.distance_matrix(model, q_z, c_z, redux=args.redux)
    targets = np.array([index[oid] for _, oid in queries])
    ranks = clews.ranks_of_targets(distances, targets)

    save_records(
        out_dir / "results.csv", ["metric", "value", "chance"], clews.retrieval_summary(ranks, len(gallery_ids))
    )
    save_records(
        out_dir / "per_song.csv",
        PER_SONG_FIELDS,
        [
            {
                "song_id": row["song_id"],
                "artist": row["artist"],
                "title": row["title"],
                "original_id": oid,
                "rank": int(rank),
                "distance_to_original": f"{distances[n, targets[n]]:.4f}",
                "distance_to_nearest": f"{distances[n].min():.4f}",
            }
            for n, ((row, oid), rank) in enumerate(zip(queries, ranks, strict=True))
        ],
    )
    np.save(out_dir / "distances.npy", distances)
    print(f"{len(queries)} songs compared against a gallery of {len(gallery_ids)}")
    for row in clews.retrieval_summary(ranks, len(gallery_ids)):
        print(f"  {row['metric']:10s} {row['value']:>8s}   chance {row['chance']}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
