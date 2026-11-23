# Embeddingi audio (LAION-CLAP, MuQ-MuLan)

Kod: `src/encoder_exorcist/embeddings.py`, skrypt: `scripts/extract_embeddings.py`, cel: `make embed`.

## Przygotowanie (maszyna z CUDA)
1. `uv sync`
2. Checkpoint LAION-CLAP (muzyczny) do `checkpoints/`:
   ```
   uv run hf download lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt --local-dir checkpoints
   ```
3. MuQ-MuLan (`OpenMuQ/MuQ-MuLan-large`) pobiera się sam z HuggingFace przy pierwszym uruchomieniu.

## Użycie
```
make embed MODEL=clap  SOURCE=generated CONFIG=no-cot
make embed MODEL=mulan SOURCE=generated CONFIG=no-cot
make embed MODEL=clap  SOURCE=reference     # audio_reference/
make embed MODEL=clap  SOURCE=reference LIMIT=5   # szybki test
```
Wynik: `outputs/embeddings/<model>/generated/<config>/` lub `.../reference/` z plikami
`embeddings.npz` (`song_ids`, `embeddings` [n, 512]), `vectors/<song_id>.npy`, `failed.csv`.
Ponowne uruchomienie pomija policzone utwory.

## Metoda
Mono, resampling (CLAP 48 kHz, MuLan 24 kHz), okna bez nakładania (CLAP 10 s, MuLan 30 s),
średnia embeddingów okien i normalizacja L2 → jeden wektor na utwór.
