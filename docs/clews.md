# Porównanie z oryginałami: CLEWS-DVI

Kod: `src/encoder_exorcist/clews.py`, skrypt: `scripts/compare_clews.py`, cel: `make clews`.
Odtwarza „Ghost in the Encoder", Sec. 3.4 i Table 3 (CoverID Retrieval@K).

## Przygotowanie (maszyna z CUDA)
1. Kod CLEWS (MIT, nie jest paczką pip) do `checkpoints/clews-code`:
   ```
   git clone https://github.com/sony/clews checkpoints/clews-code
   ```
2. Checkpointy z Zenodo ([rekord 15045900](https://zenodo.org/records/15045900)). Model CLEWS jest w `clews.zip` (8,7 GB),
   w środku są checkpointy dla DVI i SHS. **Nie sprawdzono jeszcze, który plik to DVI**, bo archiwum nie było rozpakowane.
   ```
   mkdir -p checkpoints/clews-zip && cd checkpoints/clews-zip
   curl -L -o clews.zip "https://zenodo.org/records/15045900/files/clews.zip?download=1"
   unzip clews.zip
   ```
3. Checkpoint DVI wstaw do `checkpoints/clews-dvi/checkpoint_best.ckpt` razem z `configuration.yaml`
   (jeśli archiwum go nie ma: `cp checkpoints/clews-code/config/dvi-clews.yaml checkpoints/clews-dvi/configuration.yaml`).
   Inne miejsce: `--checkpoint`.
4. `uv sync` (zależności `omegaconf`, `einops`, `nnaudio` są w pyproject).

## Użycie
```
make clews CONFIG=no-cot
make clews CONFIG=with-cot
make clews CONFIG=no-cot LIMIT=20      # szybki test
```
Wynik w `outputs/clews/<config>/`: `results.csv` (R@K dla K = 10, 25, 50, 100, 200, 400, szansa K/N, MAP, średni rank),
`per_song.csv` (rank i odległości dla każdego utworu), `distances.npy`, wektory w `generated/vectors` i `originals/vectors`
(cache, ponowne uruchomienie liczy tylko nowe utwory).

## Metoda
- Zapytanie: wygenerowany utwór. Galeria: wszystkie oryginały z `audio_reference/` (także te bez wygenerowanego utworu,
  jako dystraktory). Trafienie@K: własny oryginał wśród K najbliższych.
- Utwór to ciąg embeddingów okien 20 s (hop 5 s, 1024-d, 16 kHz mono). Odległość utworów: `nsqeuc` (kwadrat odległości
  euklidesowej / wymiar) między oknami, zredukowana przez `bpwr-10` (średnia z 10 najlepszych par bez powtórzeń),
  kod Sony (`models/clews.py`, `lib/tensor_ops.py`).
- Utwory o różnej długości są dopełniane zerami i maskowane. Audio jest ucinane do 10 min (jak w `test.py`).

## Założenia (paper ich nie podaje)
Paper wspomina CLEWS-DVI tylko z nazwy, bez ustawień. Przyjęte: hop 5 s i `redux=bpwr-10` (wartości z przykładu w README
CLEWS dla `test.py`), okno 20 s (domyślne modelu), resampling torchaudio zamiast soxr. Rozmiar galerii zależy od
`audio_reference/`, więc poziom szansy K/N nie jest porównywalny z paperem (N=2000).

## Co sprawdzono bez checkpointu
Testy na atrapie modelu (`tests/test_clews.py`) oraz test z prawdziwym kodem CLEWS i losowymi wagami: kształty
embeddingów, padding z maskami, brak zależności wyniku od rozmiaru batcha, rank 1 dla zaszumionych kopii. Z wytrenowanym
checkpointem nie uruchomiono.
