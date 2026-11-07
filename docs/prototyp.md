# Prototyp: stan projektu

*Stan na 7.11.2026.*

Ten dokument zbiera, co zrobiliśmy w ramach prototypu i jakie mamy wyniki.

## W skrócie

- **Cały proces od tekstów do wyniku sondy działa.** Zbieramy teksty i anonimizujemy nazwy artystów. Potem ACE-Step generuje piosenki, a hooki (przechwytywacze aktywacji) w tym czasie zapisują aktywacje modelu. Na końcu sonda liniowa uczy się na tych aktywacjach.
- **Dekodowalność tożsamości artysty (RQ1) jest potwierdzona.** ACE-Step 1.5 koduje tożsamość artysty na podstawie samego tekstu, bez nazwy artysty. Na 1000 wygenerowanych piosenkach sonda na enkoderze tekstu piosenki rozpoznaje artystę w **63%** przypadków przy losowym poziomie 5% (20 artystów, p = 0,001).

## Przebieg przetwarzania

```mermaid
flowchart LR
    A[Last.fm + Genius<br>teksty 20 artystów] --> B[Anonimizacja<br>nazw artystów]
    A --> C[YouTube<br>oryginalne nagrania]
    B --> D[ACE-Step 1.5<br>generacja z samego tekstu]
    D -->|hooki| E[Aktywacje<br>.npz]
    D --> F[Wygenerowane<br>piosenki]
    E --> G[Sonda liniowa<br>RQ1: dekodowalność]
    F --> H[Odsłuch<br>porównawczy]
    C --> H
```

## 1. Dane

**Wybór artystów i tekstów.** Moduł `encoder_exorcist/data/lyrics_dataset.py` pobiera najpopularniejszych artystów dla 5 gatunków z Last.fm, a ich teksty z Genius. Uruchomienie: `make data`. Odrzuca remiksy, covery i wersje live, a tytuły deduplikuje dopasowaniem rozmytym, żeby remaster nie liczył się jako osobny utwór.

| Gatunek | Artyści |
|---|---|
| pop | Ariana Grande, Sabrina Carpenter, Olivia Rodrigo, Lady Gaga |
| rock | Radiohead, Paramore, Linkin Park, My Chemical Romance |
| hip-hop | Kanye West, Kendrick Lamar, Tyler, The Creator, Drake |
| R&B | The Weeknd, Tinashe, Rihanna, Frank Ocean |
| country | Taylor Swift, Dolly Parton, Kacey Musgraves, Zach Bryan |

Zbiór referencyjny to **50 utworów na artystę, czyli 1000 tekstów**, wydanych między 1962 a 2024 r. Status treningowy tych utworów jest nieznany. Są popularne i wydane przed modelem, więc prawdopodobnie były w danych treningowych. Pierwsza wersja zbioru miała 20 utworów na artystę i Johnny'ego Casha zamiast Zacha Bryana. Na niej powstał pierwszy przebieg generacji i sondy (400 piosenek).

**Anonimizacja.** Moduł `encoder_exorcist/data/anonymization.py`, uruchamiany przez `make anonymize`. Nazwy artystów i ich aliasy pochodzą z MusicBrainz. Każde wystąpienie w tekście jest zastępowane podobnie brzmiącym słowem z CMUdict, z tą samą liczbą sylab i tym samym akcentem. Wymowę nazw spoza słownika przewiduje model g2p. Przykłady: „Kendrick” → „eric”, „Rihanna” → „hourihan”, „Kanye” → „candy”. Możliwe są też ręczne zamiany. W 1000 tekstów podmieniono **151 wystąpień u 11 artystów**.

**Nagrania referencyjne.** Moduł `encoder_exorcist/data/audio_reference.py`, uruchamiany przez `make audio`. Oryginały są pobierane z YouTube'a jako WAV. Leżą na Drive w folderze `Audio References`, w zipach po gatunkach.

## 2. Model i budżet obliczeniowy

Używamy modelu ACE-Step 1.5 w wariancie turbo (8 kroków dyfuzji), w wersji przypiętej w `pyproject.toml`. Pomiary na RTX 5070 Ti są opisane w [benchmark_generation.md](benchmark_generation.md):

| Długość utworu | Bez 5Hz LM (no-CoT) | Z 5Hz LM (with-CoT) |
|---:|---:|---:|
| 30 s | 1,5 s | 6,8 s |
| 180 s | 4,8 s | 34,8 s |

1000 piosenek po 180 s to **ok. 1,5 h w no-CoT** i ok. 10–13 h w with-CoT. Najwolniejszy jest 5Hz LM, który generuje kody audio token po tokenie.

Ważny wniosek z analizy kodu ACE-Step: tekst piosenki jest przetwarzany w kontekście tylko przez **8-warstwowy enkoder tekstu piosenki (lyric encoder)** wewnątrz DiT. Qwen3-Embedding przetwarza wyłącznie opis stylu, a dla tekstu daje jedynie tablicę embeddingów tokenów. Dlatego hooki i interwencja LEACE dotyczą enkodera tekstu piosenki.

## 3. Ekstrakcja aktywacji

Klasa `ActivationRecorder` (`encoder_exorcist/activation_recorder.py`) podpina **62 hooki** w punktach opisanych w artykule „Ghost in the Encoder”. Szczegóły są w [hook_points.md](hook_points.md):

- wyjścia strumieni warunkowania: tekst, opis, barwa, połączona sekwencja;
- 8 warstw enkodera tekstu piosenki;
- projekcje K i V atencji krzyżowej (cross-attention) w 10 warstwach DiT;
- stany audio przed atencją krzyżową, po niej i na jej wyjściu, w tych samych 10 warstwach i w każdym z 8 kroków odszumiania.

Każda aktywacja jest uśredniana po tokenach, co daje ok. 2 MB na piosenkę.

Kształty i liczby wywołań **zweryfikowaliśmy na prawdziwym modelu** w obu trybach. Wszystko zgadza się z założeniami, np. 2250 klatek audio przy 180 s, 8 kroków i K/V liczone raz. Sprawdzenie da się powtórzyć poleceniem `make generate-check`.

## 4. Generacja piosenek (PR #8)

`make generate` generuje piosenki z zanonimizowanych tekstów z podpiętymi hookami. Model dostaje **tylko tekst**: bez nazwy artysty, opisu stylu, tempa i tonacji (`caption = None`, `bpm = None`, `key = None`). Wszystkie piosenki mają 180 s, tak jak w artykule. Dostępne są dwa tryby:

| Tryb | 5Hz LM | Co dostaje DiT |
|---|---|---|
| `no-cot` | wyłączony | tekst i pusty opis |
| `with-cot` | włączony | tekst, opis i metadane napisane przez LM, kody audio z LM |

Ziarno losowe każdej piosenki jest stałe, więc ta sama piosenka ma ten sam szum w obu trybach. Przerwana generacja wznawia się od miejsca przerwania.

**Wynik:** 1000 piosenek no-CoT (20 artystów × 50 tekstów × 1 ziarno losowe), bez błędów, ok. 5 s na piosenkę. Pliki to `outputs/generated_50songs/no-cot/` z audio FLAC, aktywacjami `.npz` i `manifest.csv`. Na Drive przygotowane są zipy z MP3 (5 gatunków po ok. 860 MB) i z aktywacjami (ok. 2 GB).

## 5. Sonda liniowa: wynik dla RQ1 (PR #9)

`make probe` trenuje osobną regresję logistyczną dla każdego z **303 punktów pomiarowych** i dla dwóch punktów odniesienia. Protokół jest jak w artykule:
- 5-krotna walidacja krzyżowa (foldy podzielone po piosenkach);
- 95% przedziały ufności;
- test permutacyjny z 1000 permutacjami.

Wykresy są w [notebooks/probe_results.ipynb](../notebooks/probe_results.ipynb). Wyniki dla 1000 piosenek:

| Punkt pomiaru | Dokładność | Przedział ufności 95% | p |
|---|---:|---|---:|
| warstwa L5 enkodera tekstu | **63,2%** | 60,2–66,1% | 0,001 |
| sekwencja warunkująca DiT (`merged_out`) | 62,8% | 59,8–65,7% | 0,001 |
| wyjście enkodera tekstu (`lyric_out`) | 62,1% | 59,1–65,1% | 0,001 |
| punkt odniesienia: TF-IDF z samych słów | 53,8% | 50,7–56,9% | 0,001 |
| punkt odniesienia: embeddingi tokenów Qwen3 (wejście enkodera) | 43,8% | 40,8–46,9% | 0,001 |
| kontrola: opis i barwa (stałe w no-CoT) | 5,0% | 3,8–6,5% | 1 |
| poziom losowy (20 artystów) | 5,0% | | |

![Dokładność w kolejnych warstwach enkodera tekstu](img/probe_lyric_encoder.png)

**Wnioski:**
- **Tożsamość artysty da się odczytać z samego tekstu.** Enkoder tekstu dodaje coś ponad same słowa, bo od warstwy L3 bije oba punkty odniesienia. Dokładność rośnie w pierwszych warstwach i od L3 utrzymuje się na stałym poziomie.
- **Sygnał trafia do DiT.** Projekcje K/V atencji krzyżowej zachowują 52–60%, bo liczą się bezpośrednio z sekwencji z tekstem. W stanach audio sygnał jest słabszy (16–53%): najmocniejszy w pierwszych warstwach tuż po atencji krzyżowej, słabnie z głębokością i w pierwszych krokach odszumiania.
- **Punkty kontrolne zachowują się poprawnie.** Strumienie stałe w trybie no-CoT wychodzą dokładnie na poziomie losowym.
- **Dokładność bardzo różni się między artystami:** od 26% (Rihanna) do 94% (Zach Bryan). Tego potrzebuje RQ1, czyli korelacji dekodowalności z kopiowaniem.
- **Wynik jest stabilny.** Pierwszy przebieg na 400 piosenkach dał ten sam obraz, tylko o ok. 7 punktów procentowych niżej (55,5% na `lyric_out`, TF-IDF 42%). Więcej danych treningowych podnosi wszystkie sondy, a przedziały ufności są węższe.

![Dokładność w warstwach DiT](img/probe_dit_layers.png)

![Dokładność dla poszczególnych artystów](img/probe_per_artist.png)

**Poprawka metodologiczna.** Standardowy `permutation_test_score` ze sklearn, przy stałych stratyfikowanych foldach, dawał zaniżony rozkład zerowy. Cecha stała, na dokładnie losowym poziomie, dostawała p = 0,001. W naszej implementacji każda permutacja losuje foldy od nowa, a test regresyjny pilnuje tego zachowania.

## 6. Odsłuch porównawczy (PR #7)

`tools/song_compare.html` to statyczna strona bez serwera. Wskazuje się w niej folder z wygenerowanymi piosenkami i folder z oryginałami, a po kliknięciu tytułu pojawiają się dwa odtwarzacze. Strona ma też wyszukiwarkę, filtr gatunku, znacznik „przesłuchane” i skróty klawiszowe. Opis jest w [song_compare.md](song_compare.md).

**Pierwsze wrażenia z odsłuchu:** wygenerowane piosenki wyraźnie różnią się od oryginałów. Przy samym tekście, bez opisu stylu, model nie odtwarza oryginalnej melodii.

## 7. Jak odtworzyć

```bash
make install
make data && make anonymize      # teksty i anonimizacja (klucze API w .env)
make audio                       # oryginalne nagrania (YouTube)
make generate-check              # sprawdzenie hooków na jednej piosence

# 1000 piosenek no-CoT + aktywacje (~1,5 h, GPU)
make generate CSV="outputs/anonymized_50songs/lyrics_*.csv" GENERATED_DIR=outputs/generated_50songs
# sondy, ~3,5 h na 20 rdzeniach (PERMUTATIONS=0: dużo krócej, bez p-wartości)
make probe GENERATED_DIR=outputs/generated_50songs LYRICS_DIR=outputs/anonymized_50songs PROBE_DIR=outputs/probe_50songs
```

Wykresy powstają w `notebooks/probe_results.ipynb`.

## Status zadań

| Zadanie | Stan |
|---|---|
| Środowisko (WIMU-6), benchmark GPU (WIMU-8), hooki (WIMU-11) | ✅ |
| Wybór artystów (WIMU-9) | ✅ |
| Teksty, anonimizacja, nagrania (WIMU-10, 22, 25) | ✅ |
| Skrypt generacji (WIMU-20), sonda (WIMU-21) | ✅ |
| Generacja 1000 piosenek, 50 na artystę (WIMU-13) | ✅ |
| Analiza literatury (WIMU-7) | ✅ |
