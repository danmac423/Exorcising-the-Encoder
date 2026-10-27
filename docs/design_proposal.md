# Design Proposal

# Autorzy

* Maksymilian Baj
* Daniel Machniak
* Natalia Pieczko

# Temat projektu

**Ghosts in the encoder: artist identity inside a text-to-song model, what it predicts about copying, and whether it can be erased**

A text-to-song model can be prompted to imitate a named artist and sometimes reproduces whole songs from its training data. A 2026 study found that such a model carries a linearly decodable representation of artist identity from the lyrics alone, with no artist name given: a small classifier (a probe) trained on the model's activations can tell which artist a song is by. This project asks whether that representation predicts copying, and whether it can be removed. Copying is measured with an exact test: instead of reading a similarity score against a threshold, the test compares training items with held-out items that were never seen, so that chance matches do not count.

# Cel projektu

Celem projektu jest zbadanie, czy liniowo dekodowalna reprezentacja tożsamości artysty w modelu text-to-song ACE-Step 1.5 **przewiduje kopiowanie utworów treningowych** oraz czy można ją **usunąć** bez istotnej utraty jakości generowanej muzyki.

Projekt odpowiada na trzy pytania badawcze:

* **RQ1 - Czy dekodowalność przewiduje kopiowanie?** Dla 50 artystów mierzymy dokładność sondy liniowej rozpoznającej artystę z aktywacji enkodera (bez podawania nazwy artysty) oraz wskaźnik podobieństwa do repertuaru referencyjnego (*similarity to reference set*). Utwory sprzed 2024 r. mają status **utworów o nieznanej przynależności treningowej** (*unconfirmed training status*). Kopiowanie mierzymy dokładnym testem statystycznym z kontrolą FDR: podobieństwo wygenerowanych utworów do utworów treningowych porównujemy z podobieństwem do utworów nigdy niewidzianych przez model, dzięki czemu przypadkowe dopasowania nie są liczone jako kopiowanie. Miarą związku jest korelacja rang Spearmana między dokładnością sondy a odsetkiem kopiowania.
* **RQ2 - Czy reprezentację można wymazać?** Na wybranej warstwie stosujemy zamkniętą, liniową metodę wymazywania pojęć (LEACE). Sprawdzamy, czy (a) dokładność sondy spada do poziomu losowego, (b) spada imitacja artysty, mierzona odległością w przestrzeni embeddingów weryfikacyjnych między wygenerowanymi a prawdziwymi utworami artysty, oraz (c) jak bardzo spada jakość, mierzona automatycznym sędzią i krótkim odsłuchem w zespole.
* **RQ3 - Czy wymazywanie tożsamości artysty jest odporne na autorytatywne opisy tekstowe (CoT)?** Badamy, czy usunięcie tożsamości artysty metodą LEACE z enkodera liryki pozostaje skuteczne w sytuacji, gdy model ACE-Step generuje wewnętrzny opis muzyki (opis tekstowy / Chain-of-Thought). Porównujemy poziom imitacji i jakości nagrań przed i po interwencji w dwóch wariantach: z wyłączonym oraz włączonym modułem auto-opisu.

**Kryterium falsyfikacji:** projekt uznajemy za obalony, jeśli dokładność sondy nie przewiduje kopiowania mierzonego dokładnym testem (brak istotnej korelacji), a wymazanie kierunku nie zmienia poziomu imitacji. Wynik negatywny również jest wartościowy: pokazuje, że dekodowalna tożsamość artysty nie jest mechanizmem odpowiedzialnym za kopiowanie.

**Rezultaty projektu:**

1. Reprodukcja sondy tożsamości artysty na ACE-Step (20, a następnie 50 artystów).
2. Pomiar kopiowania pod dokładną hipotezą zerową i korelacja z dokładnością sondy.
3. Analiza kompromisu imitacja/jakość po wymazaniu kierunku metodą LEACE.
4. Publiczne repozytorium z kodem umożliwiającym reprodukcję wyników oraz raport końcowy w formie artykułu naukowego, opisujący korelację, kompromis imitacja/jakość po wymazaniu i wnioski dla twórców modeli.

**Ograniczenia:** Zbiór treningowy modelu ACE-Step 1.5 jest nieudokumentowany. Bezpośrednia weryfikacja obecności utworów w danych treningowych jest niemożliwa, dlatego analizę opieramy na heurystyce: do puli treningowej włączamy utwory bardzo popularne (np. z top 100 Last.fm) wydane przed 2025 rokiem, maksymalizując prawdopodobieństwo ich obecności w zbiorze. Zbiór *held-out* (nigdy niewidziany przez model) będą stanowić utwory wydane po 2025 roku.

# Harmonogram

| Tydzień | Data | Zadania | Terminy |
| -- | -- | -- | -- |
| 1 | 12.10 - 18.10 | * Rozpisanie Design Proposal ✅<br>* Podział prac ✅<br>* Przygotowanie środowiska ✅<br>* Przygotowanie pliku z tabelaryczną analizą literatury zgodnie z wymogami projektu 🟡 | * 14.10 - Wybór tematu projektu ✅ |
| 2 | 19.10 - 25.10 | * Finalizacja i oddanie Design Proposal ✅<br>* Uruchomienie ACE-Step z udostępnionymi wagami, pomiar czasu generacji jednego utworu na naszym GPU (szacunek budżetu obliczeniowego) ✅<br>* Wybór 20 artystów z list popularności. 🟡<br>* Zebranie metadanych i podział na zbiór referencyjny (utwory wydane < 2025 o nieznanej przynależności) oraz zbiór do walidacji (utwory wydane > 2025). 🟡<br>* Implementacja ekstrakcji aktywacji enkodera (hooki na warstwach) 🟡 | * 21.10 - Oddanie Design Proposal ✅ |
| 3 | 26.10 - 01.11 | * Reprodukcja sondy tożsamości artysty dla 20 artystów (przegląd warstw, wybór najlepszej)<br>* Generacja 50 utworów na artystę dla 20 artystów (ok. 4h)<br>* Implementacja ekstrakcji embeddingów audio (CLEWS-DVI) w celu mierzenia wskaźnika podobieństwa względem repertuaru referencyjnego.<br>* Implementacja i przeprowadzenie mikro-treningu adaptera LoRA na zbiorze post-2025 (podział 50/50 na kontrolowany trening i held-out w celu zapewnienia wymienialności próbek dla testu zerowego).<br>* Integracja kodu dokładnych testów memoryzacji |  |
| 4 | 02.11 - 08.11 | * Pierwsze porównanie rozkładów wskaźników podobieństwa: wariant bazowy oraz wariant po mikro-treningu LoRA.<br>* Spisanie wyniku, który obaliłby projekt (kryterium falsyfikacji)<br>* Przygotowanie prototypu: notebook z sondą i pierwszym wynikiem testu kopiowania<br>* Przygotowanie prezentacji prototypu | * 04.11 - Zgłoszenie gotowości prototypu |
| 5 | 09.11 - 15.11 | * Prezentacja prototypu i uwzględnienie uwag prowadzącego<br>* Rozszerzenie zbioru danych do 50 artystów (teksty, metadane, splity)<br>* Generacja 50 utworów na artystę dla pozostałych 30 artystów (ok. 6h) | * 10.11 - Deadline na spotkanie z prezentacją prototypu |
| 6 | 16.11 - 22.11 | * Trening sondy dla 50 artystów, dokładność per artysta <br>* Obliczenie wskaźnika podobieństwa do repertuaru referencyjnego per artysta przy użyciu Exact Null Test z kontrolą FDR. |  |
| 7 | 23.11 - 29.11 | * Korelacja Spearmana: dokładność sondy a wskaźnik podobieństwa do repertuaru<br>* Analizy kontrolne (liczba utworów treningowych artysty, gatunek) jako zmienne zakłócające<br>* **Wynik RQ1** |  |
| 8 | 30.11 - 06.12 | * Implementacja LEACE na wybranej warstwie<br>* Sanity check: dokładność sondy po wymazaniu spada do poziomu losowego<br>* Generacja utworów z wymazanym kierunkiem (ok. 8h) |  |
| 9 | 07.12 - 13.12 | * Pomiar imitacji (embedding weryfikacyjny) przed i po wymazaniu<br>* Pomiar jakości: automatyczny sędzia + krótki odsłuch w zespole<br>* Krzywa kompromisu imitacja/jakość (różne warstwy)<br>* **Wynik RQ2** |  |
| 10 | 14.12 - 20.12 | * Przeprowadzenie eksperymentu ablacyjnego CoT vs no-CoT (odpowiedź na RQ3).<br>* Generacja utworów z wymazanym kierunkiem LEACE przy włączonym module automatycznego opisu muzyki (CoT).<br>* Porównanie skuteczności wymazywania i jakości generacji między trybem z opisem a trybem bez opisu. |  |
| 11 | 21.12 - 27.12 | * Finalne wykresy i tabele<br>* Szkic artykułu: wstęp, related work, metoda, setup eksperymentalny |  |
| 12 | 28.12 - 03.01 | Przerwa świąteczna 🧑🏻‍🎄🎄&#10;(bufor na ewentualne opóźnione eksperymenty) |  |
| 13 | 04.01 - 10.01 | * Pisanie i edycja paperu: wyniki, dyskusja, rekomendacje dla twórców modeli<br>* Wewnętrzna recenzja całości |  |
| 14 | 11.01 - 17.01 | * Ostatnie poprawki, README i instrukcja reprodukcji<br>* Oddanie projektu | * 15.01 - Oddanie projektu z możliwością zwolnienia z kolokwium |
| 15 | 18.01 - 24.01 | * Bufor (poprawki po ewentualnych uwagach) | * 22.01 - Ostateczne oddanie projektu |

# Planowany zakres eksperymentów

Realizacja projektu została podzielona na spójne bloki eksperymentalne, z których każdy odpowiada za dostarczenie danych niezbędnych do weryfikacji pytań badawczych.

### Eksperyment 1: Detekcja tożsamości artysty w aktywacjach (Linear Probing)

* **Cel:** Udowodnienie, że model ACE-Step koduje w swoich warstwach informacje o tożsamości artysty na podstawie samego tekstu.
* **Przebieg:**
  1. Przygotowanie zbioru tekstów (ang. *lyrics*) dla 50 artystów z podziałem na utwory najprawdopodobniej obecne w zbiorze treningowym (najpopularniejsze, wydane przed 2025 r.) oraz ukryte (held-out, wydane po dacie publikacji modelu).
  2. Implementacja hooków (przechwytywaczy) w PyTorch na wybranych warstwach wewnętrznych enkodera tekstu w modelu ACE-Step 1.5.
  3. Przepuszczenie tekstów przez model i zebranie odpowiadających im wektorów aktywacji.
  4. Wyuczenie małego klasyfikatora (sondy liniowej - *linear probe*, np. regresji logistycznej z walidacją krzyżową) przewidującego etykietę artysty na podstawie aktywacji.
* **Oczekiwany wynik:** Zidentyfikowanie warstwy lub warstw, w których dokładność klasyfikacji artysty (probe accuracy) jest najwyższa i statystycznie istotnie różna od losowej (random guess).

### Eksperyment 2: Pomiary podobieństwa z wykorzystaniem dokładnego testu zerowego (Exact Null Test) oraz walidacja LoRA

* **Cel:** Ustalenie obiektywnego wskaźnika podobieństwa wygenerowanych nagrań do repertuaru referencyjnego pod dokładną hipotezą zerową z korekcją FDR (Benjamini-Hochberg) oraz weryfikacja wymienialności próbek (*exchangeability*).
* **Przebieg:**
  1. Wygenerowanie puli próbnej: 50 utworów dla każdego z 50 badanych artystów (łącznie 2500 utworów).
  2. Moduł referencyjny (heurystyczny): Ekstrakcja embeddingów (CLEWS-DVI / LAION-CLAP) z wygenerowanych utworów, zbioru referencyjnego (< 2025, nieznana przynależność) oraz zbioru held-out (> 2025).
  3. **Eksperyment walidacyjny LoRA (kontrolowana ekspozycja):** Aby wyeliminować błąd braku wymienialności próbek ze względu na czas premiery, zamrażamy bazowy model i przeprowadzamy mikro-trening adaptera LoRA na nowym materiale (utwory wydane po 2025 r.), z ostrym, losowym podziałem 50/50 na utwory treningowe i held-out wewnątrz dyskografii danego artysty.
  4. Przeprowadzenie *Exact Null Test*  na obu wariantach (referencyjnym oraz dostrojonym przez LoRA) zgodnie z metodologią z literatury bazowej - porównanie odległości wygenerowanego utworu od najbliższego sąsiada w zbiorze treningowym i w zbiorze held-out.
  5. Wykorzystanie korekty wielokrotnego testowania (FDR Benjamini-Hochberg) do odrzucenia przypadkowych podobieństw wynikających jedynie z imitacji stylu, a nie uczenia się na pamięć.
* **Oczekiwany wynik:** Wyznaczenie obiektywnego wskaźnika plagiatu (kopiowania) dla każdego z analizowanych artystów, zabezpieczonego przed błędami I rodzaju (Fałszywe Odkrycia).

### Eksperyment 3: Analiza korelacji (Odpowiedź na RQ1)

* **Cel:** Sprawdzenie hipotezy, czy poziom dekodowalności tożsamości artysty jest predyktorem (napędem) dla kopiowania jego twórczości.
* **Przebieg:**
  1. Zestawienie wyników z Eksperymentu 1 (dokładność sondy, oś X) oraz Eksperymentu 2 (wskaźnik kopiowania, oś Y) dla wszystkich 50 artystów.
  2. Wyliczenie korelacji rang Spearmana z zastosowaniem bootstrapingu do wyznaczenia przedziałów ufności.
  3. Wykonanie analiz kontrolnych wykluczających tzw. zmienne zakłócające (np. sprawdzenie, czy korelacja nie wynika jedynie z rozmiaru reprezentacji artysty w zbiorze treningowym lub specyfiki danego gatunku muzycznego).
* **Oczekiwany wynik:** Statystyczne potwierdzenie (lub odrzucenie) związku między obecnością "Ducha w koderze" a zjawiskiem zapamiętywania danych przez sieć.

### Eksperyment 4: Ingerencja i wymazywanie pojęć metodą LEACE (Odpowiedź na RQ2)

* **Cel:** Ocena wpływu chirurgicznego usunięcia tożsamości artysty na zjawisko plagiatu i na jakość muzyki.
* **Przebieg:**
  1. Zastosowanie metody LEACE (*Linear Concept Erasure*) na aktywacjach warstwy zdiagnozowanej w Eksperymencie 1 jako najsilniej kodującej tożsamość.
  2. Walidacja techniczna (*sanity check*): ponowne wytrenowanie sondy na zmodyfikowanych aktywacjach - oczekiwany spadek dokładności predykcji artysty do poziomu bliskiego losowemu.
  3. Ponowna generacja próbek audio po wymazaniu kierunku i pomiar stopnia imitacji w przestrzeni embeddingów.
  4. Ewaluacja jakości wygenerowanej muzyki za pomocą zautomatyzowanych metryk jakości sygnału (np. FAD - Fréchet Audio Distance, jeśli czas pozwoli) oraz odsłuchu przez członków zespołu badawczego.
* **Oczekiwany wynik:** Zbudowanie krzywej kompromisu (trade-off) imitacja/jakość. Zweryfikowanie, czy faktycznie można uzyskać co najmniej dwukrotny spadek wskaźnika plagiatu bez drastycznej degradacji brzmienia (sukces metody w architekturach audio-generatywnych).

### Eksperyment 5: Odporność wymazywania LEACE na generyczne opisy muzyczne CoT (Odpowiedź na RQ3)

* **Cel:** Ocena, czy automatyczne generowanie opisów stylu muzycznego przez wbudowany w ACE-Step moduł językowy wpływa na skuteczność interwencji metodą LEACE w enkoderze tekstowym.
* **Przebieg:**
  1. Wygenerowanie próbek audio dla badanych artystów z wykorzystaniem metody LEACE w dwóch trybach inferencji:
     * Tryb A: z wyłączonym modułem auto-opisów (*no-CoT*).
     * Tryb B: z włączonym automatycznym tworzeniem opisu muzyki (*with CoT*).
  2. Pomiar spadku imitacji artysty (w przestrzeni embeddingów weryfikacyjnych) oraz ewaluacja jakości brzmienia dla obu trybów.
* **Oczekiwany wynik:** Ustalenie, czy dodatkowa reprezentacja tekstowa generowana przez model "omija" interwencję na enkoderze liryki, czy też metoda LEACE zachowuje swoją skuteczność niezależnie od konfiguracyjnych trybów inferencji.

# Planowana funkcjonalność programu

* **Moduł pozyskiwania i przygotowania danych (Data Loader & Preprocessor):** Automatyczne pobieranie tekstów piosenek przez API (np. Genius), anonimizacja nazwisk artystów w tekście (homophonic replacement), odfiltrowywanie utworów po 2025 roku i generowanie podziałów na zbiory treningowe oraz *held-out*.
* **Potok ekstrakcji aktywacji (Activation Extractor):** Ładowanie wag modelu ACE-Step 1.5, automatyczne wstrzykiwanie hooków w wybrane warstwy kodera tekstu, przepuszczanie danych wejściowych i bezpieczny zapis wyekstrahowanych tensorów aktywacji na dysk/do pamięci.
* **Moduł sondowania i klasyfikacji (Linear Probing Module):** Trenowanie klasyfikatorów na zebranych aktywacjach, przeprowadzanie 5-krotnej walidacji krzyżowej i raportowanie skuteczności (accuracy) rozpoznawania artysty w porównaniu do losowego trafu.
* **Ewaluator zapamiętywania (Exact Memorization Test Suite):** Skrypt przyjmujący jako wejście wygenerowane utwory, utwory treningowe i utwory *held-out*, wyliczający między nimi odległości w przestrzeni embeddingów weryfikacyjnych i przeprowadzający rygorystyczny test statystyczny z kontrolą FDR.
* **Aplikator wymazywania pojęć (LEACE Intervener):** Narzędzie, które na podstawie wyuczonych reprezentacji matematycznie modyfikuje stany ukryte modelu w locie (inference-time intervention), usuwając kierunek tożsamości artysty bez konieczności ponownego trenowania sieci.

# Bibliografia 

* [Ghost in the Encoder: Decodable Artist Identity Representations in Lyrics-to-Song Generation](<https://arxiv.org/abs/2609.39552>) (arXiv:2609.39552)
* [The Null Is the Hard Part: Exact Tests for Memorization in Generative Models](<https://arxiv.org/abs/2610.00251>) (arXiv:2610.00251)
* [Membership and Dataset Inference Attacks on Large Audio Generative Models](<https://arxiv.org/abs/2512.09654>) (arXiv:2512.09654)
* [LEACE: Perfect linear concept erasure in closed form](<https://arxiv.org/abs/2306.03819>) (arXiv:2306.03819)
* [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](<https://arxiv.org/abs/2505.17638>) (arXiv:2505.17638)

# Planowany stack technologiczny

* **Język:** Python `>=3.13` (mogą wystąpić zmiany uwarunkowane kompatybilnością bibliotek)
* **Środowisko wirtualne i zarządzanie pakietami:** `uv` 
* **Oskryptowane budowanie, testowanie i uruchamianie aplikacji:** `make`
* **Jakość kodu:** `ruff` (linter i autoformatter), `ty` (typechecker), kod zgodny z PEP8
* **Kontrola wersji:** Git z rygorem `Conventional Commits`
* **Wykorzystywane biblioteki:** 
  * Modele językowe, wagi modelu i architektury (ACE-Step / Transformer): `PyTorch`, `transformers`, `accelerate`, `diffusers`, `einops`
  * Wymazywanie tożsamości i sondowanie liniowe (LEACE & Probing): `concept-erasure`, `scikit-learn`
  * Przetwarzanie audio i ekstrakcja reprezentacji weryfikacyjnych (Audio & Embeddings): `torchaudio`, `soundfile`, `librosa`, `laion-clap`
  * Dokładne testy statystyczne i False Discovery Rate (Exact Null & FDR): `scipy`, `statsmodels` 
  * Pobieranie danych i teksty utworów: `lyricsgenius`, `requests` / `beautifulsoup4`
* **Infrastruktura**: Środowisko z akceleracją GPU RTX 5070 Ti (CUDA) / RTX 3080 (CUDA) / MPS (Apple Silicon).