# Song comparison player

[`tools/song_compare.html`](../tools/song_compare.html) plays a generated song next to its original, for quick listening
checks (does the generation imitate the artist, does it copy the original?).

It is a single static page: no server, no install, nothing is uploaded. The browser reads two folders you pick.

## Use

1. Open `tools/song_compare.html` in Chrome, Edge or Firefox (double-click, or drag it into the browser).
2. **Generated** folder: `outputs/generated` (or one configuration, e.g. `outputs/generated/no-cot`). Pick the whole
   folder, so that `manifest.csv` is included: it gives artist, title, genre and, for with-cot, the LM caption.
3. **Originals** folder: the unzipped `audio_reference` (Drive: `WIMU/Audio References`).
4. Click a song. Keys: `1` generated, `2` original, `Space` pause, `J`/`K` next/previous, `←`/`→` ±10 s,
   `H` mark as heard (remembered in this browser only).

With Google Drive for desktop, pick the synced Drive folders directly; nothing needs downloading by hand.

## Matching

A generated file `<artist>_<title>_s<seed>.flac` matches the original `<artist>_<title>.wav`. Both names come from
`slugify` in `encoder_exorcist.data.audio_reference`. Songs whose original failed to download are marked "brak oryg.".
Several seeds or configurations of one song appear as version buttons above the players.
