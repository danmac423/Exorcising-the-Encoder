"""Download reference audio for the songs in the lyrics CSV files (YouTube search through yt-dlp, converted to wav).

For every song the first YouTube search result for "<artist> - <title> audio" is downloaded and saved as
<output_dir>/<artist>_<title>.wav with a normalized name, e.g. audio_reference/taylor_swift_blank_space.wav.
The wav conversion uses the ffmpeg binary bundled with imageio-ffmpeg. Songs whose file already exists are skipped, so a rerun continues;
songs that could not be downloaded are listed in <output_dir>/failed_downloads.csv.
"""

import csv
import re
import unicodedata
from pathlib import Path

import imageio_ffmpeg
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from encoder_exorcist.data.lyrics_dataset import PROJECT_ROOT, save_records

FAILED_FIELDS = ["artist", "title", "reason"]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")  # yt-dlp colors its error messages


def slugify(text: str) -> str:
    """Lowercase ascii text with underscores instead of anything else, e.g. "Beyoncé" -> "beyonce"."""
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def audio_path(output_dir: Path, artist: str, title: str) -> Path:
    return output_dir / f"{slugify(artist)}_{slugify(title)}.wav"


def search_query(artist: str, title: str) -> str:
    return f"ytsearch1:{artist} - {title} audio"


def read_songs(csv_paths: list[Path]) -> list[tuple[str, str]]:
    """Unique (artist, title) pairs from the lyrics files, in file order."""
    csv.field_size_limit(10**9)  # lyrics columns are long
    songs: dict[Path, tuple[str, str]] = {}
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                songs.setdefault(audio_path(Path(), row["artist"], row["title"]), (row["artist"], row["title"]))
    return list(songs.values())


def download_audio(artist: str, title: str, target: Path, url: str | None = None) -> None:
    """Downloads the first search result (or `url`, if given) as wav to `target`; raises DownloadError on failure."""
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url or search_query(artist, title)])


def download_one(artist: str, title: str, output_dir: Path, url: str | None = None) -> Path:
    """Downloads a single song (e.g. to retry a failed one); raises DownloadError on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = audio_path(output_dir, artist, title)
    download_audio(artist, title, target, url)
    return target


def download_all(csv_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    songs = read_songs(csv_paths)
    failed: list[dict] = []
    for i, (artist, title) in enumerate(songs, 1):
        target = audio_path(output_dir, artist, title)
        if target.exists():
            print(f"[{i}/{len(songs)}] already have {target.name}")
            continue
        print(f"[{i}/{len(songs)}] {artist} - {title}")
        try:
            download_audio(artist, title, target)
        except DownloadError as error:
            print(f"  failed: {error}")
            failed.append({"artist": artist, "title": title, "reason": ANSI_RE.sub("", str(error))})
    save_records(output_dir / "failed_downloads.csv", FAILED_FIELDS, failed)
    print(f"Done: {len(songs) - len(failed)}/{len(songs)} songs in {output_dir}")


def default_output_dir() -> Path:
    return PROJECT_ROOT / "audio_reference"
