from pathlib import Path

from encoder_exorcist.data.audio_reference import audio_path, read_songs, search_query, slugify


def test_slugify_strips_accents_and_punctuation():
    assert slugify("Beyoncé") == "beyonce"
    assert slugify("thank u, next") == "thank_u_next"


def test_audio_path_and_query():
    assert audio_path(Path("audio_reference"), "Taylor Swift", "Blank Space") == Path(
        "audio_reference/taylor_swift_blank_space.wav"
    )
    assert search_query("Taylor Swift", "Blank Space") == "ytsearch1:Taylor Swift - Blank Space audio"


def test_read_songs_deduplicates(tmp_path):
    csv_file = tmp_path / "lyrics.csv"
    csv_file.write_text("genre,artist,title,lyrics\npop,A,Song,x\nrock,A,Song,y\npop,B,Other,z\n", encoding="utf-8")
    assert read_songs([csv_file]) == [("A", "Song"), ("B", "Other")]
