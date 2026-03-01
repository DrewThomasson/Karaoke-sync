"""
Unit tests for karaoke_sync.py.

Heavy dependencies (torch, whisperx, demucs, mutagen) are mocked so the
tests run without requiring a GPU or a full ML install.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers: inject lightweight stubs for heavy optional dependencies so that
# karaoke_sync can be imported in a minimal CI environment.
# ---------------------------------------------------------------------------


def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_torch_stub() -> None:
    """Provide a minimal torch stub if the real package is absent."""
    if "torch" in sys.modules:
        return
    cuda_stub = types.SimpleNamespace(is_available=lambda: False)
    mps_stub = types.SimpleNamespace(is_available=lambda: False)
    backends_stub = types.SimpleNamespace(mps=mps_stub)
    torch_stub = _make_stub_module(
        "torch",
        cuda=cuda_stub,
        backends=backends_stub,
    )
    sys.modules["torch"] = torch_stub


def _install_mutagen_stub() -> None:
    """Provide a minimal mutagen stub if the real package is absent."""
    if "mutagen" in sys.modules:
        return
    mutagen_stub = _make_stub_module("mutagen")
    id3_stub = _make_stub_module("mutagen.id3", ID3=mock.MagicMock(), USLT=mock.MagicMock())
    sys.modules["mutagen"] = mutagen_stub
    sys.modules["mutagen.id3"] = id3_stub


_install_torch_stub()
_install_mutagen_stub()

# Force a fresh import of the module under test (using stubs where needed).
if "karaoke_sync" in sys.modules:
    del sys.modules["karaoke_sync"]

import karaoke_sync  # noqa: E402


# ---------------------------------------------------------------------------
# sanitize_lyrics
# ---------------------------------------------------------------------------


class TestSanitizeLyrics:
    def test_removes_bracket_label_lines(self):
        raw = "[Chorus]\nHello world\n[Verse 1]\nFoo bar"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["Hello world", "Foo bar"]

    def test_removes_parenthesis_label_lines(self):
        raw = "(Live)\nSome lyric\n(Instrumental)"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["Some lyric"]

    def test_removes_inline_labels(self):
        raw = "Hello [Ad-lib] world"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["Hello  world"]

    def test_removes_empty_lines(self):
        raw = "Line one\n\n\nLine two\n   \nLine three"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["Line one", "Line two", "Line three"]

    def test_mixed_content(self):
        raw = "[Intro]\n\nFirst line\n[Bridge]\nSecond line\n(outro)"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["First line", "Second line"]

    def test_empty_string_returns_empty_list(self):
        assert karaoke_sync.sanitize_lyrics("") == []

    def test_only_labels_returns_empty_list(self):
        raw = "[Chorus]\n[Verse]\n(Outro)"
        assert karaoke_sync.sanitize_lyrics(raw) == []

    def test_preserves_valid_lines(self):
        raw = "I'm walking on sunshine\nOh oh oh"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["I'm walking on sunshine", "Oh oh oh"]

    def test_strips_surrounding_whitespace(self):
        raw = "  Hello world  \n  Foo bar  "
        result = karaoke_sync.sanitize_lyrics(raw)
        assert result == ["Hello world", "Foo bar"]

    def test_does_not_remove_mismatched_brackets_as_full_label(self):
        # "[text)" and "(text]" are NOT valid structural labels and should be kept
        raw = "[text)\n(text]\nReal line"
        result = karaoke_sync.sanitize_lyrics(raw)
        assert "Real line" in result
        # Mismatched lines are kept (not treated as pure labels)
        assert any("[text)" in r or "text]" in r for r in result)


# ---------------------------------------------------------------------------
# format_lrc_timestamp
# ---------------------------------------------------------------------------


class TestFormatLrcTimestamp:
    def test_zero(self):
        assert karaoke_sync.format_lrc_timestamp(0.0) == "[00:00.00]"

    def test_whole_seconds(self):
        assert karaoke_sync.format_lrc_timestamp(1.0) == "[00:01.00]"

    def test_centiseconds(self):
        assert karaoke_sync.format_lrc_timestamp(1.5) == "[00:01.50]"

    def test_minutes(self):
        assert karaoke_sync.format_lrc_timestamp(65.0) == "[01:05.00]"

    def test_fractional_centiseconds_rounding(self):
        ts = karaoke_sync.format_lrc_timestamp(1.999)
        # Should round to [00:02.00] not [00:01.99]
        assert ts == "[00:02.00]"

    def test_large_value(self):
        # 3661 seconds = 61 min 01 sec
        assert karaoke_sync.format_lrc_timestamp(3661.0) == "[61:01.00]"

    def test_output_format_pattern(self):
        import re

        ts = karaoke_sync.format_lrc_timestamp(123.45)
        assert re.match(r"^\[\d{2}:\d{2}\.\d{2}\]$", ts), f"Bad format: {ts}"


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


class TestGetDevice:
    def test_cuda_preferred_when_available(self):
        cuda_stub = types.SimpleNamespace(is_available=lambda: True)
        torch_stub = _make_stub_module("torch", cuda=cuda_stub, backends=types.SimpleNamespace())
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            importlib.reload(karaoke_sync)
            assert karaoke_sync.get_device() == "cuda"

    def test_mps_used_when_cuda_unavailable(self):
        cuda_stub = types.SimpleNamespace(is_available=lambda: False)
        mps_stub = types.SimpleNamespace(is_available=lambda: True)
        backends_stub = types.SimpleNamespace(mps=mps_stub)
        torch_stub = _make_stub_module("torch", cuda=cuda_stub, backends=backends_stub)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            importlib.reload(karaoke_sync)
            assert karaoke_sync.get_device() == "mps"

    def test_cpu_when_no_gpu(self):
        cuda_stub = types.SimpleNamespace(is_available=lambda: False)
        mps_stub = types.SimpleNamespace(is_available=lambda: False)
        backends_stub = types.SimpleNamespace(mps=mps_stub)
        torch_stub = _make_stub_module("torch", cuda=cuda_stub, backends=backends_stub)
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            importlib.reload(karaoke_sync)
            assert karaoke_sync.get_device() == "cpu"

    def test_cpu_when_torch_not_installed(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            importlib.reload(karaoke_sync)
            assert karaoke_sync.get_device() == "cpu"


# ---------------------------------------------------------------------------
# write_lrc_file
# ---------------------------------------------------------------------------


class TestWriteLrcFile:
    def test_writes_sorted_lines(self, tmp_path):
        lrc_path = tmp_path / "song.lrc"
        timestamps = [(65.5, "Second line"), (0.0, "First line"), (130.25, "Third line")]
        karaoke_sync.write_lrc_file(lrc_path, timestamps)

        content = lrc_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        assert lines[0] == "[00:00.00] First line"
        assert lines[1] == "[01:05.50] Second line"
        assert lines[2] == "[02:10.25] Third line"

    def test_file_encoding_is_utf8(self, tmp_path):
        lrc_path = tmp_path / "song.lrc"
        karaoke_sync.write_lrc_file(lrc_path, [(0.0, "日本語の歌詞")])
        content = lrc_path.read_text(encoding="utf-8")
        assert "日本語の歌詞" in content

    def test_empty_timestamps_creates_empty_file(self, tmp_path):
        lrc_path = tmp_path / "song.lrc"
        karaoke_sync.write_lrc_file(lrc_path, [])
        assert lrc_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# extract_lyrics
# ---------------------------------------------------------------------------


class TestExtractLyrics:
    def test_returns_none_when_no_uslt_tag(self, tmp_path):
        fake_mp3 = tmp_path / "song.mp3"
        fake_mp3.write_bytes(b"")

        with mock.patch("karaoke_sync.ID3") as MockID3:
            MockID3.return_value = {}  # no USLT keys
            result = karaoke_sync.extract_lyrics(fake_mp3)

        assert result is None

    def test_returns_text_from_uslt_frame(self, tmp_path):
        fake_mp3 = tmp_path / "song.mp3"
        fake_mp3.write_bytes(b"")

        uslt_frame = mock.MagicMock()
        uslt_frame.text = "Hello\nWorld"
        tags = {"USLT::eng": uslt_frame}

        with mock.patch("karaoke_sync.ID3") as MockID3:
            MockID3.return_value = tags
            result = karaoke_sync.extract_lyrics(fake_mp3)

        assert result == "Hello\nWorld"

    def test_returns_none_when_uslt_text_is_empty(self, tmp_path):
        fake_mp3 = tmp_path / "song.mp3"
        fake_mp3.write_bytes(b"")

        uslt_frame = mock.MagicMock()
        uslt_frame.text = "   "
        tags = {"USLT::eng": uslt_frame}

        with mock.patch("karaoke_sync.ID3") as MockID3:
            MockID3.return_value = tags
            result = karaoke_sync.extract_lyrics(fake_mp3)

        assert result is None

    def test_returns_none_on_id3_exception(self, tmp_path, capsys):
        fake_mp3 = tmp_path / "corrupt.mp3"
        fake_mp3.write_bytes(b"")

        with mock.patch("karaoke_sync.ID3", side_effect=Exception("bad file")):
            result = karaoke_sync.extract_lyrics(fake_mp3)

        assert result is None
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out


# ---------------------------------------------------------------------------
# process_mp3 – integration-level (heavy deps fully mocked)
# ---------------------------------------------------------------------------


class TestProcessMp3:
    def _make_mp3(self, tmp_path: Path, name: str = "song.mp3") -> Path:
        p = tmp_path / name
        p.write_bytes(b"")
        return p

    def test_skips_when_lrc_exists(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)
        lrc = mp3.with_suffix(".lrc")
        lrc.write_text("[00:00.00] Already here\n", encoding="utf-8")

        karaoke_sync.process_mp3(mp3, "cpu")

        captured = capsys.readouterr()
        assert "[SKIP]" in captured.out
        # LRC should not be overwritten
        assert "[00:00.00] Already here" in lrc.read_text(encoding="utf-8")

    def test_skips_when_no_lyrics(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)

        with mock.patch("karaoke_sync.extract_lyrics", return_value=None):
            karaoke_sync.process_mp3(mp3, "cpu")

        captured = capsys.readouterr()
        assert "[SKIP]" in captured.out
        assert not mp3.with_suffix(".lrc").exists()

    def test_skips_when_lyrics_all_sanitized_away(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)

        with mock.patch("karaoke_sync.extract_lyrics", return_value="[Chorus]"):
            karaoke_sync.process_mp3(mp3, "cpu")

        captured = capsys.readouterr()
        assert "[SKIP]" in captured.out
        assert not mp3.with_suffix(".lrc").exists()

    def test_error_logged_when_vocal_separation_fails(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)

        with (
            mock.patch("karaoke_sync.extract_lyrics", return_value="Hello world"),
            mock.patch("karaoke_sync.separate_vocals", return_value=None),
        ):
            karaoke_sync.process_mp3(mp3, "cpu")

        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert not mp3.with_suffix(".lrc").exists()

    def test_error_logged_when_alignment_fails(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)
        fake_vocals = tmp_path / "vocals.wav"
        fake_vocals.write_bytes(b"")

        with (
            mock.patch("karaoke_sync.extract_lyrics", return_value="Hello world"),
            mock.patch("karaoke_sync.separate_vocals", return_value=fake_vocals),
            mock.patch("karaoke_sync.align_lyrics", return_value=None),
        ):
            karaoke_sync.process_mp3(mp3, "cpu")

        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert not mp3.with_suffix(".lrc").exists()

    def test_full_success_writes_lrc(self, tmp_path, capsys):
        mp3 = self._make_mp3(tmp_path)
        fake_vocals = tmp_path / "vocals.wav"
        fake_vocals.write_bytes(b"")

        timestamps = [(0.0, "Hello world"), (5.0, "Foo bar")]

        with (
            mock.patch("karaoke_sync.extract_lyrics", return_value="Hello world\nFoo bar"),
            mock.patch("karaoke_sync.separate_vocals", return_value=fake_vocals),
            mock.patch("karaoke_sync.align_lyrics", return_value=timestamps),
        ):
            karaoke_sync.process_mp3(mp3, "cpu")

        lrc = mp3.with_suffix(".lrc")
        assert lrc.exists()
        content = lrc.read_text(encoding="utf-8")
        assert "[00:00.00] Hello world" in content
        assert "[00:05.00] Foo bar" in content
