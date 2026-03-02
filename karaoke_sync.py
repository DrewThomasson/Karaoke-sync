#!/usr/bin/env python3
"""
karaoke_sync.py - Automatically generate synchronized .lrc karaoke files
from MP3 files with embedded unsynchronized lyrics (USLT ID3 frames).

Workflow per MP3:
  1. Skip if a .lrc file already exists alongside it.
  2. Extract lyrics from the USLT ID3 frame via mutagen.
  3. Sanitize the lyrics (remove structural labels, empty lines).
  4. Isolate vocals with demucs (htdemucs model) into a temp WAV.
  5. Perform forced alignment with whisperx to produce word/line timestamps.
  6. Write a standard [mm:ss.xx] LRC file next to the MP3.
  7. Delete the temporary vocal WAV.

Usage:
    python karaoke_sync.py /path/to/music/library
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily / with informative errors
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from mutagen.id3 import ID3
except ImportError:
    print("Error: mutagen is required. Install with: pip install mutagen", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return the best available compute device string for PyTorch models."""
    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Lyric extraction
# ---------------------------------------------------------------------------


def extract_lyrics(mp3_path: Path) -> str | None:
    """
    Read the first USLT (unsynchronised lyrics) ID3 frame from an MP3.

    Returns the raw text string, or None if no lyrics are found or the file
    cannot be read.
    """
    try:
        tags = ID3(str(mp3_path))
    except Exception as exc:
        print(f"  [ERROR] Cannot read ID3 tags from '{mp3_path.name}': {exc}")
        return None

    for key in tags.keys():
        if key.startswith("USLT"):
            uslt = tags[key]
            text: str = uslt.text
            if text and text.strip():
                return text

    return None


# ---------------------------------------------------------------------------
# Lyric sanitization
# ---------------------------------------------------------------------------

# Matches entire lines that are structural labels, e.g.  [Chorus]  (Live)
_LABEL_LINE_RE = re.compile(r"^(\[[^\]]*\]|\([^\)]*\))$")
# Matches inline structural labels anywhere on a line
_INLINE_LABEL_RE = re.compile(r"\[[^\]]*\]|\([^\)]*\)")


def sanitize_lyrics(raw: str) -> list[str]:
    """
    Clean raw lyric text:
      - Lines that consist entirely of a bracketed/parenthesised label are dropped.
      - Inline labels are removed from lines that also contain lyric text.
      - Blank / whitespace-only lines are dropped.

    Returns an ordered list of non-empty lyric line strings.
    """
    result: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _LABEL_LINE_RE.match(stripped):
            continue
        cleaned = _INLINE_LABEL_RE.sub("", stripped).strip()
        if cleaned:
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Vocal separation
# ---------------------------------------------------------------------------


def separate_vocals(mp3_path: Path, temp_dir: Path) -> Path | None:
    """
    Run demucs (htdemucs, two-stems=vocals) on *mp3_path*, placing output
    under *temp_dir*.

    Returns the path to the isolated 'vocals.wav', or None on failure.
    """
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "-o",
        str(temp_dir),
        str(mp3_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Demucs failed for '{mp3_path.name}': {exc.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("  [ERROR] Demucs is not installed or not importable via 'python -m demucs'.")
        return None

    vocals_wav = temp_dir / "htdemucs" / mp3_path.stem / "vocals.wav"
    if not vocals_wav.exists():
        print(f"  [ERROR] Expected Demucs output not found: {vocals_wav}")
        return None

    return vocals_wav


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------


def _compute_type_for_device(device: str) -> str:
    """Choose an appropriate whisperx compute type for the given device."""
    return "float16" if device == "cuda" else "float32"


def align_lyrics(
    vocals_path: Path,
    lyrics_lines: list[str],
    device: str,
) -> list[tuple[float, str]] | None:
    """
    Perform forced alignment of *lyrics_lines* against the audio at
    *vocals_path* using whisperx.

    Strategy:
      1. Load the audio and transcribe with a small Whisper model to detect
         the language.
      2. Construct alignment segments from the known lyric lines, distributed
         evenly across the audio duration as initial timestamp estimates.
      3. Run whisperx.align() so the phoneme-level aligner refines the
         timestamps against the actual audio signal.
      4. Return a list of (start_seconds, line_text) tuples.

    Returns None if alignment fails for any reason.
    """
    try:
        import whisperx  # noqa: PLC0415 – imported here to keep startup fast
    except ImportError:
        print(
            "  [ERROR] whisperx is not installed. "
            "Install with: pip install git+https://github.com/m-bain/whisperX.git",
            file=sys.stderr,
        )
        return None

    try:
        compute_type = _compute_type_for_device(device)

        # Load audio (whisperx uses 16 kHz internally)
        audio = whisperx.load_audio(str(vocals_path))
        audio_duration: float = len(audio) / 16_000.0

        if audio_duration <= 0:
            print("  [ERROR] Vocals audio file appears to be empty.")
            return None

        # ------------------------------------------------------------------
        # Step 1 – Transcribe to detect language
        # ------------------------------------------------------------------
        asr_model = whisperx.load_model("base", device, compute_type=compute_type)
        transcription = asr_model.transcribe(audio, batch_size=16)
        language: str = transcription.get("language", "en")
        print(f"  Detected language: {language}")

        # ------------------------------------------------------------------
        # Step 2 – Build forced-alignment segments from our known lyrics
        # ------------------------------------------------------------------
        n = len(lyrics_lines)
        seg_dur = audio_duration / n
        segments = [
            {
                "text": line,
                "start": i * seg_dur,
                "end": (i + 1) * seg_dur,
            }
            for i, line in enumerate(lyrics_lines)
        ]

        # ------------------------------------------------------------------
        # Step 3 – Load alignment model and align
        # ------------------------------------------------------------------
        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        result = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        # ------------------------------------------------------------------
        # Step 4 – Collect line-level timestamps
        # ------------------------------------------------------------------
        timestamps: list[tuple[float, str]] = []
        for seg in result.get("segments", []):
            start = seg.get("start")
            text = seg.get("text", "").strip()
            if start is not None and text:
                timestamps.append((float(start), text))

        return timestamps if timestamps else None

    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] Alignment failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# LRC helpers
# ---------------------------------------------------------------------------


def format_lrc_timestamp(seconds: float) -> str:
    """
    Convert a duration in seconds to an LRC timestamp string.

    Format: [mm:ss.xx]  (centiseconds, zero-padded)
    """
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    secs = int(remaining)
    centisecs = round((remaining - secs) * 100)
    # Handle edge case where rounding pushes centisecs to 100
    if centisecs >= 100:
        secs += 1
        centisecs -= 100
        if secs >= 60:
            minutes += 1
            secs -= 60
    return f"[{minutes:02d}:{secs:02d}.{centisecs:02d}]"


def write_lrc_file(lrc_path: Path, timestamps: list[tuple[float, str]]) -> None:
    """Write *timestamps* to *lrc_path* in standard LRC format (UTF-8)."""
    with open(lrc_path, "w", encoding="utf-8") as fh:
        for seconds, line in sorted(timestamps, key=lambda x: x[0]):
            fh.write(f"{format_lrc_timestamp(seconds)} {line}\n")


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_mp3(mp3_path: Path, device: str) -> None:
    """
    Run the full karaoke-sync pipeline for a single MP3 file.

    All exceptions are caught internally; errors are printed and execution
    continues with the next file.
    """
    lrc_path = mp3_path.with_suffix(".lrc")

    if lrc_path.exists():
        print(f"  [SKIP] LRC already exists: {lrc_path}")
        return

    print(f"\nProcessing: {mp3_path}")

    # --- 1. Extract lyrics ------------------------------------------------
    raw_lyrics = extract_lyrics(mp3_path)
    if not raw_lyrics:
        print(f"  [SKIP] No embedded lyrics found in '{mp3_path.name}'.")
        return

    # --- 2. Sanitize lyrics -----------------------------------------------
    lyrics_lines = sanitize_lyrics(raw_lyrics)
    if not lyrics_lines:
        print(f"  [SKIP] No lyric lines remain after sanitization for '{mp3_path.name}'.")
        return

    print(f"  Extracted {len(lyrics_lines)} lyric line(s).")

    # --- 3. Separate vocals & align (temp dir auto-cleaned on exit) --------
    with tempfile.TemporaryDirectory(prefix="karaoke_sync_") as tmp:
        temp_dir = Path(tmp)

        print("  Separating vocals with Demucs (htdemucs)…")
        vocals_path = separate_vocals(mp3_path, temp_dir)
        if vocals_path is None:
            print(f"  [ERROR] Vocal separation failed for '{mp3_path.name}'.")
            return

        print("  Aligning lyrics with WhisperX…")
        timestamps = align_lyrics(vocals_path, lyrics_lines, device)
        # temp_dir (and vocals WAV) are deleted when the 'with' block exits

    if timestamps is None:
        print(f"  [ERROR] Alignment failed for '{mp3_path.name}'.")
        return

    # --- 4. Write LRC file ------------------------------------------------
    write_lrc_file(lrc_path, timestamps)
    print(f"  [OK] Wrote LRC file: {lrc_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan a directory for MP3 files, extract embedded "
            "unsynchronised lyrics (USLT), align them to the audio with "
            "demucs + whisperx, and write standard .lrc karaoke files."
        )
    )
    parser.add_argument(
        "directory",
        help="Root directory to scan for MP3 files.",
    )
    args = parser.parse_args()

    music_dir = Path(args.directory)
    if not music_dir.is_dir():
        parser.error(f"'{music_dir}' is not an existing directory.")

    device = get_device()
    print(f"Compute device: {device}")

    mp3_files = sorted(music_dir.rglob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found under '{music_dir}'.")
        return

    print(f"Found {len(mp3_files)} MP3 file(s).\n")

    iterator = (
        _tqdm(mp3_files, desc="Processing MP3s", unit="file")
        if _tqdm is not None
        else mp3_files
    )

    for mp3_path in iterator:
        try:
            process_mp3(mp3_path, device)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Unexpected error processing '{mp3_path.name}': {exc}")


if __name__ == "__main__":
    main()
