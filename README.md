# Karaoke-sync

Automatically generate synchronized `.lrc` karaoke files from MP3s that contain embedded unsynchronised lyrics (USLT ID3 frames).

---

## How it works

For every MP3 file found in a directory tree, `karaoke_sync.py` runs the following pipeline:

1. **Skip** – if a `.lrc` file already exists alongside the MP3, nothing is done.
2. **Extract lyrics** – reads the `USLT` (unsynchronised lyrics) ID3 frame from the MP3 using [mutagen](https://mutagen.readthedocs.io/).
3. **Sanitize lyrics** – strips structural labels (e.g. `[Chorus]`, `(Instrumental)`), inline annotations, and blank lines.
4. **Separate vocals** – isolates the vocal track using [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs` model, two-stem mode) and saves it to a temporary WAV file.
5. **Forced alignment** – aligns the known lyric lines to the vocal audio using [WhisperX](https://github.com/m-bain/whisperX) to produce precise line-level timestamps.
6. **Write LRC** – saves a standard `[mm:ss.xx]` LRC file next to the MP3.
7. **Clean up** – deletes the temporary vocal WAV automatically.

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| [mutagen](https://mutagen.readthedocs.io/) ≥ 1.47 | Read ID3 USLT lyrics from MP3s |
| [demucs](https://github.com/facebookresearch/demucs) ≥ 4.0 | Vocal source separation |
| [whisperx](https://github.com/m-bain/whisperX) ≥ 3.1.1 | Forced phoneme alignment |
| [torch](https://pytorch.org/) ≥ 2.0 + torchaudio ≥ 2.0 | Deep-learning backend |
| [tqdm](https://github.com/tqdm/tqdm) ≥ 4.65 | Progress bar (optional but recommended) |
| [numpy](https://numpy.org/) ≥ 1.24 | Numerical processing |

**GPU support** is strongly recommended for acceptable performance. CUDA (NVIDIA) and MPS (Apple Silicon) are detected and used automatically when available.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/DrewThomasson/Karaoke-sync.git
cd Karaoke-sync

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install WhisperX from source (required for latest features)
pip install git+https://github.com/m-bain/whisperX.git

# 4. (Optional) Install a CUDA-enabled PyTorch wheel for GPU acceleration
#    See https://pytorch.org/get-started/locally/ for the correct command for your system.
```

---

## Usage

```bash
python karaoke_sync.py /path/to/music/library
```

The script recursively scans the given directory for `*.mp3` files and processes each one. Files that already have a matching `.lrc` are skipped.

### Example

```
$ python karaoke_sync.py ~/Music

Compute device: cuda
Found 42 MP3 file(s).

Processing: ~/Music/Artist/Album/01 - Song Title.mp3
  Extracted 32 lyric line(s).
  Separating vocals with Demucs (htdemucs)…
  Aligning lyrics with WhisperX…
  Detected language: en
  [OK] Wrote LRC file: ~/Music/Artist/Album/01 - Song Title.lrc
...
```

---

## Output format

Generated `.lrc` files use the standard LRC timestamp format `[mm:ss.xx]` (centisecond precision), UTF-8 encoded:

```
[00:12.34] First line of the song
[00:16.80] Second line of the song
[00:21.05] And so on…
```

---

## Running the tests

```bash
pip install pytest
pytest tests/
```

The unit tests mock all heavy ML dependencies so they run quickly without a GPU or a full model install.
