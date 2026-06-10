# YouTube Transcript Summarizer & Note Maker
### CUDA-Accelerated NLP on NVIDIA RTX GPU

> Paste a YouTube URL → get structured Markdown / PDF / DOCX notes in seconds,  
> powered by BART/T5 transformer models running on your NVIDIA RTX GPU via CUDA.

---

## Features

| Feature | Detail |
|---|---|
| Transcript Extraction | YouTube Transcript API → Whisper (GPU fallback) |
| Summarization | BART-large-CNN / T5 via HuggingFace, runs on CUDA |
| Keyword Extraction | TF-IDF + spaCy NER (people, orgs, places, products) |
| Note Structuring | Auto-sectioned Markdown with headings + bullets |
| Export | `.md` + `.pdf` + `.docx` in one command |
| Batch Mode | Process many URLs from a text file |

---

## Requirements

| Component | Minimum |
|---|---|
| OS | Windows 11 / Ubuntu 22.04 |
| Python | 3.10+ |
| GPU | NVIDIA RTX 2060 or better |
| VRAM | 6 GB (BART-large) · 2 GB (T5-base) · 1 GB (T5-small) |
| CUDA | 11.8 or 12.x |
| RAM | 16 GB |

---

## Setup

### 1 — Clone / download the project
```bash
cd yt_summarizer
```

### 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3 — Install CUDA PyTorch first
```bash
# CUDA 11.8  (RTX 30xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  (RTX 40xx series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4 — Install all other dependencies
```bash
pip install -r requirements.txt
```

### 5 — Download the spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### 6 — (Optional) Install FFmpeg for Whisper audio fallback
- **Windows:** https://ffmpeg.org/download.html — add to PATH  
- **Ubuntu:** `sudo apt install ffmpeg`

---

## Usage

### Single video
```bash
python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Specify export formats (md only, or pdf+docx, etc.)
```bash
python main.py --url "https://youtu.be/dQw4w9WgXcQ" --formats md pdf
```

### Use a faster / lighter model (less VRAM)
```bash
python main.py --url "https://youtu.be/dQw4w9WgXcQ" --model t5-base
python main.py --url "https://youtu.be/dQw4w9WgXcQ" --model t5-small
```

### Print to terminal only (no files saved)
```bash
python main.py --url "https://youtu.be/dQw4w9WgXcQ" --no-export
```

### Batch mode (one URL per line in a .txt file)
```bash
python main.py --batch urls.txt
```
`urls.txt` example:
```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
# lines starting with # are ignored
```

---

## Project Structure

```
yt_summarizer/
├── main.py                  # CLI entry point — run this
├── config.py                # All settings (model, device, paths)
├── requirements.txt
├── README.md
├── outputs/                 # Generated notes (.md / .pdf / .docx)
├── models_cache/            # HuggingFace model cache (auto-populated)
└── src/
    ├── transcript_fetcher.py  # YouTube API + Whisper fallback
    ├── summarizer.py          # CUDA-accelerated BART/T5 summarization
    ├── keyword_extractor.py   # TF-IDF + spaCy NER
    ├── note_generator.py      # Structures notes into Markdown
    └── exporter.py            # Exports to .md / .pdf / .docx
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `SUMMARIZATION_MODEL` | `facebook/bart-large-cnn` | HuggingFace model ID |
| `WHISPER_MODEL` | `base` | Whisper ASR model size |
| `CHUNK_TOKEN_LIMIT` | `1024` | Max tokens per summarization chunk |
| `SUMMARY_MIN_LENGTH` | `60` | Min tokens in each summary |
| `SUMMARY_MAX_LENGTH` | `200` | Max tokens in each summary |
| `TOP_KEYWORDS` | `10` | Number of TF-IDF keywords to extract |

---

## How It Works

```
YouTube URL
    │
    ▼
[1] Transcript Fetcher
    ├─ youtube-transcript-api  (fast, no download)
    └─ Whisper on CUDA          (fallback for uncaptioned videos)
    │
    ▼
[2] Summarizer (CUDA)
    ├─ Split transcript into 1024-token chunks
    ├─ Load BART/T5 model → .to("cuda")
    └─ torch.autocast(fp16) batch inference
    │
    ▼
[3] Keyword Extractor
    ├─ TF-IDF scoring (pure Python, no sklearn)
    └─ spaCy NER  (PERSON, ORG, GPE, PRODUCT …)
    │
    ▼
[4] Note Generator
    └─ Auto-sectioned Markdown (Introduction → Conclusion)
    │
    ▼
[5] Exporter
    ├─ .md   (Markdown)
    ├─ .pdf  (fpdf2, dark-themed)
    └─ .docx (python-docx)
```

---

## Example Output

```markdown
# Deep Learning Explained — dQw4w9WgXcQ
**Source:** https://youtu.be/dQw4w9WgXcQ
**Generated:** 2025-01-15 14:32

## Key Terms & Entities
`neural network`  `backpropagation`  `gradient descent`  `PyTorch`

## Named Entities
- **Yann LeCun**
- **Google DeepMind**

## Notes

### Section 1 — Introduction
- Deep learning is a subfield of machine learning using layered neural networks.
- It has revolutionised computer vision, NLP, and speech recognition.

### Section 2 — Core Concepts
- Backpropagation computes gradients efficiently using the chain rule.
- Gradient descent iteratively updates weights to minimise the loss.
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Switch to `--model t5-small` or `--model t5-base` |
| `No captions found` | Install FFmpeg; Whisper fallback will handle it |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| `yt-dlp error` | Update: `pip install -U yt-dlp` |
| CPU used instead of GPU | Check `torch.cuda.is_available()` — reinstall CUDA PyTorch |

---
