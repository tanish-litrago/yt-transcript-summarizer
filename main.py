"""
main.py
───────
YouTube Transcript Summarizer & Note Maker
CUDA-accelerated via NVIDIA RTX GPU + HuggingFace Transformers

Usage:
    python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    python main.py --url "..." --formats md pdf docx
    python main.py --url "..." --model t5-base
    python main.py --url "..." --no-export   (print notes to terminal only)

Run  python main.py --help  for full options.
"""

import argparse
import sys
import os
import time

import torch
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.transcript_fetcher  import fetch_transcript
from src.summarizer          import summarize_transcript
from src.keyword_extractor   import extract_keywords
from src.note_generator      import generate_notes
from src.exporter            import export_all


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║   YouTube Transcript Summarizer & Note Maker             ║
║   CUDA-Accelerated NLP  |  HuggingFace Transformers      ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""


def print_gpu_info():
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{Fore.GREEN}✔  GPU detected : {name}  ({vram:.1f} GB VRAM){Style.RESET_ALL}")
        print(f"{Fore.GREEN}✔  CUDA version : {torch.version.cuda}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠  No CUDA GPU detected — running on CPU (slower).{Style.RESET_ALL}")
        print(f"   To enable GPU acceleration, install the CUDA version of PyTorch:")
        print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu118")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="YouTube Transcript Summarizer & Note Maker (CUDA-accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --url "https://youtu.be/dQw4w9WgXcQ"
  python main.py --url "https://youtu.be/dQw4w9WgXcQ" --formats md pdf
  python main.py --url "https://youtu.be/dQw4w9WgXcQ" --model t5-base --no-export
  python main.py --batch urls.txt
        """,
    )

    parser.add_argument(
        "--url", "-u",
        type=str,
        help="YouTube video URL to summarise.",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        metavar="FILE",
        help="Path to a text file with one YouTube URL per line (batch mode).",
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="",
        help="Optional video title to use as the notes heading.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=(
            "HuggingFace model to use for summarisation. "
            "Default: as set in config.py. "
            "Options: facebook/bart-large-cnn | t5-base | t5-small"
        ),
    )
    parser.add_argument(
        "--formats", "-f",
        nargs="+",
        choices=["md", "pdf", "docx"],
        default=["md", "pdf", "docx"],
        help="Output formats to export (default: md pdf docx).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Print notes to terminal only — do not write any files.",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Preferred transcript language code (default: en).",
    )

    return parser.parse_args()


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(url: str, args) -> dict:
    """
    Runs the full pipeline for a single URL:
      fetch → summarise → extract keywords → generate notes → export

    Returns a dict with all produced artefacts.
    """
    t0 = time.time()

    # Override model from CLI if provided
    if args.model:
        cfg.SUMMARIZATION_MODEL = args.model

    print(f"\n{Fore.CYAN}{'─'*60}")
    print(f" Processing: {url}")
    print(f"{'─'*60}{Style.RESET_ALL}\n")

    # 1. Fetch transcript ──────────────────────────────────────────────────────
    print(f"{Fore.YELLOW}[1/4] Fetching transcript …{Style.RESET_ALL}")
    transcript_data = fetch_transcript(url, preferred_language=args.language)
    transcript      = transcript_data["transcript"]
    video_id        = transcript_data["video_id"]
    word_count      = len(transcript.split())
    print(f"      ✔ {word_count:,} words fetched  "
          f"(source: {transcript_data['source']}  |  lang: {transcript_data['language']})")

    # 2. Summarise ─────────────────────────────────────────────────────────────
    print(f"\n{Fore.YELLOW}[2/4] Summarising transcript on {cfg.DEVICE.upper()} …{Style.RESET_ALL}")
    summary_data = summarize_transcript(transcript)
    print(f"      ✔ {summary_data['num_chunks']} chunk(s) summarised  "
          f"({len(summary_data['full_summary'].split())} words in summary)")

    # 3. Extract keywords ──────────────────────────────────────────────────────
    print(f"\n{Fore.YELLOW}[3/4] Extracting keywords & entities …{Style.RESET_ALL}")
    keyword_data = extract_keywords(transcript)
    combined_kw  = keyword_data["combined"]
    print(f"      ✔ Keywords: {', '.join(combined_kw[:8])}{'…' if len(combined_kw) > 8 else ''}")

    # 4. Generate notes ────────────────────────────────────────────────────────
    print(f"\n{Fore.YELLOW}[4/4] Generating structured notes …{Style.RESET_ALL}")
    notes_md = generate_notes(
        video_url    = url,
        video_id     = video_id,
        summary_data = summary_data,
        keyword_data = keyword_data,
        title        = args.title,
    )

    # 5. Export ────────────────────────────────────────────────────────────────
    output_paths = {}
    if args.no_export:
        print(f"\n{Fore.CYAN}{'─'*60}\n NOTES OUTPUT\n{'─'*60}{Style.RESET_ALL}\n")
        print(notes_md)
    else:
        print(f"\n{Fore.YELLOW}[Export] Writing output files …{Style.RESET_ALL}")
        output_paths = export_all(notes_md, video_id, formats=args.formats)

    elapsed = time.time() - t0
    print(f"\n{Fore.GREEN}✔  Pipeline complete in {elapsed:.1f}s{Style.RESET_ALL}")

    return {
        "video_id"    : video_id,
        "transcript"  : transcript_data,
        "summary"     : summary_data,
        "keywords"    : keyword_data,
        "notes_md"    : notes_md,
        "output_paths": output_paths,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(BANNER)
    print_gpu_info()
    print()

    args = parse_args()

    # Validate input
    if not args.url and not args.batch:
        print(f"{Fore.RED}Error: provide --url or --batch.{Style.RESET_ALL}")
        print("Run  python main.py --help  for usage.")
        sys.exit(1)

    urls = []
    if args.url:
        urls.append(args.url.strip())
    if args.batch:
        if not os.path.isfile(args.batch):
            print(f"{Fore.RED}Error: batch file not found: {args.batch}{Style.RESET_ALL}")
            sys.exit(1)
        with open(args.batch, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        print(f"Batch mode: {len(urls)} URL(s) loaded from {args.batch}")

    results = []
    for url in urls:
        try:
            result = run_pipeline(url, args)
            results.append(result)
        except Exception as exc:
            print(f"{Fore.RED}✘  Failed for {url}: {exc}{Style.RESET_ALL}")
            if len(urls) == 1:
                raise

    # Summary table for batch mode
    if len(urls) > 1:
        print(f"\n{Fore.CYAN}{'─'*60}")
        print(" Batch Summary")
        print(f"{'─'*60}{Style.RESET_ALL}")
        for r in results:
            vid = r["video_id"]
            kw  = ", ".join(r["keywords"]["combined"][:4])
            print(f"  {Fore.GREEN}✔{Style.RESET_ALL}  {vid}  |  keywords: {kw}")
        print()


if __name__ == "__main__":
    main()
