"""
src/fact_checker.py  —  v3.0
─────────────────────────────
Claim extraction + web fact-checking pipeline.

Steps:
  1. extract_claims(summary)       — Gemma extracts 5–8 checkable factual claims
  2. search_claim(claim)           — DuckDuckGo returns top-4 web snippets
  3. verify_claim(claim, hits)     — Gemma compares claim against snippets
  4. fact_check(summary, video_id) — orchestrates all three; caches per video_id

Verdicts:
  supported     — web sources clearly back the claim
  contradicted  — web sources clearly conflict with the claim
  unverifiable  — not enough reliable info to decide
  partially     — mixed evidence

Lazy imports: DDGS (DuckDuckGo) only happens at call time so CI passes without it.
"""

import json
import os
import re

from config import OUTPUT_DIR

FACT_CHECK_DIR = os.path.join(OUTPUT_DIR, "fact_checks")
os.makedirs(FACT_CHECK_DIR, exist_ok=True)

MAX_CLAIMS         = 8    # max claims extracted per summary
SEARCH_MAX_RESULTS = 4    # DuckDuckGo results per claim
SNIPPET_MAX_CHARS  = 300  # characters kept per snippet


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(video_id: str) -> str:
    return os.path.join(FACT_CHECK_DIR, f"{video_id}.json")


def _load_cached(video_id: str):
    path = _cache_path(video_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(video_id: str, data: list):
    with open(_cache_path(video_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Gemma — delegates to the project's battle-tested engine ───────────────────
# gemma_engine._call_gemma uses /api/chat with think=False, which is the
# correct path for Gemma 4 via Ollama. The old /api/generate approach was
# silently returning empty strings, causing extract_claims to always fail.

def _call_gemma(prompt: str) -> str:
    from src.gemma_engine import _call_gemma as _engine_call
    return _engine_call(prompt, temperature=0.1)



def _extract_json(text: str):
    """Pull the first JSON object or array out of a messy LLM response."""
    # Try a direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for [...] or {...}
    for pattern in (r"\[.*?\]", r"\{.*?\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None


# ── Step 1 — Claim extraction ─────────────────────────────────────────────────

def extract_claims(summary: str) -> list[str]:
    """Ask Gemma to extract up to MAX_CLAIMS checkable factual claims."""
    prompt = f"""You are a fact-checking assistant.
Read the following video summary and extract up to {MAX_CLAIMS} specific, \
checkable factual claims — statements that can be verified or refuted by \
searching the web (statistics, dates, names, technical facts).

IMPORTANT:
- Return ONLY a JSON array of strings, no other text.
- First character must be [
- Each claim should be a complete sentence.
- Do NOT include vague opinions or predictions.
- Do NOT include more than {MAX_CLAIMS} items.

Summary:
{summary[:3000]}

JSON array of claims:"""

    for attempt in range(3):
        raw = _call_gemma(prompt)
        claims = _extract_json(raw)
        if isinstance(claims, list) and claims:
            # Keep only strings, trim whitespace
            return [str(c).strip() for c in claims if str(c).strip()][:MAX_CLAIMS]
        print(f"[FC] extract_claims attempt {attempt+1} failed to parse JSON")

    return []


# ── Step 2 — Web search ───────────────────────────────────────────────────────

def search_claim(claim: str) -> list[dict]:
    """
    DuckDuckGo search for the claim.
    Returns a list of {title, url, snippet} dicts.
    Lazy import so CI doesn't need the package installed.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("[FC] duckduckgo_search not installed — run: pip install duckduckgo-search")
        return []

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(claim, max_results=SEARCH_MAX_RESULTS):
                snippet = (r.get("body") or "")[:SNIPPET_MAX_CHARS]
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href",  ""),
                    "snippet": snippet,
                })
    except Exception as e:
        print(f"[FC] DuckDuckGo search failed for '{claim[:60]}': {e}")

    return results


# ── Step 3 — Gemma verification ───────────────────────────────────────────────

_VALID_VERDICTS = {"supported", "contradicted", "unverifiable", "partially"}


def verify_claim(claim: str, search_hits: list[dict]) -> dict:
    """
    Ask Gemma to evaluate the claim against the web snippets.
    Returns {verdict, explanation, sources}.
    """
    if not search_hits:
        return {
            "verdict":     "unverifiable",
            "explanation": "No web results found to evaluate this claim.",
            "sources":     [],
        }

    snippets_text = "\n\n".join(
        f"[{i+1}] {h['title']}\n{h['snippet']}"
        for i, h in enumerate(search_hits)
    )

    prompt = f"""You are a precise fact-checker. Evaluate the following claim against \
the web search results provided.

Claim: "{claim}"

Web search results:
{snippets_text}

Instructions:
- Choose exactly ONE verdict from: supported | contradicted | unverifiable | partially
- Write a 1–2 sentence explanation citing which result(s) support your verdict.
- Return ONLY a JSON object, no other text. First character must be {{

Expected format:
{{
  "verdict": "supported",
  "explanation": "According to [1], ..."
}}"""

    for attempt in range(3):
        raw = _call_gemma(prompt)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if verdict not in _VALID_VERDICTS:
                verdict = "unverifiable"
            return {
                "verdict":     verdict,
                "explanation": str(parsed.get("explanation", "")).strip(),
                "sources":     [{"title": h["title"], "url": h["url"]} for h in search_hits],
            }
        print(f"[FC] verify_claim attempt {attempt+1} failed to parse JSON")

    return {
        "verdict":     "unverifiable",
        "explanation": "Gemma could not produce a structured verdict.",
        "sources":     [{"title": h["title"], "url": h["url"]} for h in search_hits],
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def fact_check(summary: str, video_id: str, force: bool = False) -> list[dict]:
    """
    Full pipeline: extract claims → search each → verify each.
    Results are cached per video_id; pass force=True to re-run.

    Returns a list of:
    {
        "claim":       str,
        "verdict":     "supported" | "contradicted" | "unverifiable" | "partially",
        "explanation": str,
        "sources":     [{"title": str, "url": str}, ...]
    }
    """
    if not force:
        cached = _load_cached(video_id)
        if cached:
            return cached

    print("[FC] Extracting claims from summary...")
    claims = extract_claims(summary)
    if not claims:
        print("[FC] No claims extracted.")
        return []

    print(f"[FC] Extracted {len(claims)} claims. Searching and verifying...")
    results = []
    for i, claim in enumerate(claims):
        print(f"[FC]   ({i+1}/{len(claims)}) {claim[:70]}...")
        hits    = search_claim(claim)
        verdict = verify_claim(claim, hits)
        results.append({"claim": claim, **verdict})

    _save_cache(video_id, results)
    print(f"[FC] Done — {len(results)} claims fact-checked.")
    return results
