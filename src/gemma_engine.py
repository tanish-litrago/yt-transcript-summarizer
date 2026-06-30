import os
import sys
import json
import re

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_HOST, GEMMA_MODEL, TOP_KEYWORDS


# ── Low-level Ollama call ──────────────────────────────────────────────────────

def _call_gemma(prompt: str, system: str = None, temperature: float = 0.3) -> str:
    """
    Sends a single prompt to the local Gemma 4 model via Ollama and
    returns the raw text response.

    Gemma 4 has "thinking mode" enabled by default in Ollama, which adds
    20-70+ seconds of latency and — critically — has a known Ollama bug
    where combining "think": false with "format": "json" silently breaks
    structured output (returns empty/plain text instead of JSON; see
    ollama/ollama#15260). We avoid that broken combination entirely by:
      - explicitly disabling thinking (think=false), which IS confirmed
        to correctly route the final answer into message.content
      - NOT using the format="json" parameter
      - relying on strong prompt instructions + defensive parsing in
        _extract_json() to pull JSON out of the plain-text response
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            # num_ctx: total token budget for prompt + response.
            # The e4b model default is often 2048 which is too small for
            # a transcript + JSON output. 8192 keeps us well within the
            # model's actual context limit while giving the response room.
            "num_ctx": 8192,
            # num_predict: max tokens Gemma may generate in one response.
            # Without this Ollama uses a tiny default (~128) which causes
            # the response to be cut off mid-JSON (producing just '{').
            "num_predict": 1024,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()

        if not content:
            # Defensive: if content is still empty, surface whatever
            # the API gave us (e.g. a thinking field) for debugging
            # instead of failing with a silent empty string.
            thinking = data.get("message", {}).get("thinking", "")
            raise RuntimeError(
                "Gemma returned an empty response.\n"
                f"Thinking field (if any): {thinking[:300]}"
            )

        return content

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running:\n"
            "  ollama serve\n"
            f"And the model is pulled:\n  ollama pull {GEMMA_MODEL}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Gemma took too long to respond (over 300s). "
            "Try a shorter transcript or check Ollama's logs."
        )
    except Exception as e:
        raise RuntimeError(f"Gemma API call failed: {e}")


def _truncate_transcript(transcript: str, max_words: int = 1200) -> str:
    """
    Caps the transcript at `max_words` words before sending to Gemma.

    The e4b (4B-parameter) model has a small effective context window.
    Feeding it a very long transcript causes it to run out of token budget
    before it can generate a complete JSON response, producing truncated
    output like bare '{'. 1200 words (~1600 tokens) leaves enough room in
    an 8192-token context for the prompt template + a full JSON reply.
    """
    words = transcript.split()
    if len(words) <= max_words:
        return transcript
    truncated = " ".join(words[:max_words])
    print(
        f"[Gemma] Transcript truncated from {len(words):,} to {max_words:,} words "
        f"(e4b context limit)."
    )
    return truncated


def _extract_json(text: str) -> dict:
    """
    Robustly extracts a JSON object from Gemma's raw text response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Preamble / trailing commentary around the object
    - Nested JSON objects (brace-counting, not a greedy regex)
    - Truncated responses where the closing '}' is missing
    - Partial/malformed JSON (attempts light repair before giving up)
    """
    cleaned = text.strip()

    # 1. Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. Direct parse (fastest path — works when output is clean)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Brace-counting extraction: find the outermost {...} block
    #    This correctly handles nested objects unlike a greedy regex.
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        end = -1
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        # 3a. We found a balanced block — try parsing it
        if end != -1:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 3b. Truncated response — closing brace(s) are missing.
        #     Attempt a light repair: close every unclosed '{' with '}'.
        if start != -1:
            partial = cleaned[start:]
            # Count unclosed braces outside of strings
            depth = 0
            in_string = False
            escape_next = False
            for ch in partial:
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

            if depth > 0:
                repaired = partial.rstrip().rstrip(",") + ("}" * depth)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    raise RuntimeError(
        f"Gemma returned invalid JSON. Raw output: {text[:500]}"
    )


# ── Summarization ──────────────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = (
    "You are a precise academic note-taking assistant. You read video "
    "transcripts and divide them into logical sections, summarising each "
    "section concisely and factually. You never invent information that "
    "is not present in the transcript. "
    "CRITICAL: Your entire response must be ONLY a single raw JSON object. "
    "Do not write any introduction, explanation, or closing remarks. "
    "Do not wrap the JSON in markdown code fences. "
    "The very first character of your response must be '{' and the very "
    "last character must be '}'."
)


def summarize_transcript(transcript: str) -> dict:
    """
    Uses Gemma 4's long context window to summarise the full transcript
    in a single call — no chunking needed.

    Returns:
        {
            "full_summary": str,
            "chunks"      : list[str],   # one summary per logical section
            "num_chunks"  : int,
        }
    """
    print(f"[Gemma] Summarising transcript ({len(transcript.split())} words) with {GEMMA_MODEL}...")

    # Truncate *before* building the prompt so the whole payload fits in
    # the model's context window (e4b is a 4B-parameter model).
    transcript_input = _truncate_transcript(transcript)

    prompt = f"""
Read the following video transcript and divide it into 4 to 8 logical
sections based on topic changes. For each section, write a concise
2-4 sentence summary covering only what is actually said.

Respond with ONLY this JSON structure, nothing else:

{{
  "sections": [
    "summary of section 1...",
    "summary of section 2...",
    ...
  ]
}}

Transcript:
\"\"\"
{transcript_input}
\"\"\"
""".strip()

    last_error = None
    for attempt in range(1, 4):  # up to 3 attempts
        raw = _call_gemma(prompt, system=SUMMARY_SYSTEM_PROMPT, temperature=0.2)
        try:
            data = _extract_json(raw)
            chunks = data.get("sections", [])
            if not chunks:
                # Gemma parsed OK but returned no sections — treat as a
                # bad response and retry rather than silently using raw text.
                raise RuntimeError(
                    f"Gemma returned empty 'sections' list. Raw output: {raw[:200]}"
                )
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"[Gemma] Bad response (attempt {attempt}/3): {exc}. Retrying...")
    else:
        raise last_error

    full_summary = " ".join(chunks)

    print(f"[Gemma] Done. {len(chunks)} section(s), {len(full_summary.split())} words in summary.")

    return {
        "full_summary": full_summary,
        "chunks"      : chunks,
        "num_chunks"  : len(chunks),
    }


# ── Keyword & entity extraction ───────────────────────────────────────────────

KEYWORD_SYSTEM_PROMPT = (
    "You are a precise text analysis assistant. You extract keywords and "
    "named entities from transcripts. "
    "CRITICAL: Your entire response must be ONLY a single raw JSON object. "
    "Do not write any introduction, explanation, or closing remarks. "
    "Do not wrap the JSON in markdown code fences. "
    "The very first character of your response must be '{' and the very "
    "last character must be '}'."
)


def extract_keywords(transcript: str) -> dict:
    """
    Uses Gemma 4 to extract keywords and named entities in a single call,
    replacing the previous TF-IDF + spaCy NER pipeline.

    Returns:
        {
            "keywords"      : list[str],
            "entities"      : list[str],           # flat names (back-compat)
            "entities_typed": list[dict],          # [{"name": str, "type": str}, ...]
            "combined"      : list[str],
        }
    """
    print(f"[Gemma] Extracting keywords and entities with {GEMMA_MODEL}...")

    # Truncate *before* building the prompt.
    transcript_input = _truncate_transcript(transcript)

    prompt = f"""
Read the following transcript and extract:

1. Up to {TOP_KEYWORDS} important keywords or short key phrases
   (technical terms, concepts, recurring topics).
2. Named entities mentioned: people, organisations, products, places.
   For each entity provide its name AND its type.
   Valid types: Person, Organization, Product, Location, Other.
   Do not include generic words.

Respond with ONLY this JSON structure, nothing else:

{{
  "keywords": ["term1", "term2", ...],
  "entities": [
    {{"name": "Entity Name 1", "type": "Person"}},
    {{"name": "Entity Name 2", "type": "Organization"}}
  ]
}}

Transcript:
\"\"\"
{transcript_input}
\"\"\"
""".strip()

    last_error = None
    for attempt in range(1, 4):  # up to 3 attempts
        raw = _call_gemma(prompt, system=KEYWORD_SYSTEM_PROMPT, temperature=0.2)
        try:
            data = _extract_json(raw)
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"[Gemma] JSON parse failed (attempt {attempt}/3): {exc}. Retrying...")
    else:
        raise last_error

    keywords = [k.strip() for k in data.get("keywords", []) if k.strip()]

    # Accept both new typed format [{"name": ..., "type": ...}] and the old
    # flat list-of-strings format so the function degrades gracefully if
    # Gemma ignores the type instruction.
    raw_entities = data.get("entities", [])
    entities_typed = []
    flat_entities  = []
    for e in raw_entities:
        if isinstance(e, dict):
            name = e.get("name", "").strip()
            etype = e.get("type", "Other").strip()
            if name:
                entities_typed.append({"name": name, "type": etype})
                flat_entities.append(name)
        elif isinstance(e, str) and e.strip():
            # Fallback: Gemma returned a flat string — wrap it as typed
            name = e.strip()
            entities_typed.append({"name": name, "type": "Other"})
            flat_entities.append(name)

    # Merge: entities first (higher priority), then keywords, deduplicated
    seen = set()
    combined = []
    for item in flat_entities + keywords:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            combined.append(item)

    print(f"[Gemma] Found {len(entities_typed)} entities, {len(keywords)} keywords.")

    return {
        "keywords"      : keywords,
        "entities"      : flat_entities,
        "entities_typed": entities_typed,
        "combined"      : combined,
    }


# ── Health check ───────────────────────────────────────────────────────────────

def check_ollama_ready() -> bool:
    """
    Quick check that Ollama is running and the Gemma model is available.
    Call this once at app startup to fail fast with a clear message.
    """
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        if not any(GEMMA_MODEL in m for m in models):
            print(f"[Gemma] WARNING: '{GEMMA_MODEL}' not found in Ollama.")
            print(f"         Run:  ollama pull {GEMMA_MODEL}")
            return False
        print(f"[Gemma] Ready — {GEMMA_MODEL} available via Ollama.")
        return True
    except Exception as e:
        print(f"[Gemma] WARNING: Ollama not reachable at {OLLAMA_HOST} ({e})")
        print(f"         Start it with:  ollama serve")
        return False
