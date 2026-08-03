"""
src/knowledge_graph.py  —  v2.6
Knowledge Graph extraction from video transcripts using Gemma 4.

Architecture:
  extract_knowledge_graph(transcript, entities_typed, video_id)
    → Gemma 4 call  →  nodes + edges JSON
    → cached to outputs/kg/{video_id}.json  (idempotent)

The entities_typed list from extract_keywords() is passed as seed nodes so
Gemma only needs to find *relationships*, not re-discover entities from scratch.

Node types:
  Concept, Person, Organization, Model, Dataset, Location, Event, Other

Edge relation types:
  uses, is_a, part_of, founded_by, contrasts_with, leads_to,
  gave_rise_to, applied_to, compared_to, mentioned_with

"explained": false (rumour nodes) means the transcript mentions the term but
  doesn't discuss it with enough depth for KG-RAG to give a good answer.
  These render dimmed/dashed in the D3 graph.
"""

import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_HOST, GEMMA_MODEL, KG_DIR, KG_CONFIDENCE_THRESHOLD
from src.gemma_engine import _extract_json, _truncate_transcript


# ── System prompt ─────────────────────────────────────────────────────────────

KG_SYSTEM_PROMPT = (
    "You are a precise knowledge graph extraction assistant. You read video "
    "transcripts and extract concepts, entities, and the relationships between "
    "them as a structured graph. "
    "CRITICAL: Your entire response must be ONLY a single raw JSON object. "
    "Do not write any introduction, explanation, or closing remarks. "
    "Do not wrap the JSON in markdown code fences. "
    "The very first character of your response must be '{' and the very "
    "last character must be '}'."
)


# ── Ollama call — KG-specific options ────────────────────────────────────────

def _call_gemma_kg(prompt: str, system: str = None) -> str:
    """
    Ollama call tuned specifically for KG extraction.

    Uses num_predict=2048 instead of the standard 1024 because a graph JSON
    with 20-40 nodes and edges is significantly larger than a keyword list —
    the standard budget would truncate it mid-response.

    Uses temperature=0.1 for maximum determinism: graph structure should be
    consistent across retries.

    This is a self-contained call (not delegated to gemma_engine._call_gemma)
    so we can freely set different options without modifying the shared helper.
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
            "temperature": 0.1,
            "num_ctx":     8192,
            "num_predict": 2048,   # larger budget for graph JSON output
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data    = response.json()
        content = data.get("message", {}).get("content", "").strip()

        if not content:
            thinking = data.get("message", {}).get("thinking", "")
            raise RuntimeError(
                "Gemma returned an empty response for KG extraction.\n"
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
            "Gemma took too long for KG extraction (over 300s). "
            "Try a shorter transcript or check Ollama's logs."
        )
    except Exception as e:
        raise RuntimeError(f"Gemma KG API call failed: {e}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _kg_cache_path(video_id: str) -> str:
    """Absolute path to the cached KG JSON for this video_id."""
    return os.path.join(KG_DIR, f"{video_id}.json")


def _load_cached_kg(video_id: str):
    """Return cached KG dict if it exists on disk, else None."""
    path = _kg_cache_path(video_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None   # corrupt cache — will rebuild
    return None


def _save_kg_cache(video_id: str, kg: dict):
    """Persist the KG dict as JSON to disk."""
    path = _kg_cache_path(video_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)


# ── Normalisation + parsing ───────────────────────────────────────────────────

_VALID_NODE_TYPES = {
    "Concept", "Person", "Organization", "Model",
    "Dataset", "Location", "Event", "Other",
}

_VALID_RELATIONS = {
    "uses", "is_a", "part_of", "founded_by", "contrasts_with",
    "leads_to", "gave_rise_to", "applied_to", "compared_to", "mentioned_with",
}


def _normalise_id(label: str) -> str:
    """Convert a human label to a stable snake_case node ID."""
    return (
        label.lower().strip()
        .replace(" ", "_").replace("-", "_")
        .replace("(", "").replace(")", "")
    )


def _parse_kg(data: dict) -> dict:
    """
    Validate and normalise the raw Gemma KG JSON response.

    Guarantees:
      - Every node has: id (str), label (str), type (valid), explained (bool)
      - Every edge has: source, target (valid node IDs), relation (valid), confidence (float)
      - Edges are split into 'edges' (confidence >= threshold) and 'weak_edges' (below threshold)
      - Duplicate nodes and self-loop edges are dropped
      - Edges referencing unknown node IDs are dropped
    """
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    # --- Nodes ---
    nodes    = []
    seen_ids = set()

    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        label = str(n.get("label") or n.get("id") or "").strip()
        if not label:
            continue
        nid = _normalise_id(str(n.get("id") or label))
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        nodes.append({
            "id":        nid,
            "label":     label,
            "type":      n.get("type") if n.get("type") in _VALID_NODE_TYPES else "Other",
            "explained": bool(n.get("explained", True)),
        })

    # --- Edges ---
    strong_edges = []
    weak_edges   = []

    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src = _normalise_id(str(e.get("source") or ""))
        tgt = _normalise_id(str(e.get("target") or ""))

        if not src or not tgt or src == tgt:
            continue
        # Drop edges whose nodes don't exist in our validated node list
        if src not in seen_ids or tgt not in seen_ids:
            continue

        relation   = e.get("relation", "mentioned_with")
        if relation not in _VALID_RELATIONS:
            relation = "mentioned_with"

        confidence = float(e.get("confidence", 0.8))
        edge = {
            "source":     src,
            "target":     tgt,
            "relation":   relation,
            "confidence": round(confidence, 3),
        }

        if confidence >= KG_CONFIDENCE_THRESHOLD:
            strong_edges.append(edge)
        else:
            weak_edges.append(edge)

    return {
        "nodes":      nodes,
        "edges":      strong_edges,
        "weak_edges": weak_edges,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_knowledge_graph(
    transcript: str,
    entities_typed: list,
    video_id: str = None,
) -> dict:
    """
    Extract a knowledge graph from a video transcript using Gemma 4.

    Passes entities_typed (from extract_keywords()) as seed nodes so Gemma
    only needs to discover *relationships*, not re-extract entities from scratch.

    If video_id is provided, result is cached to outputs/kg/{video_id}.json
    and reused on subsequent calls without re-running Gemma (idempotent).

    Args:
        transcript:     Full transcript text.
        entities_typed: List of {"name": str, "type": str} from extract_keywords().
                        Used as the starting seed node list for the prompt.
        video_id:       YouTube video ID used as cache key. Optional but recommended.

    Returns:
        {
            "nodes":      list[dict],   # {id, label, type, explained}
            "edges":      list[dict],   # {source, target, relation, confidence}  confidence >= 0.7
            "weak_edges": list[dict],   # same shape, confidence < 0.7  (render dimmed/dashed)
        }

    Raises:
        RuntimeError if Gemma fails 3 times or Ollama is unreachable.
    """
    # --- Idempotency check ---
    if video_id:
        cached = _load_cached_kg(video_id)
        if cached is not None:
            print(f"[KG] Cached graph found for {video_id} — reusing (no Gemma call).")
            return cached

    print(f"[KG] Extracting knowledge graph with {GEMMA_MODEL}...")

    # Truncate transcript to fit in context window alongside the graph JSON output
    transcript_input = _truncate_transcript(transcript, max_words=1200)

    # Build seed-node block for the prompt (cap at 25 to avoid prompt bloat)
    seed_lines = [
        f'  - "{e["name"]}" (type: {e.get("type", "Other")})'
        for e in entities_typed[:25]
    ]
    seed_block = "\n".join(seed_lines) if seed_lines else "  (none provided)"

    prompt = f"""
You are given a video transcript and a list of seed entities already identified in it.

Your task:
1. Use the seed entities below as your starting node list.
2. Add up to 10 additional important concepts or entities from the transcript
   that are NOT in the seed list (skip generic or filler words).
3. For each pair of nodes that are meaningfully related in the transcript,
   define one directed edge with a relation label and a confidence score (0.0–1.0).
   Only add an edge if the relationship is clearly stated or strongly implied.

Seed entities already identified:
{seed_block}

Node type options:
  Concept, Person, Organization, Model, Dataset, Location, Event, Other

Edge relation options (pick the single closest match):
  uses, is_a, part_of, founded_by, contrasts_with, leads_to,
  gave_rise_to, applied_to, compared_to, mentioned_with

For each node, set "explained": true if the transcript discusses it with enough
depth to answer a question about it, or false if it is only briefly mentioned.

Respond with ONLY this JSON structure, nothing else:

{{
  "nodes": [
    {{"id": "snake_case_id", "label": "Human Readable Label", "type": "Concept", "explained": true}},
    {{"id": "another_id",    "label": "Another Label",         "type": "Person",  "explained": false}}
  ],
  "edges": [
    {{"source": "snake_case_id", "target": "another_id", "relation": "uses", "confidence": 0.92}},
    {{"source": "another_id",    "target": "snake_case_id", "relation": "founded_by", "confidence": 0.65}}
  ]
}}

Transcript:
\"\"\"
{transcript_input}
\"\"\"
""".strip()

    last_error = None
    for attempt in range(1, 4):   # up to 3 attempts
        raw = _call_gemma_kg(prompt, system=KG_SYSTEM_PROMPT)
        try:
            data = _extract_json(raw)
            if not data.get("nodes"):
                raise RuntimeError(
                    f"Gemma returned empty 'nodes' list. Raw output: {raw[:200]}"
                )
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"[KG] Bad response (attempt {attempt}/3): {exc}. Retrying...")
    else:
        raise last_error

    kg = _parse_kg(data)

    print(
        f"[KG] Done. {len(kg['nodes'])} nodes, "
        f"{len(kg['edges'])} strong edges, "
        f"{len(kg['weak_edges'])} weak edges."
    )

    # --- Persist to cache ---
    if video_id:
        _save_kg_cache(video_id, kg)
        print(f"[KG] Graph cached to {_kg_cache_path(video_id)}")

    return kg
