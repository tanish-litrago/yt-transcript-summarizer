import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_HOST, GEMMA_MODEL, KG_DIR, KG_CONFIDENCE_THRESHOLD


def _get_gemma_helpers():
    from src.gemma_engine import _extract_json, _truncate_transcript
    return _extract_json, _truncate_transcript


_VALID_NODE_TYPES = {
    "Concept", "Person", "Organization", "Model",
    "Dataset", "Location", "Event", "Other",
}

_VALID_RELATIONS = {
    "uses", "is_a", "part_of", "founded_by", "contrasts_with",
    "leads_to", "gave_rise_to", "applied_to", "compared_to", "mentioned_with",
}

_KG_SYSTEM = (
    "You are a knowledge graph extraction assistant. Read the transcript and "
    "output ONLY a raw JSON object — no markdown fences, no explanation. "
    "First character must be '{', last must be '}'."
)


def _call_gemma_kg(prompt: str, system: str = None) -> str:
    # Needs num_predict=2048 — graph JSON is much larger than a keyword list,
    # the shared _call_gemma's 1024 budget truncates it mid-response.
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 2048},
    }

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Gemma returned empty content for KG extraction.")
        return content
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST}. Run: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError("KG extraction timed out (300s). Try a shorter video.")
    except Exception as e:
        raise RuntimeError(f"KG API call failed: {e}")


def _kg_path(video_id: str) -> str:
    return os.path.join(KG_DIR, f"{video_id}.json")


def _load_cached(video_id: str):
    path = _kg_path(video_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(video_id: str, kg: dict):
    with open(_kg_path(video_id), "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)


def _normalise_id(label: str) -> str:
    return label.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def _parse_kg(data: dict) -> dict:
    nodes, seen = [], set()
    for n in data.get("nodes", []):
        if not isinstance(n, dict):
            continue
        label = str(n.get("label") or n.get("id") or "").strip()
        if not label:
            continue
        nid = _normalise_id(str(n.get("id") or label))
        if nid in seen:
            continue
        seen.add(nid)
        nodes.append({
            "id":        nid,
            "label":     label,
            "type":      n.get("type") if n.get("type") in _VALID_NODE_TYPES else "Other",
            "explained": bool(n.get("explained", True)),
        })

    strong, weak = [], []
    for e in data.get("edges", []):
        if not isinstance(e, dict):
            continue
        src = _normalise_id(str(e.get("source") or ""))
        tgt = _normalise_id(str(e.get("target") or ""))
        if not src or not tgt or src == tgt:
            continue
        if src not in seen or tgt not in seen:
            continue
        relation = e.get("relation", "mentioned_with")
        if relation not in _VALID_RELATIONS:
            relation = "mentioned_with"
        confidence = float(e.get("confidence", 0.8))
        edge = {"source": src, "target": tgt, "relation": relation, "confidence": round(confidence, 3)}
        (strong if confidence >= KG_CONFIDENCE_THRESHOLD else weak).append(edge)

    return {"nodes": nodes, "edges": strong, "weak_edges": weak}


def extract_knowledge_graph(transcript: str, entities_typed: list, video_id: str = None) -> dict:
    if video_id:
        cached = _load_cached(video_id)
        if cached is not None:
            print(f"[KG] Reusing cached graph for {video_id}.")
            return cached

    print(f"[KG] Extracting graph with {GEMMA_MODEL}...")

    _extract_json, _truncate_transcript = _get_gemma_helpers()
    transcript_input = _truncate_transcript(transcript, max_words=1200)
    seed_block = "\n".join(
        f'  - "{e["name"]}" (type: {e.get("type", "Other")})'
        for e in entities_typed[:25]
    ) or "  (none provided)"

    prompt = f"""You are given a video transcript and a list of seed entities already identified in it.

Your task:
1. Use the seed entities as your starting node list.
2. Add up to 10 more important concepts from the transcript not in the seed list.
3. For each meaningfully related pair, add one directed edge with a relation and confidence (0.0-1.0).

Seed entities:
{seed_block}

Node types: Concept, Person, Organization, Model, Dataset, Location, Event, Other
Edge relations: uses, is_a, part_of, founded_by, contrasts_with, leads_to, gave_rise_to, applied_to, compared_to, mentioned_with

Set "explained": true if the transcript discusses the node in enough depth to answer a question about it.

Output ONLY this JSON:
{{
  "nodes": [{{"id": "snake_case", "label": "Human Label", "type": "Concept", "explained": true}}],
  "edges": [{{"source": "id_a", "target": "id_b", "relation": "uses", "confidence": 0.9}}]
}}

Transcript:
\"\"\"
{transcript_input}
\"\"\"
""".strip()

    last_error = None
    for attempt in range(1, 4):
        raw = _call_gemma_kg(prompt, system=_KG_SYSTEM)
        try:
            data = _extract_json(raw)
            if not data.get("nodes"):
                raise RuntimeError(f"Empty nodes list. Raw: {raw[:200]}")
            break
        except RuntimeError as exc:
            last_error = exc
            print(f"[KG] Attempt {attempt}/3 failed: {exc}")
    else:
        raise last_error

    kg = _parse_kg(data)
    print(f"[KG] {len(kg['nodes'])} nodes, {len(kg['edges'])} edges, {len(kg['weak_edges'])} weak.")

    if video_id:
        _save_cache(video_id, kg)

    return kg
