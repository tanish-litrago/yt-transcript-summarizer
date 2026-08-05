import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_HOST, GEMMA_MODEL, RAG_TOP_K
from src.gemma_engine import _call_gemma


def _get_store(video_id: str):
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from config import CHROMA_DIR, EMBED_MODEL

    store_path = os.path.join(CHROMA_DIR, video_id)
    if not os.path.isdir(store_path):
        raise RuntimeError(
            f"No RAG index for '{video_id}'. Summarize the video first."
        )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    return Chroma(persist_directory=store_path, embedding_function=embeddings, collection_name=video_id)


def _retrieve(query: str, video_id: str, k: int = None) -> list[str]:
    results = _get_store(video_id).similarity_search(query, k=k or RAG_TOP_K)
    return [doc.page_content for doc in results]


def _node_by_id(node_id: str, graph: dict):
    return next((n for n in graph.get("nodes", []) if n["id"] == node_id), None)


def _neighbours(node_id: str, graph: dict) -> list[str]:
    connected = set()
    for e in graph.get("edges", []) + graph.get("weak_edges", []):
        if e["source"] == node_id:
            connected.add(e["target"])
        elif e["target"] == node_id:
            connected.add(e["source"])
    return list(connected)


def _edge_between(src: str, tgt: str, graph: dict):
    for e in graph.get("edges", []) + graph.get("weak_edges", []):
        if {e["source"], e["target"]} == {src, tgt}:
            return e
    return None


def _chunks_block(chunks: list[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(chunks))


def _grounded_prompt(question: str, chunks: list[str], context: str = "") -> str:
    return f"""Answer the following question about a YouTube video using only the transcript excerpts below.{context}

Question: {question}

Excerpts:
{_chunks_block(chunks)}

Instructions: answer from the excerpts only. If they don't cover the question, say so. Keep it under 200 words.

Answer:"""


def query_node(node_id: str, graph: dict, video_id: str) -> dict:
    node = _node_by_id(node_id, graph)
    if node is None:
        raise ValueError(f"Node '{node_id}' not found.")

    label = node["label"]
    # Widen the search using immediate neighbours — finds more relevant chunks
    neighbour_labels = [
        _node_by_id(nid, graph)["label"]
        for nid in _neighbours(node_id, graph)
        if _node_by_id(nid, graph)
    ]
    query = f"{label} {' '.join(neighbour_labels[:3])}".strip()
    chunks = _retrieve(query, video_id)

    if not chunks:
        return {
            "answer": f'The video doesn\'t discuss "{label}" in enough detail.',
            "sources": [], "nodes_used": [node_id],
        }

    answer = _call_gemma(_grounded_prompt(f'What does this video say about "{label}"?', chunks))
    return {
        "answer":     answer.strip(),
        "sources":    [c[:120] for c in chunks],
        "nodes_used": [node_id] + _neighbours(node_id, graph)[:9],
    }


def query_edge(src_id: str, tgt_id: str, graph: dict, video_id: str) -> dict:
    src = _node_by_id(src_id, graph)
    tgt = _node_by_id(tgt_id, graph)
    if not src or not tgt:
        raise ValueError(f"Node(s) not found: {src_id}, {tgt_id}")

    chunks = _retrieve(f"{src['label']} {tgt['label']}", video_id, k=RAG_TOP_K + 2)

    edge = _edge_between(src_id, tgt_id, graph)
    context = ""
    if edge:
        context = (f"\nNote: the graph identifies this relationship as "
                   f"\"{edge['relation']}\" ({edge['confidence']:.0%} confidence).")

    if not chunks:
        return {
            "answer": f'The video doesn\'t clearly discuss the link between "{src["label"]}" and "{tgt["label"]}".',
            "sources": [], "nodes_used": [src_id, tgt_id],
        }

    question = f'How are "{src["label"]}" and "{tgt["label"]}" related in this video?'
    answer = _call_gemma(_grounded_prompt(question, chunks, context))
    return {
        "answer":     answer.strip(),
        "sources":    [c[:120] for c in chunks],
        "nodes_used": [src_id, tgt_id],
    }


def query_question(question: str, graph: dict, video_id: str) -> dict:
    chunks = _retrieve(question, video_id, k=RAG_TOP_K + 2)

    if not chunks:
        return {
            "answer": "Couldn't find relevant content in the transcript for that question.",
            "sources": [], "nodes_used": [],
        }

    combined = " ".join(chunks).lower()
    nodes_used = [n["id"] for n in graph.get("nodes", []) if n["label"].lower() in combined]

    answer = _call_gemma(_grounded_prompt(question, chunks))
    return {
        "answer":     answer.strip(),
        "sources":    [c[:120] for c in chunks],
        "nodes_used": nodes_used[:8],
    }
