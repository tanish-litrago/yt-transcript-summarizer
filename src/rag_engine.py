"""
src/rag_engine.py  -  v2.5
RAG (Retrieval-Augmented Generation) engine for Chat-with-Video.

Architecture:
  Build:    Transcript -> LangChain splitter -> chunks
                       -> Ollama nomic-embed-text -> embeddings
                       -> ChromaDB (persisted per video_id)

  Answer:   Question -> embed -> ChromaDB similarity search (top-k)
                     -> grounded prompt -> Gemma 4 via Ollama -> answer + sources

Design notes:
  - Uses OllamaEmbeddings so embeddings run through the same Ollama server as
    Gemma -- no extra model downloads, no new GPU dependency.
  - build_vector_store() is idempotent: if the ChromaDB store for this
    video_id already exists on disk it is reused, not rebuilt.
  - _call_gemma() is imported from gemma_engine to keep all Ollama call logic
    in one place (options, timeout, error handling, think:False workaround).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OLLAMA_HOST,
    CHROMA_DIR,
    EMBED_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
)


# -- Lazy imports (so the module can be imported in tests without heavy deps) --

def _get_splitter():
    # langchain.text_splitter was removed in LangChain 1.x;
    # use the standalone langchain_text_splitters package instead.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )


def _get_embeddings():
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)


def _get_chroma():
    from langchain_community.vectorstores import Chroma
    return Chroma


# -- Internal helpers ---------------------------------------------------------

def _store_path(video_id: str) -> str:
    """Returns the absolute path to the ChromaDB directory for this video_id."""
    return os.path.join(CHROMA_DIR, video_id)


# -- Public API ---------------------------------------------------------------

def build_vector_store(transcript: str, video_id: str):
    """
    Chunks the transcript, embeds each chunk via Ollama nomic-embed-text,
    and persists the resulting ChromaDB vector store to disk.

    Idempotent: if the store already exists for this video_id it is opened
    and returned immediately without re-embedding.

    Args:
        transcript: Full transcript text.
        video_id:   YouTube video ID (used as the store directory name).

    Returns:
        A LangChain Chroma vector store object.
    """
    store_dir = _store_path(video_id)

    if os.path.isdir(store_dir) and os.listdir(store_dir):
        print(f"[RAG] Existing index found for {video_id} -- reusing (no re-embedding).")
        Chroma = _get_chroma()
        return Chroma(
            persist_directory=store_dir,
            embedding_function=_get_embeddings(),
        )

    print(f"[RAG] Building vector store for {video_id}...")

    splitter = _get_splitter()
    chunks   = splitter.split_text(transcript)
    print(f"[RAG] {len(chunks)} chunks from {len(transcript.split())} words.")

    Chroma = _get_chroma()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=_get_embeddings(),
        persist_directory=store_dir,
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
    )

    print(f"[RAG] Index built and saved to {store_dir}")
    return vectorstore


def load_vector_store(video_id: str):
    """
    Opens an existing ChromaDB store for the given video_id.

    Returns the Chroma store object, or None if no store exists yet
    (caller should run build_vector_store first).
    """
    store_dir = _store_path(video_id)
    if not os.path.isdir(store_dir) or not os.listdir(store_dir):
        return None

    Chroma = _get_chroma()
    return Chroma(
        persist_directory=store_dir,
        embedding_function=_get_embeddings(),
    )


def answer_question(question: str, video_id: str) -> dict:
    """
    Retrieves the top-k most relevant transcript chunks for the question
    and asks Gemma 4 to answer using only those excerpts.

    Args:
        question:  The user's natural-language question.
        video_id:  YouTube video ID -- identifies which ChromaDB store to use.

    Returns:
        {
            "answer"  : str,          # Gemma's grounded answer
            "sources" : list[str],    # First 200 chars of each retrieved chunk
        }

    Raises:
        RuntimeError if no RAG index exists for this video_id.
    """
    from src.gemma_engine import _call_gemma

    vectorstore = load_vector_store(video_id)
    if vectorstore is None:
        raise RuntimeError(
            f"No RAG index found for video '{video_id}'. "
            "Run build_vector_store() first."
        )

    docs = vectorstore.similarity_search(question, k=RAG_TOP_K)
    if not docs:
        return {
            "answer" : "I couldn't find relevant information in this video to answer that question.",
            "sources": [],
        }

    excerpts = "\n\n".join(
        f"[Excerpt {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    prompt = (
        "You are answering a question about a YouTube video based ONLY on the "
        "transcript excerpts provided below.\n\n"
        "Rules:\n"
        "- Answer using ONLY the information present in the excerpts.\n"
        "- If the excerpts do not contain enough information to answer, say exactly: "
        "\"I don't have enough information from this video to answer that.\"\n"
        "- Do not invent, extrapolate, or use outside knowledge.\n"
        "- Be concise and direct. Aim for 2-5 sentences.\n\n"
        f"Transcript excerpts:\n\"\"\"\n{excerpts}\n\"\"\"\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    system = (
        "You are a factual Q&A assistant. You answer questions strictly based on "
        "the provided transcript excerpts. You never invent information."
    )

    answer = _call_gemma(prompt, system=system, temperature=0.2)

    sources = [doc.page_content[:200].strip() for doc in docs]

    return {
        "answer" : answer.strip(),
        "sources": sources,
    }
