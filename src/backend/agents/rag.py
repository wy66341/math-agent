"""RAG Agent — 基于 FAISS 的检索增强生成问答。

Chunking: 600 chars/block, 80 chars overlap (within contest 500-800 / 50-100).
Index:    FAISS IndexFlatIP (inner product = cosine on normalized vectors).
Citation: [教材名称, 第X章, 第X页] — P0 mandatory format.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.schemas import Citation, RAGAnswerOut, RAGQueryIn, RAGStatusOut, TextbookOut

# ── Config ─────────────────────────────────────────────

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 5

# ── Global state ───────────────────────────────────────

_faiss_index = None          # faiss.IndexFlatIP
_all_chunks: list[ChunkMeta] = []
_embedding_model = None
_model_name = ""
_indexed_book_names: set[str] = set()


@dataclass
class ChunkMeta:
    chunk_id: str
    text: str
    textbook_name: str
    chapter_title: str
    page_start: int
    page_end: int = 0

    def citation(self) -> str:
        return f"[{self.textbook_name}, {self.chapter_title}, 第{self.page_start}页]"

    def to_document(self) -> str:
        return f"教材: {self.textbook_name}\n章节: {self.chapter_title}\n页码: 第{self.page_start}页\n\n{self.text}"


# ── Chunking ───────────────────────────────────────────

def _sliding_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            search_start = max(start + int(size * 0.8), start)
            for sep in ("。", "！", "？", "\n\n", "\n", ".", "!", "?"):
                pos = text.rfind(sep, search_start, end)
                if pos > search_start:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks


# ── FAISS Index ────────────────────────────────────────

def _get_model():
    global _embedding_model, _model_name
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        model_path = _resolve_model_path(_model_name)
        import os as _os
        cache_dir = _os.path.expanduser("~/.cache/huggingface/hub")
        if _os.path.isdir(cache_dir):
            _embedding_model = SentenceTransformer(model_path, local_files_only=True)
        else:
            _embedding_model = SentenceTransformer(model_path)
    return _embedding_model


def _resolve_model_path(model_name: str) -> str:
    """Resolve model path — ModelScope mirror first, then HF."""
    import os
    # Try ModelScope SDK
    try:
        from modelscope import snapshot_download
        cache = snapshot_download(model_name)
        if os.path.isdir(cache):
            return cache
    except Exception:
        pass
    # Fallback: direct name (HF download or local cache)
    return model_name


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Build Index ────────────────────────────────────────

async def build_index(textbooks: list[TextbookOut]) -> RAGStatusOut:
    """Build FAISS index from parsed textbooks.

    Args:
        textbooks: List of parsed TextbookOut from Parser Agent.

    Returns:
        RAGStatusOut with indexing statistics.
    """
    global _faiss_index, _all_chunks, _indexed_book_names
    import faiss

    model = _get_model()
    _all_chunks = []
    _indexed_book_names = set()

    # 1. Chunk all chapters
    raw_chunks: list[tuple[str, ChunkMeta]] = []
    for book in textbooks:
        _indexed_book_names.add(book.title)
        for ch in book.chapters:
            if not ch.content.strip():
                continue
            chunks = _sliding_window(ch.content)
            pg_start = ch.page_start
            pg_end = ch.page_end
            pages = max(pg_end - pg_start + 1, 1)
            for i, chunk_text in enumerate(chunks):
                approx_page = pg_start + int((i / max(len(chunks), 1)) * pages)
                meta = ChunkMeta(
                    chunk_id=hashlib.md5(f"{book.textbook_id}_{ch.chapter_id}_{i}".encode()).hexdigest()[:12],
                    text=chunk_text,
                    textbook_name=book.title,
                    chapter_title=ch.title,
                    page_start=min(approx_page, pg_end),
                    page_end=pg_end,
                )
                raw_chunks.append((chunk_text, meta))

    if not raw_chunks:
        _faiss_index = None
        return RAGStatusOut(indexed_books=0, total_chunks=0, embedding_model=_model_name)

    # 2. Encode all chunks
    texts = [t for t, _ in raw_chunks]
    _all_chunks = [m for _, m in raw_chunks]
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    vecs = np.array(vecs, dtype=np.float32)

    # 3. Build FAISS index (IP = cosine on normalized vectors)
    dim = vecs.shape[1]
    _faiss_index = faiss.IndexFlatIP(dim)
    _faiss_index.add(vecs)

    return RAGStatusOut(
        indexed_books=len(_indexed_book_names),
        total_chunks=len(_all_chunks),
        embedding_model=_model_name,
    )


# ── Query ──────────────────────────────────────────────

_RAG_SYSTEM = """你是学科知识助教。你只能根据下方【参考资料】中的内容回答问题。

## 要求
1. 仅使用参考资料中的信息，不要加入你自己的知识
2. 每个关键陈述后标注引用来源：**[教材名称, 第X章, 第X页]**
3. 如果参考资料不足以回答，明确说「当前知识库中未找到相关信息」
4. 回答简洁、条理清晰，用中文"""


async def query(q: RAGQueryIn, llm_callable=None) -> RAGAnswerOut:
    """Retrieve top-k chunks and generate a citation-grounded answer.

    Args:
        q: Question + top_k.
        llm_callable: Async (messages) -> str. Uses DashScope if None.

    Returns:
        RAGAnswerOut with answer, citations, and source texts.
    """
    if _faiss_index is None or not _all_chunks:
        return RAGAnswerOut(
            answer="尚未构建索引。请先上传教材并完成解析和索引构建。",
            citations=[],
            source_chunks=[],
        )

    model = _get_model()

    # 1. Embed question
    q_vec = model.encode([q.question], show_progress_bar=False, normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype=np.float32)

    # 2. FAISS search
    k = min(q.top_k * 2, len(_all_chunks))
    scores, indices = _faiss_index.search(q_vec, k)

    # 3. Dedup by textbook+chapter+page proximity, pick top_k
    seen_keys: set[str] = set()
    ranked: list[tuple[int, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(_all_chunks):
            continue
        meta = _all_chunks[idx]
        key = f"{meta.textbook_name}|{meta.chapter_title}"
        if key not in seen_keys:
            seen_keys.add(key)
            ranked.append((int(idx), float(score)))

    top = ranked[:q.top_k]

    # 4. Build context + citations
    context_parts: list[str] = []
    citations: list[Citation] = []
    source_chunks: list[str] = []

    for i, (idx, score) in enumerate(top):
        meta = _all_chunks[idx]
        context_parts.append(f"[{i + 1}] {meta.to_document()}")
        citations.append(Citation(
            textbook=meta.textbook_name,
            chapter=meta.chapter_title,
            page=meta.page_start,
            relevance_score=round(score, 4),
        ))
        source_chunks.append(meta.text)

    context = "\n\n---\n\n".join(context_parts)

    # 5. Generate answer
    messages = [
        {"role": "system", "content": _RAG_SYSTEM},
        {"role": "user", "content": f"参考资料：\n\n{context}\n\n问题：{q.question}"},
    ]

    answer = await _call_llm(messages, llm_callable)

    return RAGAnswerOut(
        answer=answer,
        citations=citations,
        source_chunks=source_chunks,
    )


async def get_status() -> RAGStatusOut:
    return RAGStatusOut(
        indexed_books=len(_indexed_book_names),
        total_chunks=len(_all_chunks),
        embedding_model=_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
    )


# ── LLM ────────────────────────────────────────────────

async def _call_llm(messages: list[dict], callable_fn=None) -> str:
    if callable_fn:
        return await callable_fn(messages)

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return "（未配置 API Key，无法生成回答）"

    import httpx
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("LLM_MODEL", "qwen-max"),
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
