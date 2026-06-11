"""学科知识整合智能体 — 数学教材学习版

功能:
  - 上传 PDF 教材 → 自动 OCR + 解析章节结构
  - 交互式思维导图 (ECharts Tree) → 展开/收回节点
  - 分层知识点搜索: 章 → 节 → 定理/定义
  - RAG 智能问答 (FAISS + 大模型), 引用溯源
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
import time
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "backend"))

env_file = ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

import gradio as gr
import numpy as np

DATA_DIR = ROOT / "data" / "processed"
UPLOAD_DIR = ROOT / "data" / "textbooks"
OCR_CACHE_DIR = ROOT / "data" / "ocr_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CSS Design System
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

:root {
  --primary: #6366f1; --primary-dark: #4f46e5; --accent: #8b5cf6;
  --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
  --bg: #f8fafc; --bg-card: #ffffff; --text: #0f172a; --text-secondary: #475569;
  --text-muted: #94a3b8; --border: #e2e8f0;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.05);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.06);
  --radius: 12px; --radius-lg: 16px; --radius-xl: 20px;
  --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.gradio-container {
  max-width: 1500px !important; margin: 0 auto !important;
  font-family: 'Inter','Noto Sans SC',system-ui,sans-serif !important;
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #faf5ff 100%) !important;
}
footer { display: none !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

.app-header {
  text-align: center; padding: 28px 20px 18px; margin-bottom: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a78bfa 100%);
  border-radius: var(--radius-xl); box-shadow: var(--shadow-lg);
}
.app-header h1 { font-size: 1.6rem; font-weight: 800; color: #fff; margin: 0 0 4px; }
.app-header p { font-size: 0.82rem; color: rgba(255,255,255,0.82); margin: 0; }

.section-title {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
  padding-bottom: 8px; border-bottom: 2px solid #f1f5f9;
}
.section-title .icon-circle {
  width: 34px; height: 34px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center; font-size: 17px;
}
.section-title h3 { margin: 0; font-size: 0.95rem; font-weight: 700; color: var(--text); }

.glass-card {
  background: rgba(255,255,255,0.75); backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,0.7); border-radius: var(--radius-lg);
  padding: 18px; margin-bottom: 10px; box-shadow: var(--shadow-md);
  transition: all var(--transition);
}
.glass-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }
.glass-card .card-label {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-muted); margin-bottom: 6px; font-weight: 600;
}
.glass-card .card-value { font-size: 1.5rem; font-weight: 800; color: var(--text); }

.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 7px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--text-secondary); }
.stat-num { font-weight: 600; color: var(--text); }

.progress-bar-wrap {
  background: #f1f5f9; border-radius: 10px; height: 8px; margin: 8px 0; overflow: hidden;
}
.progress-bar-fill {
  height: 100%; border-radius: 10px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  transition: width 0.3s ease;
}

.status-badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600;
}
.status-badge.ready { background: #ecfdf5; color: #059669; }
.status-badge.processing { background: #fffbeb; color: #d97706; }
.status-badge.empty { background: #f1f5f9; color: #94a3b8; }

button.primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
  transition: all var(--transition) !important;
}
button.primary:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important; }

.cite-panel {
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: 14px; margin: 8px 0; box-shadow: var(--shadow-sm);
}
.cite-panel summary { font-weight: 600; font-size: 0.85rem; cursor: pointer; color: var(--text); }
.cite-panel summary:hover { color: var(--primary); }
.cite-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin: 8px 0; }
.cite-table th { text-align: left; padding: 5px 8px; font-size: 0.68rem; color: var(--text-muted); background: #f8fafc; }
.cite-table td { padding: 5px 8px; border-bottom: 1px solid #f1f5f9; }
.cite-score { color: var(--success); font-weight: 600; }
.chunk-block {
  background: #f8fafc; border-left: 3px solid var(--accent);
  padding: 10px 14px; margin: 4px 0; border-radius: 0 8px 8px 0;
  font-size: 0.8rem; line-height: 1.7; color: var(--text-secondary);
}
.chunk-details { margin: 4px 0; }
.chunk-details summary { cursor: pointer; color: var(--primary); font-size: 0.8rem; font-weight: 500; }

.empty-state { text-align: center; padding: 28px 16px; color: var(--text-muted); }
.empty-state .icon { font-size: 2.2rem; opacity: 0.4; margin-bottom: 6px; }
.empty-state p { font-size: 0.82rem; margin: 0; }
"""

HEADER_HTML = """
<div class="app-header">
  <h1>数学教材智能学习系统</h1>
  <p>上传 PDF · 自动 OCR 解析 · 思维导图导航 · 知识点搜索 · RAG 智能问答</p>
</div>"""


# ═══════════════════════════════════════════════════════════════
# OCR Engine
# ═══════════════════════════════════════════════════════════════

def _ocr_page_from_pdf(args: tuple) -> tuple[int, str]:
    """OCR a single page (pg_idx, image_bytes) → (pg_idx, text). Module-level for multiprocessing."""
    pg_idx, img_bytes = args
    import subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(img_bytes)
            inp = f.name
        out = tempfile.mktemp(suffix='')
        subprocess.run(
            ['tesseract', inp, out, '-l', 'chi_sim', '--psm', '6'],
            capture_output=True, timeout=30,
        )
        txt_path = Path(out + '.txt')
        text = txt_path.read_text(errors='ignore') if txt_path.exists() else ''
        try:
            os.unlink(inp)
            if txt_path.exists():
                os.unlink(txt_path)
        except OSError:
            pass
        return pg_idx, text
    except Exception:
        return pg_idx, ''


# ═══════════════════════════════════════════════════════════════
# Math Content Extraction
# ═══════════════════════════════════════════════════════════════

# Patterns for math items in OCR'd text
MATH_ITEM_PATTERNS = [
    (re.compile(r'(定义\s*\d+(?:\.\d+)*)\s*(.*?)(?=(?:定义\s*\d|定理\s*\d|命题\s*\d|推论\s*\d|引理\s*\d|例\s*\d|证明|$))', re.DOTALL), '定义'),
    (re.compile(r'(定理\s*\d+(?:\.\d+)*)\s*(.*?)(?=(?:定义\s*\d|定理\s*\d|命题\s*\d|推论\s*\d|引理\s*\d|例\s*\d|证明|$))', re.DOTALL), '定理'),
    (re.compile(r'(命题\s*\d+(?:\.\d+)*)\s*(.*?)(?=(?:定义\s*\d|定理\s*\d|命题\s*\d|推论\s*\d|引理\s*\d|例\s*\d|证明|$))', re.DOTALL), '命题'),
    (re.compile(r'(推论\s*\d+(?:\.\d+)*)\s*(.*?)(?=(?:定义\s*\d|定理\s*\d|命题\s*\d|推论\s*\d|引理\s*\d|例\s*\d|证明|$))', re.DOTALL), '推论'),
    (re.compile(r'(引理\s*\d+(?:\.\d+)*)\s*(.*?)(?=(?:定义\s*\d|定理\s*\d|命题\s*\d|推论\s*\d|引理\s*\d|例\s*\d|证明|$))', re.DOTALL), '引理'),
]


def extract_math_items(text: str) -> list[dict]:
    """Extract definitions, theorems, propositions etc. from text."""
    items = []
    seen = set()
    for pattern, item_type in MATH_ITEM_PATTERNS:
        for m in pattern.finditer(text):
            label = m.group(1).strip()
            desc = m.group(2).strip()[:150] if m.group(2) else ''
            # Clean up description
            desc = re.sub(r'\s+', ' ', desc)
            if label not in seen:
                seen.add(label)
                items.append({'type': item_type, 'label': label, 'desc': desc})
    return items


# ═══════════════════════════════════════════════════════════════
# Mind Map Builder
# ═══════════════════════════════════════════════════════════════

def build_mindmap_from_toc(pdf_path: str) -> dict:
    """Instant: build mind map skeleton from PDF table of contents only (no OCR)."""
    import fitz
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    doc.close()

    filename = Path(pdf_path).stem
    tree = {"name": filename, "children": []}

    for level, title, page in toc:
        title = title.strip()
        if any(kw in title for kw in ['封面','版权','丛书','前言','目录']):
            continue
        if level == 1:
            tree["children"].append({"name": title, "children": [], "_page": page})
        elif level == 2 and tree["children"]:
            tree["children"][-1]["children"].append(
                {"name": title, "children": [], "_page": page})

    return tree


def enrich_tree_with_ocr(pdf_path: str, tree: dict) -> dict:
    """Background: OCR all pages in parallel, extract math items, enrich the tree in-place.
    Returns the same tree object enriched with math items. Caches full result.
    Uses multiprocessing (~8 workers → ~30s for 192 pages at 200 DPI).
    """
    import fitz, multiprocessing

    filename = Path(pdf_path).stem
    tree_cache = OCR_CACHE_DIR / f"{filename}_tree_full.json"

    if tree_cache.exists():
        try:
            cached = json.loads(tree_cache.read_text())
            # Copy items from cache into current tree
            for ch_idx, ch in enumerate(tree.get("children", [])):
                if ch_idx < len(cached.get("children", [])):
                    for sec_idx, sec in enumerate(ch.get("children", [])):
                        cached_secs = cached["children"][ch_idx].get("children", [])
                        if sec_idx < len(cached_secs):
                            sec["children"] = cached_secs[sec_idx].get("children", [])
            return tree
        except Exception:
            pass

    # Extract page images at 200 DPI
    doc = fitz.open(pdf_path)
    total = len(doc)
    page_images = [(i, doc[i].get_pixmap(dpi=200).tobytes("png")) for i in range(total)]
    doc.close()

    n_workers = min(8, multiprocessing.cpu_count())
    print(f"  🔍 并行 OCR ({n_workers} workers × {total} pages @ 200 DPI)...")
    start = datetime.now()

    with multiprocessing.Pool(n_workers) as pool:
        results = {}
        done = 0
        for pg_idx, text in pool.imap_unordered(_ocr_page_from_pdf, page_images):
            if text.strip():
                results[pg_idx] = text
            done += 1
            if done % 20 == 0:
                print(f"    OCR: {done}/{total} ({100*done//total}%)")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"  ✅ OCR: {len(results)}/{total} 页, {elapsed:.0f}s")

    # Build page→(ch_idx, sec_idx) mapping
    page_map: dict[int, tuple] = {}
    for ch_idx, ch in enumerate(tree["children"]):
        for s_idx, sec in enumerate(ch.get("children", [])):
            start = sec.get("_page", 0)
            end = 99999
            if s_idx + 1 < len(ch["children"]):
                end = ch["children"][s_idx + 1].get("_page", 99999)
            for p in range(start, end):
                page_map[p] = (ch_idx, s_idx)

    # Collect text per section
    section_texts: dict[tuple, list[str]] = {}
    for pg_idx, text in sorted(results.items()):
        book_page = pg_idx + 1
        key = page_map.get(book_page)
        if key is None:
            # Find nearest
            for p in range(book_page, 0, -1):
                if p in page_map:
                    key = page_map[p]
                    break
        if key is not None:
            section_texts.setdefault(key, []).append(text)

    # Extract and add items
    total_items = 0
    for (ch_idx, sec_idx), texts in section_texts.items():
        combined = '\n'.join(texts)
        items = extract_math_items(combined)
        seen = set()
        sec = tree["children"][ch_idx]["children"][sec_idx]
        for item in items:
            if item['label'] not in seen:
                seen.add(item['label'])
                sec["children"].append({
                    "name": f"{item['label']}",
                    "_type": item['type'],
                    "_desc": item['desc'][:100],
                })
        sec["children"] = sec["children"][:30]
        total_items += len(sec["children"])

    print(f"  📌 提取 {total_items} 个数学条目")
    tree_cache.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    return tree


# ═══════════════════════════════════════════════════════════════
# OCR Full Text for RAG
# ═══════════════════════════════════════════════════════════════

def ocr_full_text(pdf_path: str, progress_callback=None) -> str:
    """OCR entire PDF in parallel. Cached to disk."""
    import fitz, multiprocessing

    filename = Path(pdf_path).stem
    text_cache = OCR_CACHE_DIR / f"{filename}_fulltext.txt"

    if text_cache.exists():
        return text_cache.read_text()

    doc = fitz.open(pdf_path)
    total = len(doc)
    page_images = []
    for i in range(total):
        pix = doc[i].get_pixmap(dpi=200)
        page_images.append((i, pix.tobytes("png")))
    doc.close()

    n_workers = min(8, multiprocessing.cpu_count())
    results = {}
    done = 0
    with multiprocessing.Pool(n_workers) as pool:
        for pg_idx, text in pool.imap_unordered(_ocr_page_from_pdf, page_images):
            if text.strip():
                results[pg_idx] = text
            done += 1
            if progress_callback and done % 20 == 0:
                progress_callback(done / total)

    full_text = '\n\n'.join(
        f"[第{pg+1}页]\n{results[pg]}"
        for pg in sorted(results)
    )
    text_cache.write_text(full_text)
    print(f"  📝 全文 OCR 缓存: {len(full_text)} 字符")
    return full_text


def build_rag_index(pdf_path: str, progress_callback=None) -> dict:
    """Build FAISS index from OCR'd textbook."""
    global _faiss_index, _all_chunks, _rag_ready
    import faiss

    if progress_callback:
        progress_callback(0.05)

    text = ocr_full_text(pdf_path, progress_callback)
    if not text.strip():
        return {"status": "error", "message": "OCR 未提取到文字"}

    if progress_callback:
        progress_callback(0.5)

    raw_chunks = _chunk_text(text)
    if not raw_chunks:
        return {"status": "error", "message": "文本分块失败"}

    model = _get_embedding_model()
    vecs = model.encode(raw_chunks, show_progress_bar=False, normalize_embeddings=True)
    vecs = np.array(vecs, dtype=np.float32)

    if progress_callback:
        progress_callback(0.85)

    dim = vecs.shape[1]
    _faiss_index = faiss.IndexFlatIP(dim)
    _faiss_index.add(vecs)

    _all_chunks = []
    for i, chunk in enumerate(raw_chunks):
        page_match = re.search(r'\[第(\d+)页\]', chunk)
        page = int(page_match.group(1)) if page_match else 0
        _all_chunks.append({'text': chunk, 'page': page, 'id': i})

    _rag_ready = True
    if progress_callback:
        progress_callback(1.0)

    return {
        "status": "ok",
        "total_chunks": len(_all_chunks),
        "total_chars": len(text),
    }

_faiss_index = None
_all_chunks: list[dict] = []
_embedding_model = None
_rag_ready = False

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
TOP_K = 5


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ('。', '！', '？', '\n\n', '\n', '.', '!', '?'):
                pos = text.rfind(sep, int(start + size * 0.8), end)
                if pos > start + 100:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)
    return chunks



def rag_query(question: str) -> tuple[str, str]:
    """Query the RAG index. Returns (answer, citations_html)."""
    global _faiss_index, _all_chunks, _rag_ready

    if not _rag_ready or _faiss_index is None:
        return "请先上传教材并等待索引构建完成", ""

    model = _get_embedding_model()
    q_vec = model.encode([question], show_progress_bar=False, normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype=np.float32)

    k = min(TOP_K * 2, len(_all_chunks))
    scores, indices = _faiss_index.search(q_vec, k)

    seen_pages = set()
    ranked = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(_all_chunks):
            continue
        page = _all_chunks[idx]['page']
        if page not in seen_pages:
            seen_pages.add(page)
            ranked.append((int(idx), float(score)))

    top = ranked[:TOP_K]
    if not top:
        return "未找到相关内容", ""

    # Build context for LLM
    context_parts = []
    citations = []
    for i, (idx, score) in enumerate(top):
        chunk = _all_chunks[idx]
        text_clean = re.sub(r'\[第\d+页\]\s*', '', chunk['text'])[:800]
        context_parts.append(f"[{i+1}] (第{chunk['page']}页)\n{text_clean}")
        citations.append({'idx': i+1, 'page': chunk['page'], 'score': round(score, 4)})

    context = "\n\n---\n\n".join(context_parts)

    # Call LLM
    import httpx
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return "（未配置 API Key）", _build_citations_html(citations)

    try:
        resp = httpx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("LLM_MODEL", "qwen-max"),
                "messages": [
                    {"role": "system", "content": "你是数学助教。仅根据参考资料回答，每个关键陈述标注 [第X页]。参考资料不足时说明。"},
                    {"role": "user", "content": f"参考资料：\n\n{context}\n\n问题：{question}"},
                ],
                "temperature": 0.3, "max_tokens": 1500,
            },
            timeout=httpx.Timeout(60),
        )
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"LLM 调用失败: {e}\n\n检索到的相关资料:\n{context[:1500]}"

    return answer, _build_citations_html(citations)


def _build_citations_html(citations: list[dict]) -> str:
    rows = ""
    for c in citations:
        rows += f"""<tr>
<td>{c['idx']}</td><td>第{c['page']}页</td>
<td class="cite-score">{c['score']:.2f}</td></tr>"""
    return f"""<div class="cite-panel">
<details open><summary>引用来源（{len(citations)} 条）</summary>
<table class="cite-table">
<thead><tr><th>#</th><th>位置</th><th>相关度</th></tr></thead>
<tbody>{rows}</tbody></table>
</details></div>"""


# ═══════════════════════════════════════════════════════════════
# ECharts Mind Map
# ═══════════════════════════════════════════════════════════════

def _make_mindmap_html(tree: dict | None) -> str:
    """Generate ECharts tree chart via iframe with data URI for reliable JS execution."""
    if tree is None or not tree.get("children"):
        return """<div class="empty-state"><div class="icon">🗺️</div>
<p>请上传教材以生成思维导图</p></div>"""

    # Remove BOM and invisible chars from node names
    def clean_node(node):
        node['name'] = node['name'].strip().lstrip('﻿​‎‏')
        for child in node.get('children', []):
            clean_node(child)
    clean_node(tree)

    import base64
    data_json = json.dumps(tree, ensure_ascii=False)
    # Escape backticks and closing script tags for JS embedding
    data_safe = data_json.replace('\\', '\\\\').replace('`', '\\`').replace('</script>', '<\\/script>')

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif;background:#fff;overflow:hidden}}
#mindmap{{width:100vw;height:100vh}}
</style></head><body>
<div id="mindmap"></div>
<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.5.0/echarts.min.js">
</script>
<script>
var treeData = {data_safe};
(function(){{
  function processNode(node) {{
    if (node.children && node.children.length > 0) {{
      node.children = node.children.slice(0, 60);
      node.children.forEach(processNode);
    }}
    if (node._type === '定义') node.itemStyle = {{ color: '#6366f1' }};
    else if (node._type === '定理') node.itemStyle = {{ color: '#ef4444' }};
    else if (node._type === '命题') node.itemStyle = {{ color: '#f59e0b' }};
    else if (node._type === '推论') node.itemStyle = {{ color: '#10b981' }};
    else if (node._type === '引理') node.itemStyle = {{ color: '#3b82f6' }};
    if (node.name.length > 22) node.name = node.name.slice(0, 20) + '…';
    return node;
  }}
  processNode(treeData);
  var c = echarts.init(document.getElementById('mindmap'));
  c.setOption({{
    tooltip: {{
      trigger: 'item', triggerOn: 'mousemove',
      formatter: function(p) {{
        return '<b>' + p.name + '</b>' + (p.data._desc ? '<br/><span style="font-size:12px;color:#64748b">' + p.data._desc.slice(0, 200) + '</span>' : '');
      }}
    }},
    series: [{{
      type: 'tree', data: [treeData],
      top: '2%', left: '6%', bottom: '2%', right: '12%',
      symbol: 'circle', symbolSize: [16, 28],
      orient: 'LR', expandAndCollapse: true, initialTreeDepth: 2,
      label: {{ position: 'left', verticalAlign: 'middle', align: 'right', fontSize: 11, color: '#334155' }},
      leaves: {{ label: {{ position: 'right', verticalAlign: 'middle', align: 'left' }} }},
      emphasis: {{ focus: 'descendant' }},
      lineStyle: {{ color: '#cbd5e1', width: 1.2, curviness: 0.5 }},
    }}]
  }});
  window.addEventListener('resize', function() {{ c.resize(); }});
}})();
</script>
</body></html>"""

    encoded = base64.b64encode(html_doc.encode('utf-8')).decode('ascii')
    return f'<iframe src="data:text/html;base64,{encoded}" style="width:100%;height:640px;border:none;border-radius:16px;background:#fff"></iframe>'


# ═══════════════════════════════════════════════════════════════
# Chapter Selector (hierarchical search)
# ═══════════════════════════════════════════════════════════════

def _build_chapter_selector(tree: dict | None) -> list[dict]:
    """Build a hierarchical structure for the chapter-selector accordion."""
    if tree is None or not tree.get("children"):
        return []

    result = []
    for ch in tree["children"]:
        ch_name = ch["name"]
        sections = []
        for sec in ch.get("children", []):
            sec_name = sec["name"]
            items = []
            for item in sec.get("children", []):
                items.append({
                    "label": f"  {item['name']}",
                    "value": item['name'],
                    "chapter": ch_name,
                    "section": sec_name,
                })
            sections.append({
                "label": f" {sec_name}",
                "value": sec_name,
                "chapter": ch_name,
                "items": items,
            })
        result.append({
            "chapter": ch_name,
            "sections": sections,
        })
    return result


# ═══════════════════════════════════════════════════════════════
# Global state
# ═══════════════════════════════════════════════════════════════

_current_pdf_path: str | None = None
_current_tree: dict | None = None


# ═══════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════

def upload_and_process(file):
    """Two-phase processing: (1) instant TOC mind map → (2) background OCR + RAG index."""
    global _current_pdf_path, _current_tree

    if file is None:
        return (
            '<span class="status-badge empty">未上传</span>',
            _make_mindmap_html(None),
            gr.Dropdown(choices=[], value=None),
            "请上传教材文件",
        )

    fpath = file.name if hasattr(file, 'name') else str(file)
    dst = UPLOAD_DIR / Path(fpath).name
    import shutil
    shutil.copy(fpath, str(dst))
    _current_pdf_path = str(dst)

    # ── Phase 1: Instant TOC mind map (no OCR) ──────────────
    try:
        tree = build_mindmap_from_toc(str(dst))
        _current_tree = tree
    except Exception as e:
        yield (
            f'<span class="status-badge empty">解析失败: {e}</span>',
            _make_mindmap_html(None),
            gr.Dropdown(choices=[], value=None),
            f"❌ {e}",
        )
        return

    ch_count = len(tree.get("children", []))
    sec_count = sum(len(ch.get("children", [])) for ch in tree.get("children", []))
    choices = _build_choices(tree)

    yield (
        '<span class="status-badge processing">思维导图就绪 — OCR 进行中...</span>',
        _make_mindmap_html(tree),
        gr.Dropdown(choices=choices[:500], value=None),
        f"📘 {ch_count} 章 · {sec_count} 节 | 🔍 正在 OCR 识别定理/定义...",
    )

    # ── Phase 2: Background OCR + enrich + RAG ─────────────
    try:
        import threading

        ocr_done = [False]
        rag_done = [False]
        progress = [0.0]

        def ocr_thread():
            nonlocal tree
            try:
                enrich_tree_with_ocr(str(dst), tree)
            except Exception as e:
                print(f"OCR enrich error: {e}")
            ocr_done[0] = True

        def rag_thread():
            try:
                build_rag_index(str(dst))
            except Exception as e:
                print(f"RAG build error: {e}")
            rag_done[0] = True

        t1 = threading.Thread(target=ocr_thread)
        t2 = threading.Thread(target=rag_thread)
        t1.start()
        t2.start()

        # Wait for OCR to finish (RAG may take longer, show progress)
        while not ocr_done[0]:
            time.sleep(2)

        # Re-count with items
        item_count = sum(
            len(sec.get("children", []))
            for ch in tree.get("children", [])
            for sec in ch.get("children", [])
        )
        choices = _build_choices(tree)

        if rag_done[0]:
            status_html = '<span class="status-badge ready">已就绪</span>'
            info = f"✅ {ch_count} 章 · {sec_count} 节 · {item_count} 个知识点 · RAG 索引已构建"
        else:
            status_html = '<span class="status-badge processing">RAG 索引构建中...</span>'
            info = f"📘 {ch_count} 章 · {sec_count} 节 · {item_count} 个知识点 | ⏳ RAG 索引构建中..."

        yield (
            status_html,
            _make_mindmap_html(tree),
            gr.Dropdown(choices=choices[:500], value=None),
            info,
        )

        # Wait for RAG
        t2.join(timeout=30)

        choices = _build_choices(tree)
        yield (
            '<span class="status-badge ready">已就绪</span>',
            _make_mindmap_html(tree),
            gr.Dropdown(choices=choices[:500], value=None),
            f"✅ {ch_count} 章 · {sec_count} 节 · {item_count} 个知识点 · RAG 就绪",
        )

    except Exception as e:
        yield (
            f'<span class="status-badge ready">部分就绪</span>',
            _make_mindmap_html(tree),
            gr.Dropdown(choices=choices[:500], value=None),
            f"⚠️ 思维导图就绪，但 OCR 出错: {str(e)[:80]}",
        )


def _build_choices(tree: dict) -> list:
    """Build dropdown choices from tree."""
    choices = []
    for ch in tree.get("children", []):
        choices.append(f"📘 {ch['name']}")
        for sec in ch.get("children", []):
            choices.append(f"  📄 {sec['name']}")
            for item in sec.get("children", []):
                choices.append(f"    📌 {item['name']}")
    return choices


def on_node_select(selected: str):
    """When a node is selected in the search dropdown, prepare RAG question."""
    if not selected:
        return "", ""
    # Strip tree prefix
    name = selected.strip().lstrip('📘📄📌').strip()
    question = f"请详细解释「{name}」的内容，包括其数学表述和相关背景。"
    return question, name


def on_rag_search(question: str):
    if not question.strip():
        return "", ""
    answer, cites = rag_query(question)
    return answer, cites


# ── Build UI ──────────────────────────────────────────────

with gr.Blocks(title="数学教材智能学习系统") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # ── LEFT: Upload + Chapter Tree ────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML("""<div class="section-title">
              <div class="icon-circle" style="background:linear-gradient(135deg,#ede9fe,#ddd6fe)">📁</div>
              <h3>教材管理</h3></div>""")

            file_status = gr.HTML('<span class="status-badge empty">未上传</span>')
            uploader = gr.File(
                label="上传数学教材 (PDF)",
                file_count="single",
                file_types=[".pdf"],
            )
            btn_process = gr.Button("解析教材", variant="primary", size="sm", elem_classes="primary")
            upload_info = gr.Markdown("请上传教材文件")

            gr.HTML("""<div class="section-title" style="margin-top:18px">
              <div class="icon-circle" style="background:linear-gradient(135deg,#dbeafe,#bfdbfe)">🔍</div>
              <h3>知识点导航</h3></div>""")

            chapter_selector = gr.Dropdown(
                label="",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=True,
            )

        # ── CENTER: Mind Map ────────────────────────────
        with gr.Column(scale=2, min_width=520):
            gr.HTML("""<div class="section-title">
              <div class="icon-circle" style="background:linear-gradient(135deg,#d1fae5,#a7f3d0)">🧠</div>
              <h3>思维导图</h3>
              <span style="font-size:0.72rem;color:#94a3b8;margin-left:auto">点击节点展开/收回 · 滚轮缩放</span>
              </div>""")
            mindmap_display = gr.HTML(_make_mindmap_html(None))

            gr.HTML("""<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:0.72rem;color:#64748b;margin-top:6px">
              <span>🟣 定义</span><span>🔴 定理</span><span>🟠 命题</span><span>🟢 推论</span><span>🔵 引理</span>
              </div>""")

        # ── RIGHT: RAG Q&A ──────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.HTML("""<div class="section-title">
              <div class="icon-circle" style="background:linear-gradient(135deg,#fef3c7,#fde68a)">💬</div>
              <h3>智能问答</h3></div>""")
            node_name_display = gr.Markdown("")
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="选择左侧知识点或直接输入问题，如：请解释勒贝格积分的定义",
                lines=3,
            )
            btn_ask = gr.Button("查询", variant="primary", size="sm", elem_classes="primary")
            answer_output = gr.Textbox(label="回答", lines=12, interactive=False)
            cites_output = gr.HTML("")

    # ── Events ──────────────────────────────────────────

    btn_process.click(
        fn=upload_and_process,
        inputs=[uploader],
        outputs=[file_status, mindmap_display, chapter_selector, upload_info],
    )

    chapter_selector.change(
        fn=on_node_select,
        inputs=[chapter_selector],
        outputs=[question_input, node_name_display],
    )

    btn_ask.click(
        fn=on_rag_search,
        inputs=[question_input],
        outputs=[answer_output, cites_output],
    )


if __name__ == "__main__":
    print("🚀 数学教材智能学习系统启动中...")
    print(f"🌐 http://0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS)
