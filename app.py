"""学科知识整合智能体 — 完整 P0 前端。

Feature:
  - 拖拽/点击上传多格式教材 (PDF/MD/TXT/DOCX/XLSX)
  - 文件列表 + 解析状态追踪
  - 交互式知识图谱 (Plotly → 缩放/拖拽/悬停/点击)
  - 4 种关系类型: 前置依赖/并列/包含/应用
  - RAG 问答 + [教材名, 页码] 引用
  - 整合仪表盘 (压缩比统计)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "backend"))

env_file = ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

import gradio as gr
import plotly.graph_objects as go
import networkx as nx
import numpy as np

DATA_DIR = ROOT / "data" / "processed"
UPLOAD_DIR = ROOT / "data" / "textbooks"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Mock data ──────────────────────────────────────────

MOCK_GRAPH = {
    "nodes": [
        {"id":"n1","name":"细胞膜","definition":"由脂质双分子层和镶嵌蛋白质构成的细胞外层结构","importance":"关键","textbook":"生理学","chapter":"第二章","page":18},
        {"id":"n2","name":"静息电位","definition":"细胞未受刺激时膜内外相对稳定的电位差","importance":"关键","textbook":"生理学","chapter":"第二章","page":25},
        {"id":"n3","name":"动作电位","definition":"细胞受刺激后膜电位发生的快速去极化-反极化-复极化过程","importance":"关键","textbook":"生理学","chapter":"第二章","page":30},
        {"id":"n4","name":"钠钾泵","definition":"利用ATP水解能量将Na⁺泵出、K⁺泵入细胞的跨膜蛋白质","importance":"重要","textbook":"生理学","chapter":"第二章","page":22},
        {"id":"n5","name":"离子通道","definition":"细胞膜上控制特定离子跨膜流动的跨膜蛋白质","importance":"重要","textbook":"生理学","chapter":"第二章","page":19},
        {"id":"n6","name":"细胞适应","definition":"细胞为应对环境变化而发生的形态和功能改变","importance":"关键","textbook":"病理学","chapter":"第一章","page":3},
        {"id":"n7","name":"充血","definition":"器官组织血管内血液含量增多","importance":"重要","textbook":"病理学","chapter":"第二章","page":24},
        {"id":"n8","name":"钠离子通道","definition":"选择性允许钠离子通过的跨膜通道蛋白","importance":"重要","textbook":"生理学","chapter":"第二章","page":19},
    ],
    "edges": [
        {"source":"n1","target":"n5","relation_type":"contains","description":"离子通道是细胞膜的组成结构"},
        {"source":"n4","target":"n2","relation_type":"prerequisite","description":"钠钾泵维持的离子梯度是静息电位的基础"},
        {"source":"n2","target":"n3","relation_type":"prerequisite","description":"静息电位是理解动作电位的前提"},
        {"source":"n5","target":"n3","relation_type":"prerequisite","description":"钠通道开放是动作电位去极化的原因"},
        {"source":"n4","target":"n3","relation_type":"applies_to","description":"钠钾泵异常直接影响动作电位产生"},
        {"source":"n5","target":"n8","relation_type":"contains","description":"钠离子通道是离子通道的亚型"},
        {"source":"n6","target":"n1","relation_type":"parallel","description":"细胞适应与细胞膜功能同为细胞基础机制"},
    ],
}

MOCK_STATS = {
    "textbook_count": 2, "original_nodes": 112, "merged_nodes": 34,
    "original_chars": 677000, "merged_chars": 189000, "compression_ratio": 0.279,
    "total_decisions": 78, "merge_count": 42, "keep_count": 28, "remove_count": 8,
}


# ── Data helpers ───────────────────────────────────────

def _load_json(name: str):
    p = DATA_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _has_real_data():
    return (DATA_DIR / "textbook_metadata.json").exists()


_parsed_textbooks: list = []  # global cache of parsed TextbookOut objects


def load_graph_data() -> dict:
    """Load graph data — prefer LLM-extracted KG, fall back to chapter graph, then mock."""
    # Build ID → name mapping
    meta = _load_json("textbook_metadata.json")
    id_to_name = {}
    if meta and isinstance(meta, list):
        for m in meta:
            id_to_name[m.get("textbook_id", "")] = m.get("title", m.get("filename", "?"))

    # 1st priority: integration result (real extracted knowledge points)
    integ = _load_json("integration_result.json")
    if integ and isinstance(integ, dict) and "merged_knowledge_graph" in integ:
        g = integ["merged_knowledge_graph"]
        for n in g.get("nodes", []):
            nid = n.get("textbook", n.get("textbook_id", ""))
            if nid in id_to_name:
                n["textbook"] = id_to_name[nid]
        if g.get("nodes"):
            return g

    # 2nd: initial LLM-extracted knowledge graph
    kg = _load_json("initial_knowledge_graph.json")
    if kg and isinstance(kg, list):
        nodes, edges = [], []
        for b in kg:
            bid = b.get("textbook_id", "?")
            name = id_to_name.get(bid, bid)
            for n in b.get("nodes", []):
                n["textbook"] = name
                nodes.append(n)
            edges.extend(b.get("edges", []))
        if nodes:
            return {"nodes": nodes, "edges": edges}

    # 3rd: quick chapter-based graph from uploaded textbooks
    if _parsed_textbooks:
        return _build_graph_from_textbooks(_parsed_textbooks)

    return MOCK_GRAPH


# ── File upload & parse ────────────────────────────────

async def handle_upload(files):
    """Parse uploaded files and return status list."""
    if not files:
        return "请上传教材文件", gr.update(), gr.update()

    from agents.parser import parse_textbook

    results = []
    for f in files:
        if hasattr(f, 'name'):
            fname = Path(f.name).name
            fpath = str(UPLOAD_DIR / fname)
        else:
            fname = Path(f).name
            fpath = str(f)

        size = Path(fpath).stat().st_size if Path(fpath).exists() else 0
        try:
            tb = await parse_textbook(fpath, fname)
            results.append({
                "filename": fname,
                "format": Path(fname).suffix.upper(),
                "size_mb": round(size / 1024 / 1024, 1),
                "status": "✅ 已完成",
                "pages": tb.total_pages,
                "chars": tb.total_chars,
                "chapters": len(tb.chapters),
                "textbook": tb,
            })
        except Exception as e:
            results.append({
                "filename": fname,
                "format": Path(fname).suffix.upper(),
                "size_mb": round(size / 1024 / 1024, 1) if size else "?",
                "status": f"❌ 失败: {str(e)[:60]}",
                "pages": 0, "chars": 0, "chapters": 0,
                "textbook": None,
            })

    # Build file list HTML
    rows = ""
    for r in results:
        rows += f"""<tr style="border-bottom:1px solid #f1f5f9">
            <td style="padding:4px 8px;font-size:13px">{r['filename']}</td>
            <td style="padding:4px 8px;font-size:12px;color:#64748b">{r['format']}</td>
            <td style="padding:4px 8px;font-size:12px">{r['size_mb']} MB</td>
            <td style="padding:4px 8px;font-size:12px">{r['status']}</td></tr>"""

    file_html = f"""<table style="width:100%;border-collapse:collapse;font-family:monospace">
    <tr style="background:#f8fafc"><th style="text-align:left;padding:6px 8px;font-size:12px">文件名</th>
    <th style="text-align:left;padding:6px 8px;font-size:12px">格式</th>
    <th style="text-align:left;padding:6px 8px;font-size:12px">大小</th>
    <th style="text-align:left;padding:6px 8px;font-size:12px">状态</th></tr>{rows}</table>"""

    # Store globally and build graph
    global _parsed_textbooks
    new_books = [r["textbook"] for r in results if r["textbook"]]
    _parsed_textbooks.extend(new_books)
    graph = load_graph_data()

    # Textbook selector choices
    tb_choices = [tb.title for tb in _parsed_textbooks]
    return file_html, _make_plotly(graph), _build_dashboard(_parsed_textbooks), gr.CheckboxGroup(choices=tb_choices, value=tb_choices)


def _build_graph_from_textbooks(textbooks) -> dict:
    """Quick single-book KG extraction from parsed textbooks."""
    if not textbooks:
        return MOCK_GRAPH
    # For demo: extract basic nodes from chapter titles
    nodes, edges = [], []
    for tb in textbooks:
        for i, ch in enumerate(tb.chapters[:10]):
            node_id = f"{tb.textbook_id}_ch{i}"
            nodes.append({
                "id": node_id, "name": ch.title[:20],
                "definition": ch.content[:120] if ch.content else ch.title,
                "importance": "重要", "textbook": tb.title,
                "chapter": ch.title, "page": ch.page_start,
            })
            if i > 0:
                edges.append({
                    "source": f"{tb.textbook_id}_ch{i-1}",
                    "target": node_id,
                    "relation_type": "prerequisite",
                    "description": f"前置章节: {tb.chapters[i-1].title[:20]}",
                })
    return {"nodes": nodes, "edges": edges} if nodes else MOCK_GRAPH


# ── Interactive Knowledge Graph (Plotly + HTML) ─────────

def _make_plotly(graph_data: dict):
    """Build interactive Plotly figure rendered via gr.Plot (lightweight)."""
    from collections import Counter

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            title="暂无知识点 — 请上传教材",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='#f8fafc', paper_bgcolor='#f8fafc', height=520,
        )
        return fig

    name_counts = Counter(n.get("name", "") for n in nodes)
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"])
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    palette = ['#8b5cf6','#3b82f6','#10b981','#f59e0b','#ef4444','#ec4899','#06b6d4']
    base_size = {"关键": 24, "重要": 17, "补充": 10}
    tb_to_color = {}
    ci = 0

    line_styles = {
        "prerequisite": dict(color='#ef4444', dash='solid', width=1.8),
        "parallel":     dict(color='#3b82f6', dash='dash', width=1.4),
        "contains":     dict(color='#10b981', dash='dot', width=1.4),
        "applies_to":   dict(color='#f59e0b', dash='dashdot', width=1.4),
    }

    fig = go.Figure()

    # Edges
    for e in edges:
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            rt = e.get("relation_type", "parallel")
            style = line_styles.get(rt, line_styles["parallel"])
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines', line=style,
                hoverinfo='text', text=f"<b>{rt}</b>: {e.get('description','')}",
                showlegend=False,
            ))

    # Nodes per textbook
    for n in nodes:
        tb = n.get("textbook", "?")
        if tb not in tb_to_color:
            tb_to_color[tb] = palette[ci % len(palette)]
            ci += 1

    for tb, color in tb_to_color.items():
        nds = [n for n in nodes if n.get("textbook", "?") == tb and n["id"] in pos]
        if not nds:
            continue
        xs = [pos[n["id"]][0] for n in nds]
        ys = [pos[n["id"]][1] for n in nds]
        names = [n["name"] for n in nds]
        defs = [n.get("definition", "")[:60] for n in nds]
        imps = [n.get("importance", "重要") for n in nds]
        freqs = [name_counts.get(n.get("name", ""), 1) for n in nds]
        sizes = [base_size.get(imp, 15) * (1 + 0.3 * (f - 1)) for imp, f in zip(imps, freqs)]
        chs = [n.get("chapter", "") for n in nds]
        pgs = [n.get("page", "") for n in nds]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=sizes, color=color, line=dict(width=2, color='white'), opacity=0.92),
            text=[name[:5] for name in names],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='sans-serif'),
            name=tb,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "<span style='font-size:12px'>%{customdata[1]}</span><br>"
                "<i style='font-size:11px'>%{customdata[2]} · %{customdata[3]}本 · %{customdata[4]}</i>"
                "<extra></extra>"
            ),
            customdata=[[n, d, f"{imp}·频{freq}次", f"{ch}"] for n, d, imp, freq, ch
                        in zip(names, defs, imps, freqs, chs)],
        ))

    # Legend for relations
    for rt, style in line_styles.items():
        label = {"prerequisite":"前置依赖","parallel":"并列关系",
                 "contains":"包含关系","applies_to":"应用关系"}.get(rt, rt)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=style, name=label, showlegend=True,
        ))

    fig.update_layout(
        title=f"知识图谱 — {len(nodes)} 节点 · {len(edges)} 关系",
        showlegend=True,
        legend=dict(orientation='h', y=1.08, x=0, font=dict(size=11)),
        hovermode='closest',
        plot_bgcolor='#f8fafc', paper_bgcolor='#f8fafc',
        height=540, margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        dragmode='pan',
    )
    return fig


# ── Dashboard ──────────────────────────────────────────

def _build_dashboard(parsed_textbooks=None) -> str:
    integ = _load_json("integration_result.json")
    s = integ.get("stats", MOCK_STATS) if integ else MOCK_STATS

    if parsed_textbooks:
        books = len(parsed_textbooks)
        pages = sum(t.total_pages for t in parsed_textbooks)
        chars = sum(t.total_chars for t in parsed_textbooks)
    else:
        meta = _load_json("textbook_metadata.json")
        if meta:
            books = len(meta)
            pages = sum(b.get("total_pages", 0) for b in meta)
            chars = sum(b.get("total_chars", 0) for b in meta)
        else:
            books, pages, chars = 2, 900, 677000

    ratio = s['compression_ratio']
    color = "#059669" if ratio <= 0.32 else "#f59e0b"

    return f"""
    <div style="font-family:system-ui,sans-serif">
      <div style="background:#fff;border-radius:12px;padding:16px;margin:6px 0;border:1px solid #e2e8f0">
        <h3 style="color:#2563eb;margin:0 0 10px;font-size:15px">📚 教材概览</h3>
        <p style="margin:4px 0;color:#334155"><b style="color:#0f172a;font-size:16px">{books}</b> 本 &nbsp;
        <b style="color:#0f172a;font-size:16px">{pages}</b> 页 &nbsp;
        <b style="color:#0f172a;font-size:16px">{chars:,}</b> 字</p>
      </div>
      <div style="background:#fff;border-radius:12px;padding:16px;margin:6px 0;border:1px solid #e2e8f0">
        <h3 style="color:#7c3aed;margin:0 0 10px;font-size:15px">📊 整合统计</h3>
        <table style="width:100%;color:#334155;font-size:13px">
          <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:4px 0">节点</td><td style="text-align:right;font-weight:600">{s['original_nodes']} → {s['merged_nodes']}</td></tr>
          <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:4px 0">字数</td><td style="text-align:right;font-weight:600">{s['original_chars']:,} → {s['merged_chars']:,}</td></tr>
          <tr style="border-bottom:1px solid #f1f5f9;background:#f5f3ff">
            <td style="padding:6px 0"><b style="color:#6d28d9">🎯 压缩比</b></td>
            <td style="text-align:right"><b style="color:{color};font-size:18px">{ratio:.1%}</b></td></tr>
          <tr><td style="padding:4px 0">merge / keep / remove</td><td style="text-align:right;font-weight:600">{s['merge_count']} / {s['keep_count']} / {s['remove_count']}</td></tr>
        </table>
      </div>
    </div>"""


async def rag_query(question: str) -> tuple[str, str, str]:
    """Returns (answer, citations_html, index_status)"""
    if not question.strip():
        return "", "", ""

    # Index status line
    idx_status = "索引状态: 🎭 演示模式"
    if _has_real_data():
        try:
            from agents.rag import get_status
            st = await get_status()
            idx_status = f"✅ 已索引 {st.indexed_books} 本教材，共 {st.total_chunks} 个知识块 | 模型: {st.embedding_model}"
        except Exception:
            idx_status = "索引状态: ⚠️ 未构建"

    if not _has_real_data():
        demo_answer = f"### {question}\n\n（🎭 演示模式。正式部署后基于 FAISS 返回带 **[教材名, 第X章, 第X页]** 引用的回答。）"
        demo_cites = """
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin:8px 0">
<details open>
<summary style="font-weight:600;font-size:14px;color:#0f172a;cursor:pointer">📚 引用来源（2 条·演示）</summary>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin:8px 0">
<thead><tr style="background:#f8fafc">
<th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">#</th>
<th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">教材</th>
<th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">章节</th>
<th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">页码</th>
<th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">相关度</th>
</tr></thead>
<tbody>
<tr><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">1</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;font-weight:500">生理学</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">第二章 细胞的基本功能</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">p25</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;color:#059669">0.95</td></tr>
<tr><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">2</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;font-weight:500">病理学</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">第一章 细胞和组织的适应与损伤</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">p3</td><td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;color:#059669">0.87</td></tr>
</tbody></table>
<details style="margin:6px 0"><summary style="cursor:pointer;color:#7c3aed;font-size:13px">📄 原文片段 1 — 生理学</summary><blockquote style="background:#f8fafc;border-left:3px solid #8b5cf6;padding:8px 12px;margin:6px 0;font-size:13px;line-height:1.6;color:#334155">静息电位是指细胞在静息状态下膜内外的电位差，通常膜内为负、膜外为正。钠钾泵通过主动转运维持细胞内外钠钾离子的浓度梯度...</blockquote></details>
<details style="margin:6px 0"><summary style="cursor:pointer;color:#7c3aed;font-size:13px">📄 原文片段 2 — 病理学</summary><blockquote style="background:#f8fafc;border-left:3px solid #8b5cf6;padding:8px 12px;margin:6px 0;font-size:13px;line-height:1.6;color:#334155">细胞适应是细胞为应对环境变化而发生的形态和功能改变，包括肥大、增生、萎缩和化生四种类型...</blockquote></details>
</details>
</div>"""
        return demo_answer, demo_cites, idx_status

    try:
        from agents.rag import query
        from models.schemas import RAGQueryIn
        result = await query(RAGQueryIn(question=question))

        # Build citation table (pure HTML for gr.HTML)
        cite_rows = ""
        for i, c in enumerate(result.citations, 1):
            cite_rows += f"""<tr>
<td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">{i}</td>
<td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;font-weight:500">{c.textbook}</td>
<td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">{c.chapter}</td>
<td style="padding:4px 8px;border-bottom:1px solid #f1f5f9">p{c.page}</td>
<td style="padding:4px 8px;border-bottom:1px solid #f1f5f9;color:#059669">{c.relevance_score:.2f}</td>
</tr>"""

        # Build expandable chunk viewer
        chunks_html = ""
        for i, chunk in enumerate(result.source_chunks, 1):
            text = getattr(chunk, 'text', str(chunk))[:500]
            chunks_html += f"""
<details style="margin:6px 0">
<summary style="cursor:pointer;color:#7c3aed;font-size:13px;padding:4px 0">
  📄 原文片段 {i} — {result.citations[i-1].textbook if i <= len(result.citations) else '?'}
</summary>
<blockquote style="background:#f8fafc;border-left:3px solid #8b5cf6;padding:8px 12px;margin:6px 0;font-size:13px;line-height:1.6;white-space:pre-wrap;color:#334155">{text}</blockquote>
</details>"""

        cites_html = f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin:8px 0">
<details open>
<summary style="font-weight:600;font-size:14px;color:#0f172a;cursor:pointer;margin-bottom:10px">
  📚 引用来源（{len(result.citations)} 条）
</summary>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin:8px 0">
<thead><tr style="background:#f8fafc">
  <th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">#</th>
  <th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">教材</th>
  <th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">章节</th>
  <th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">页码</th>
  <th style="text-align:left;padding:6px 8px;font-size:12px;color:#64748b">相关度</th>
</tr></thead>
<tbody>{cite_rows}</tbody>
</table>
{chunks_html}
</details>
</div>"""

        return result.answer, cites_html, idx_status
    except Exception as e:
        return f"❌ {e}", "", idx_status


# ── UI ─────────────────────────────────────────────────

css = ".gradio-container{max-width:1500px!important;margin:0 auto}footer{display:none!important}"

with gr.Blocks(css=css, title="学科知识整合智能体") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:18px 0 6px">
      <h1 style="font-size:1.6rem;font-weight:700;color:#0f172a;margin:0">🧠 学科知识整合智能体</h1>
      <p style="color:#64748b;font-size:0.85rem;margin:4px 0 0">多教材解析 · 知识图谱 · 跨书去重 ≤30% · RAG 引用溯源</p>
    </div>""")

    with gr.Row(equal_height=False):
        # ── LEFT ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📁 教材管理")
            tb_selector = gr.CheckboxGroup(
                label="选择教材",
                choices=[],
                value=[],
            )
            uploader = gr.File(
                label="拖拽或点击上传教材 (PDF/MD/TXT/DOCX/XLSX)",
                file_count="multiple",
                file_types=[".pdf", ".md", ".txt", ".docx", ".xlsx"],
            )
            btn_upload = gr.Button("🔍 解析文件", variant="primary", size="sm")
            file_status = gr.HTML("""
            <p style="color:#94a3b8;font-size:13px">尚未上传教材。支持批量上传，解析状态将在此显示。</p>
            """)
            gr.Markdown("---")
            dashboard = gr.HTML(_build_dashboard())

        # ── CENTER ────────────────────────────────────
        with gr.Column(scale=2, min_width=520):
            gr.Markdown("### 🗺️ 交互式知识图谱  \n"
                        "<small>🖱️ 悬停看详情 &nbsp; 🔍 缩放拖拽 &nbsp; 🔴前置 🔵并列 🟢包含 🟠应用</small>")
            graph_display = gr.Plot(_make_plotly(MOCK_GRAPH))
            node_list = gr.Dropdown(
                label="🔍 搜索/选择知识点",
                choices=[n["name"] for n in MOCK_GRAPH["nodes"]],
                interactive=True,
                allow_custom_value=True,
            )
            node_detail = gr.Markdown("")

        # ── RIGHT ─────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 💬 RAG 问答")
            idx_status = gr.Markdown("*索引状态: 加载中...*")
            question = gr.Textbox(label="", placeholder="输入问题，例如：什么是动作电位？", lines=3)
            btn_ask = gr.Button("🔎 查询", variant="primary", size="sm")
            answer = gr.Textbox(label="回答", lines=8, interactive=False)
            cites = gr.HTML("")

    # Events
    def on_select_textbooks(selected_titles):
        """Rebuild graph from selected textbooks only."""
        global _parsed_textbooks
        selected = [t for t in _parsed_textbooks if t.title in (selected_titles or [])]
        graph = _build_graph_from_textbooks(selected) if selected else MOCK_GRAPH
        nodes = graph.get("nodes", [])
        return _make_plotly(graph), gr.Dropdown(choices=[n["name"] for n in nodes]), _build_dashboard(selected)

    tb_selector.change(
        fn=on_select_textbooks, inputs=[tb_selector],
        outputs=[graph_display, node_list, dashboard],
    )

    btn_upload.click(
        fn=handle_upload, inputs=[uploader],
        outputs=[file_status, graph_display, dashboard, tb_selector],
    ).then(
        fn=lambda: (gr.Dropdown(choices=[n["name"] for n in load_graph_data().get("nodes", [])]), "", ""),
        outputs=[node_list, node_detail, question],
    )
    # Node select → show detail + auto-fill RAG
    def show_node_detail(node_name: str):
        graph_data = load_graph_data()
        nodes = graph_data.get("nodes", [])
        for n in nodes:
            if n.get("name") == node_name:
                detail = f"""### {n['name']}\n\n{n.get('definition','')}\n\n---\n**教材**: {n.get('textbook','?')} | **章节**: {n.get('chapter','?')} | **页码**: p{n.get('page','?')}\n\n**重要性**: {n.get('importance','?')}"""
                question = f"请详细解释「{n['name']}」的概念、定义和作用"
                return detail, question
        return "", f"请解释「{node_name}」"

    node_list.change(
        fn=show_node_detail, inputs=[node_list],
        outputs=[node_detail, question],
    ).then(
        fn=rag_query, inputs=[question], outputs=[answer, cites, idx_status],
    )

    # Refresh graph on upload
    def update_graph_on_upload(graph_data_dict):
        graph = graph_data_dict if graph_data_dict else MOCK_GRAPH
        return _make_plotly(graph), gr.Dropdown(choices=[n["name"] for n in graph.get("nodes", [])])

    btn_ask.click(
        fn=rag_query, inputs=[question],
        outputs=[answer, cites, idx_status],
    )

def _startup_index():
    """Auto-rebuild FAISS index on startup."""
    if not _has_real_data():
        return
    try:
        from models.schemas import TextbookOut
        from agents.rag import build_index
        import asyncio
        meta = _load_json("textbook_metadata.json")
        books = [TextbookOut(**m) for m in meta]
        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(build_index(books))
        loop.close()
        print(f"  📇 FAISS 就绪: {status.indexed_books} 本, {status.total_chunks} 分块")
    except Exception as e:
        print(f"  ⚠️  索引跳过: {e}")


if __name__ == "__main__":
    print(f"🚀 启动中...")
    _startup_index()
    print(f"🌐 http://0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
