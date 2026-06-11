"""学科知识整合智能体 — 完整前端（美化版）。

Feature:
  - 现代玻璃态设计系统 (Glassmorphism)
  - 渐变色主题 · 流畅动画 · 专业排版
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


# ═══════════════════════════════════════════════════════════════
# Design System — Custom CSS
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --primary-light: #818cf8;
  --accent: #8b5cf6;
  --accent-light: #a78bfa;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --info: #3b82f6;
  --bg: #f1f5f9;
  --bg-card: #ffffff;
  --bg-glass: rgba(255,255,255,0.72);
  --text: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --border-light: #f1f5f9;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.04);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.06), 0 4px 10px rgba(0,0,0,0.04);
  --shadow-xl: 0 20px 40px rgba(0,0,0,0.08);
  --radius-sm: 8px;
  --radius: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;
  --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ── */
.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  font-family: 'Inter', 'Noto Sans SC', system-ui, -apple-system, sans-serif !important;
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #faf5ff 100%) !important;
  min-height: 100vh;
}
footer { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Header Area ── */
.app-header {
  position: relative;
  text-align: center;
  padding: 32px 20px 20px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a78bfa 100%);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}
.app-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.1) 0%, transparent 60%);
  pointer-events: none;
}
.app-header h1 {
  font-size: 1.8rem;
  font-weight: 800;
  color: #ffffff;
  margin: 0 0 6px;
  letter-spacing: -0.02em;
  position: relative;
  text-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.app-header p {
  font-size: 0.9rem;
  color: rgba(255,255,255,0.85);
  margin: 0;
  font-weight: 400;
  position: relative;
}
.app-header .header-badge {
  display: inline-block;
  padding: 4px 14px;
  margin: 0 4px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
  background: rgba(255,255,255,0.2);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.25);
  color: #fff;
}

/* ── Section Headers ── */
.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--border-light);
}
.section-title .icon-circle {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  box-shadow: var(--shadow-sm);
}
.section-title .icon-purple { background: linear-gradient(135deg, #ede9fe, #ddd6fe); }
.section-title .icon-blue   { background: linear-gradient(135deg, #dbeafe, #bfdbfe); }
.section-title .icon-green  { background: linear-gradient(135deg, #d1fae5, #a7f3d0); }
.section-title .icon-amber  { background: linear-gradient(135deg, #fef3c7, #fde68a); }
.section-title h3 { margin: 0; font-size: 1rem; font-weight: 700; color: var(--text); }
.section-title .subtitle { font-size: 0.75rem; color: var(--text-muted); margin-left: auto; }

/* ── Cards (Glassmorphism) ── */
.glass-card {
  background: var(--bg-glass);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.7);
  border-radius: var(--radius-lg);
  padding: 20px;
  margin-bottom: 12px;
  box-shadow: var(--shadow-md);
  transition: all var(--transition);
}
.glass-card:hover {
  box-shadow: var(--shadow-lg);
  border-color: rgba(99,102,241,0.15);
  transform: translateY(-1px);
}
.glass-card .card-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 6px;
  font-weight: 600;
}
.glass-card .card-value {
  font-size: 1.6rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.02em;
}
.glass-card .card-value-sm {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--text);
}
.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-bottom: 1px solid var(--border-light);
  font-size: 0.85rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--text-secondary); }
.stat-num { font-weight: 600; color: var(--text); }

/* ── Compression Badge ── */
.compression-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px;
  border-radius: 24px;
  font-weight: 700;
  font-size: 1.1rem;
}
.compression-badge.good { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.compression-badge.warn { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }

/* ── File Table ── */
.file-table {
  width: 100%; border-collapse: collapse;
  font-size: 0.8rem;
}
.file-table th {
  text-align: left; padding: 8px 10px;
  font-size: 0.7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--text-muted); background: #f8fafc;
  border-bottom: 2px solid var(--border);
}
.file-table td {
  padding: 8px 10px; border-bottom: 1px solid var(--border-light);
  color: var(--text-secondary);
}
.file-table tr:hover td { background: #f8fafc; }
.file-table .file-name { font-weight: 500; color: var(--text); }
.file-table .status-ok { color: var(--success); font-weight: 500; }
.file-table .status-err { color: var(--danger); font-weight: 500; }

/* ── Citation Panel ── */
.cite-panel {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin: 10px 0;
  box-shadow: var(--shadow-sm);
}
.cite-panel summary {
  font-weight: 600; font-size: 0.88rem; color: var(--text);
  cursor: pointer; padding: 4px 0;
}
.cite-panel summary:hover { color: var(--primary); }
.cite-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 10px 0; }
.cite-table th {
  text-align: left; padding: 6px 10px;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--text-muted); background: #f8fafc;
}
.cite-table td { padding: 6px 10px; border-bottom: 1px solid var(--border-light); }
.cite-score { color: var(--success); font-weight: 600; }
.chunk-block {
  background: #f8fafc; border-left: 3px solid var(--accent);
  padding: 10px 14px; margin: 6px 0;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-size: 0.82rem; line-height: 1.7; color: var(--text-secondary);
}
.chunk-details { margin: 6px 0; }
.chunk-details summary { cursor: pointer; color: var(--primary); font-size: 0.82rem; font-weight: 500; }
.chunk-details summary:hover { color: var(--primary-dark); }

/* ── Graph Container ── */
.graph-wrapper {
  background: #ffffff;
  border-radius: var(--radius-lg);
  padding: 8px;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border);
}

/* ── Empty State ── */
.empty-state {
  text-align: center; padding: 32px 20px; color: var(--text-muted);
}
.empty-state .empty-icon { font-size: 2.5rem; margin-bottom: 8px; opacity: 0.5; }
.empty-state p { font-size: 0.85rem; margin: 0; }

/* ── Input & Button overrides ── */
.gr-textbox textarea, .gr-textbox input {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  font-family: 'Inter', 'Noto Sans SC', sans-serif !important;
  transition: all var(--transition) !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
button.primary {
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
  transition: all var(--transition) !important;
}
button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
}
.gr-checkbox-group {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
}

/* ── Responsive tweaks ── */
@media (max-width: 1200px) {
  .app-header h1 { font-size: 1.4rem; }
}
"""


# ═══════════════════════════════════════════════════════════════
# Mock data
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════

def _load_json(name: str):
    p = DATA_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _has_real_data():
    return (DATA_DIR / "textbook_metadata.json").exists()


_parsed_textbooks: list = []


def load_graph_data() -> dict:
    meta = _load_json("textbook_metadata.json")
    id_to_name = {}
    if meta and isinstance(meta, list):
        for m in meta:
            id_to_name[m.get("textbook_id", "")] = m.get("title", m.get("filename", "?"))

    integ = _load_json("integration_result.json")
    if integ and isinstance(integ, dict) and "merged_knowledge_graph" in integ:
        g = integ["merged_knowledge_graph"]
        for n in g.get("nodes", []):
            nid = n.get("textbook", n.get("textbook_id", ""))
            if nid in id_to_name:
                n["textbook"] = id_to_name[nid]
        if g.get("nodes"):
            return g

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

    if _parsed_textbooks:
        return _build_graph_from_textbooks(_parsed_textbooks)

    return MOCK_GRAPH


# ═══════════════════════════════════════════════════════════════
# File upload & parse
# ═══════════════════════════════════════════════════════════════

async def handle_upload(files):
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
                "filename": fname, "format": Path(fname).suffix.upper(),
                "size_mb": round(size / 1024 / 1024, 1),
                "status": "ok", "status_text": "已完成",
                "pages": tb.total_pages, "chars": tb.total_chars,
                "chapters": len(tb.chapters), "textbook": tb,
            })
        except Exception as e:
            results.append({
                "filename": fname, "format": Path(fname).suffix.upper(),
                "size_mb": round(size / 1024 / 1024, 1) if size else "?",
                "status": "err", "status_text": str(e)[:60],
                "pages": 0, "chars": 0, "chapters": 0, "textbook": None,
            })

    rows = ""
    for r in results:
        status_cls = "status-ok" if r["status"] == "ok" else "status-err"
        status_icon = "✓" if r["status"] == "ok" else "✗"
        rows += f"""<tr>
            <td class="file-name">{r['filename']}</td>
            <td>{r['format']}</td>
            <td>{r['size_mb']} MB</td>
            <td class="{status_cls}">{status_icon} {r['status_text']}</td></tr>"""

    file_html = f"""<table class="file-table">
    <thead><tr><th>文件名</th><th>格式</th><th>大小</th><th>状态</th></tr></thead>
    <tbody>{rows}</tbody></table>"""

    global _parsed_textbooks
    new_books = [r["textbook"] for r in results if r["textbook"]]
    _parsed_textbooks.extend(new_books)
    graph = load_graph_data()

    tb_choices = [tb.title for tb in _parsed_textbooks]
    return file_html, _make_plotly(graph), _build_dashboard(_parsed_textbooks), gr.CheckboxGroup(choices=tb_choices, value=tb_choices)


def _build_graph_from_textbooks(textbooks) -> dict:
    if not textbooks:
        return MOCK_GRAPH
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


# ═══════════════════════════════════════════════════════════════
# Interactive Knowledge Graph (Plotly)
# ═══════════════════════════════════════════════════════════════

def _make_plotly(graph_data: dict):
    from collections import Counter

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="暂无知识点 — 请上传教材", font=dict(size=15, color="#94a3b8")),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', height=520,
        )
        return fig

    name_counts = Counter(n.get("name", "") for n in nodes)
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"])
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    palette = ['#6366f1','#3b82f6','#10b981','#f59e0b','#ef4444','#ec4899','#06b6d4']
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

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=sizes, color=color, line=dict(width=2, color='white'), opacity=0.92),
            text=[name[:5] for name in names],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Inter, sans-serif'),
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

    for rt, style in line_styles.items():
        label = {"prerequisite":"前置依赖","parallel":"并列关系",
                 "contains":"包含关系","applies_to":"应用关系"}.get(rt, rt)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=style, name=label, showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=f"知识图谱 — {len(nodes)} 节点 · {len(edges)} 关系", font=dict(size=14, color='#334155')),
        showlegend=True,
        legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10, color='#64748b')),
        hovermode='closest',
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        height=540, margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        dragmode='pan',
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Dashboard
# ═══════════════════════════════════════════════════════════════

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
    badge_cls = "good" if ratio <= 0.32 else "warn"

    return f"""
    <div style="font-family:'Inter','Noto Sans SC',system-ui,sans-serif">

      <div class="glass-card">
        <div class="card-label">教材概览</div>
        <div style="display:flex;gap:16px;margin:8px 0">
          <div><div class="card-value">{books}</div><div style="font-size:0.75rem;color:#94a3b8">本教材</div></div>
          <div><div class="card-value">{pages}</div><div style="font-size:0.75rem;color:#94a3b8">页</div></div>
          <div><div class="card-value-sm">{chars:,}</div><div style="font-size:0.75rem;color:#94a3b8">字</div></div>
        </div>
      </div>

      <div class="glass-card">
        <div class="card-label">整合统计</div>
        <div class="stat-row"><span class="stat-label">节点</span><span class="stat-num">{s['original_nodes']} → {s['merged_nodes']}</span></div>
        <div class="stat-row"><span class="stat-label">字数</span><span class="stat-num">{s['original_chars']:,} → {s['merged_chars']:,}</span></div>
        <div class="stat-row" style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);border-radius:8px;padding:10px 12px;margin:4px -4px">
          <span style="font-weight:700;color:#6d28d9;font-size:0.9rem">压缩比</span>
          <span class="compression-badge {badge_cls}">{ratio:.1%}</span>
        </div>
        <div class="stat-row"><span class="stat-label">merge / keep / remove</span><span class="stat-num">{s['merge_count']} / {s['keep_count']} / {s['remove_count']}</span></div>
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════
# RAG Query
# ═══════════════════════════════════════════════════════════════

async def rag_query(question: str) -> tuple[str, str, str]:
    if not question.strip():
        return "", "", ""

    idx_status = "索引状态: 演示模式"
    if _has_real_data():
        try:
            from agents.rag import get_status
            st = await get_status()
            idx_status = f"已索引 {st.indexed_books} 本教材，共 {st.total_chunks} 个知识块 | 模型: {st.embedding_model}"
        except Exception:
            idx_status = "索引状态: 未构建"

    if not _has_real_data():
        demo_answer = f"### {question}\n\n（演示模式。正式部署后基于 FAISS 返回带 **[教材名, 第X章, 第X页]** 引用的回答。）"
        demo_cites = f"""
<div class="cite-panel">
<details open>
<summary>引用来源（2 条 · 演示）</summary>
<table class="cite-table">
<thead><tr><th>#</th><th>教材</th><th>章节</th><th>页码</th><th>相关度</th></tr></thead>
<tbody>
<tr><td>1</td><td style="font-weight:500">生理学</td><td>第二章 细胞的基本功能</td><td>p25</td><td class="cite-score">0.95</td></tr>
<tr><td>2</td><td style="font-weight:500">病理学</td><td>第一章 细胞和组织的适应与损伤</td><td>p3</td><td class="cite-score">0.87</td></tr>
</tbody></table>
<details class="chunk-details"><summary>原文片段 1 — 生理学</summary>
<div class="chunk-block">静息电位是指细胞在静息状态下膜内外的电位差，通常膜内为负、膜外为正。钠钾泵通过主动转运维持细胞内外钠钾离子的浓度梯度...</div>
</details>
<details class="chunk-details"><summary>原文片段 2 — 病理学</summary>
<div class="chunk-block">细胞适应是细胞为应对环境变化而发生的形态和功能改变，包括肥大、增生、萎缩和化生四种类型...</div>
</details>
</details>
</div>"""
        return demo_answer, demo_cites, idx_status

    try:
        from agents.rag import query
        from models.schemas import RAGQueryIn
        result = await query(RAGQueryIn(question=question))

        cite_rows = ""
        for i, c in enumerate(result.citations, 1):
            cite_rows += f"""<tr>
<td>{i}</td>
<td style="font-weight:500">{c.textbook}</td>
<td>{c.chapter}</td>
<td>p{c.page}</td>
<td class="cite-score">{c.relevance_score:.2f}</td>
</tr>"""

        chunks_html = ""
        for i, chunk in enumerate(result.source_chunks, 1):
            text = getattr(chunk, 'text', str(chunk))[:500]
            cite_label = result.citations[i-1].textbook if i <= len(result.citations) else '?'
            chunks_html += f"""
<details class="chunk-details">
<summary>原文片段 {i} — {cite_label}</summary>
<div class="chunk-block">{text}</div>
</details>"""

        cites_html = f"""
<div class="cite-panel">
<details open>
<summary>引用来源（{len(result.citations)} 条）</summary>
<table class="cite-table">
<thead><tr><th>#</th><th>教材</th><th>章节</th><th>页码</th><th>相关度</th></tr></thead>
<tbody>{cite_rows}</tbody></table>
{chunks_html}
</details>
</div>"""

        return result.answer, cites_html, idx_status
    except Exception as e:
        return f"查询出错: {e}", "", idx_status


# ═══════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════

HEADER_HTML = """
<div class="app-header">
  <h1>学科知识整合智能体</h1>
  <p>5-Agent 协作 &nbsp;·&nbsp; 多教材解析 &nbsp;·&nbsp; 知识图谱 &nbsp;·&nbsp; 跨书去重 &le;30% &nbsp;·&nbsp; RAG 引用溯源</p>
</div>
"""

LEFT_SECTION_HEADER = """
<div class="section-title">
  <div class="icon-circle icon-purple">📁</div>
  <h3>教材管理</h3>
</div>"""

GRAPH_SECTION_HEADER = """
<div class="section-title">
  <div class="icon-circle icon-blue">🗺️</div>
  <h3>交互式知识图谱</h3>
  <span class="subtitle">悬停看详情 · 滚轮缩放 · 拖拽平移</span>
</div>"""

RAG_SECTION_HEADER = """
<div class="section-title">
  <div class="icon-circle icon-green">💬</div>
  <h3>RAG 智能问答</h3>
</div>"""

LEGEND_HTML = """
<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.75rem;color:#64748b;padding:4px 0;margin-top:4px">
  <span>🔴 <b>前置依赖</b></span>
  <span>🔵 <b>并列关系</b></span>
  <span>🟢 <b>包含关系</b></span>
  <span>🟠 <b>应用关系</b></span>
</div>"""


with gr.Blocks(css=CUSTOM_CSS, title="学科知识整合智能体") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        # ── LEFT COLUMN ─────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.HTML(LEFT_SECTION_HEADER)
            tb_selector = gr.CheckboxGroup(
                label="选择教材",
                choices=[],
                value=[],
                elem_classes="gr-checkbox-group",
            )
            uploader = gr.File(
                label="拖拽或点击上传教材",
                file_count="multiple",
                file_types=[".pdf", ".md", ".txt", ".docx", ".xlsx"],
            )
            btn_upload = gr.Button("解析文件", variant="primary", size="sm", elem_classes="primary")
            file_status = gr.HTML("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <p>尚未上传教材<br><span style="font-size:0.75rem">支持 PDF / MD / TXT / DOCX / XLSX 批量上传</span></p>
            </div>""")
            dashboard = gr.HTML(_build_dashboard())

        # ── CENTER COLUMN ────────────────────────────────
        with gr.Column(scale=2, min_width=520):
            gr.HTML(GRAPH_SECTION_HEADER)
            with gr.Group(elem_classes="graph-wrapper"):
                graph_display = gr.Plot(_make_plotly(MOCK_GRAPH))
            gr.HTML(LEGEND_HTML)
            node_list = gr.Dropdown(
                label="搜索/选择知识点",
                choices=[n["name"] for n in MOCK_GRAPH["nodes"]],
                interactive=True,
                allow_custom_value=True,
            )
            node_detail = gr.Markdown("")

        # ── RIGHT COLUMN ─────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.HTML(RAG_SECTION_HEADER)
            idx_status = gr.Markdown("*索引状态: 加载中...*")
            question = gr.Textbox(
                label="输入问题",
                placeholder="例如：什么是动作电位？其产生机制是什么？",
                lines=3,
            )
            btn_ask = gr.Button("查询", variant="primary", size="sm", elem_classes="primary")
            answer = gr.Textbox(label="回答", lines=10, interactive=False)
            cites = gr.HTML("")

    # ── Events ──────────────────────────────────────────

    def on_select_textbooks(selected_titles):
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

    def show_node_detail(node_name: str):
        graph_data = load_graph_data()
        nodes = graph_data.get("nodes", [])
        for n in nodes:
            if n.get("name") == node_name:
                detail = f"""### {n['name']}\n\n{n.get('definition','')}\n\n---\n**教材**: {n.get('textbook','?')} | **章节**: {n.get('chapter','?')} | **页码**: p{n.get('page','?')}\n\n**重要性**: {n.get('importance','?')}"""
                question_text = f"请详细解释「{n['name']}」的概念、定义和作用"
                return detail, question_text
        return "", f"请解释「{node_name}」"

    node_list.change(
        fn=show_node_detail, inputs=[node_list],
        outputs=[node_detail, question],
    ).then(
        fn=rag_query, inputs=[question], outputs=[answer, cites, idx_status],
    )

    btn_ask.click(
        fn=rag_query, inputs=[question],
        outputs=[answer, cites, idx_status],
    )


def _startup_index():
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
