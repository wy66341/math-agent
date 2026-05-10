"""Gradio 前端 — 三栏布局：整合仪表盘 | 知识图谱 | RAG 问答

Auto-loads data from data/processed/ on startup.
Click any graph node → RAG query triggered automatically.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import gradio as gr

# ── Data loader ────────────────────────────────────────

DATA_DIR = ROOT / "data" / "processed"


def _load_json(name: str) -> dict:
    path = DATA_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_dashboard() -> str:
    """Build dashboard HTML from processed data."""
    meta = _load_json("textbook_metadata.json")
    integ = _load_json("integration_result.json")
    kg = _load_json("initial_knowledge_graph.json")

    if isinstance(meta, list):
        books = len(meta)
        pages = sum(b.get("total_pages", 0) for b in meta)
        chars = sum(b.get("total_chars", 0) for b in meta)
        names = [b.get("title", "?") for b in meta[:5]]
    else:
        books, pages, chars, names = 0, 0, 0, []

    if isinstance(integ, dict) and "stats" in integ:
        s = integ["stats"]
        stats_html = f"""
        <div style="background:#ffffff;border-radius:12px;padding:20px;margin:10px 0;border:1px solid #e2e8f0">
          <h3 style="color:#7c3aed;margin:0 0 14px;font-size:16px">📊 整合统计</h3>
          <table style="width:100%;color:#334155;font-size:14px;border-collapse:collapse">
            <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:6px 0">原始节点</td><td style="text-align:right;font-weight:600;color:#0f172a">{s.get('original_nodes','-')}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:6px 0">整合后节点</td><td style="text-align:right;font-weight:600;color:#0f172a">{s.get('merged_nodes','-')}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:6px 0">原始字数</td><td style="text-align:right;font-weight:600;color:#0f172a">{s.get('original_chars',0):,}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:6px 0">整合后字数</td><td style="text-align:right;font-weight:600;color:#0f172a">{s.get('merged_chars',0):,}</td></tr>
            <tr style="border-bottom:1px solid #f1f5f9;background:#f5f3ff">
              <td style="padding:8px 0"><b style="color:#6d28d9;font-size:15px">压缩比</b></td>
              <td style="text-align:right"><b style="color:#6d28d9;font-size:20px">{s.get('compression_ratio',0):.1%}</b></td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9"><td style="padding:6px 0">🟢 merge</td><td style="text-align:right;font-weight:600;color:#059669">{s.get('merge_count','-')}</td></tr>
            <tr><td style="padding:6px 0">keep / remove</td><td style="text-align:right;font-weight:600;color:#0f172a">{s.get('keep_count','-')} / {s.get('remove_count','-')}</td></tr>
          </table>
        </div>"""
    else:
        stats_html = "<p style='color:#64748b;font-style:italic;padding:12px'>暂无整合数据，请先运行 run_pipeline.py</p>"

    html = f"""
    <div style="font-family:system-ui,sans-serif">
      <div style="background:#ffffff;border-radius:12px;padding:20px;margin:10px 0;border:1px solid #e2e8f0">
        <h3 style="color:#2563eb;margin:0 0 14px;font-size:16px">📚 教材概览</h3>
        <p style="margin:6px 0;color:#334155;font-size:15px"><b style="color:#0f172a;font-size:18px">{books}</b> 本教材 | <b style="color:#0f172a">{pages}</b> 页 | <b style="color:#0f172a">{chars:,}</b> 字</p>
        <p style="color:#64748b;font-size:13px;margin:6px 0 0">{', '.join(names[:4])}{'...' if len(names) > 4 else ''}</p>
      </div>
      {stats_html}
    </div>"""
    return html


def load_graph_data() -> dict:
    """Load merged graph data for ECharts."""
    integ = _load_json("integration_result.json")
    if isinstance(integ, dict) and "merged_knowledge_graph" in integ:
        return integ["merged_knowledge_graph"]
    # Fallback: initial graph
    kg = _load_json("initial_knowledge_graph.json")
    if isinstance(kg, list) and len(kg) > 0:
        all_nodes = []
        all_edges = []
        for book_data in kg:
            for n in book_data.get("nodes", []):
                n["textbook"] = book_data.get("textbook_id", "?")
                all_nodes.append(n)
            all_edges.extend(book_data.get("edges", []))
        return {"nodes": all_nodes, "edges": all_edges}
    return {"nodes": [], "edges": []}


# ── ECharts Graph ──────────────────────────────────────

def make_graph_html(graph_data: dict) -> str:
    nodes = json.dumps(graph_data.get("nodes", []), ensure_ascii=False)
    edges = json.dumps(graph_data.get("edges", []), ensure_ascii=False)
    # Pre-compute unique textbooks for the legend
    textbooks = list({n.get("textbook", n.get("textbook_id", "?")) for n in graph_data.get("nodes", [])})

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
</head><body>
<div id="kgChart" style="width:100%;height:520px;background:#f8fafc;border-radius:12px"></div>
<script>
const nodes = {nodes};
const edges = {edges};

const palette = ['#8b5cf6','#3b82f6','#10b981','#f59e0b','#ef4444','#ec4899','#06b6d4'];
const textbookColors = {{}};
let ci = 0;
nodes.forEach(n => {{
    const tb = n.textbook || n.textbook_id || '?';
    if(!textbookColors[tb]) textbookColors[tb] = palette[ci++ % palette.length];
}});

const sizeMap = {{'关键': 28, '重要': 20, '补充': 14}};

const option = {{
    backgroundColor: '#f8fafc',
    tooltip: {{
        formatter: p => p.dataType === 'node'
            ? '<b style="color:' + (textbookColors[p.data.textbook||p.data.textbook_id]||'#fff') + '">' + p.data.name + '</b><br/>' + (p.data.definition||'').slice(0,150) + '<br/><i>重要性: ' + (p.data.importance||'?') + ' | ' + (p.data.textbook||p.data.textbook_id||'') + '</i>'
            : (p.data.relation_type||'') + ': ' + (p.data.description||'')
    }},
    series: [{{
        type: 'graph', layout: 'force', roam: true, draggable: true,
        force: {{ repulsion: 350, edgeLength: [80, 260], gravity: 0.06 }},
        data: nodes.map(n => ({{
            id: n.id, name: n.name,
            symbolSize: sizeMap[n.importance||'重要'] || 18,
            itemStyle: {{ color: textbookColors[n.textbook||n.textbook_id||'?'] }},
            definition: n.definition||'', importance: n.importance||'重要',
            textbook: n.textbook||n.textbook_id||'?', chapter: n.chapter||'', page: n.page||''
        }})),
        edges: edges.map(e => ({{
            source: e.source, target: e.target,
            lineStyle: {{ color: '#475569', width: 1, curveness: 0.15 }},
            relation_type: e.relation_type||'',
            description: e.description||''
        }})),
        emphasis: {{ focus: 'adjacency', lineStyle: {{ width: 2.5 }} }},
        label: {{ show: true, fontSize: 11, color: '#1e293b', fontWeight: 500, formatter: p => p.data.name.length>8 ? p.data.name.slice(0,7)+'…' : p.data.name }}
    }}]
}};

const chart = echarts.init(document.getElementById('kgChart'));
chart.setOption(option);
chart.on('click', function(params) {{
    if (params.dataType === 'node' && params.data.name) {{
        const el = parent.document.querySelector('#node-click-box textarea');
        if (el) {{
            const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
            setter.call(el, params.data.name);
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }}
}});
window.addEventListener('resize', () => chart.resize());
</script></body></html>"""


# ── RAG ────────────────────────────────────────────────

async def rag_query(question: str) -> tuple[str, str]:
    """Execute RAG query with [教材名, 页码] citation enforcement."""
    if not question.strip():
        return "", ""

    from agents.rag import query as rag_query_fn, build_index
    from models.schemas import RAGQueryIn

    try:
        result = await rag_query_fn(RAGQueryIn(question=question))

        # Format citations with mandatory format
        cite_lines = ""
        for c in result.citations:
            cite_lines += f"- **[{c.textbook}, {c.chapter}, 第{c.page}页]** (相关度: {c.relevance_score:.2f})\n"

        return result.answer, cite_lines
    except Exception as e:
        return f"❌ {e}", ""


# ── Startup: auto-rebuild FAISS index ──────────────────

def init_rag_index():
    """Rebuild FAISS index from saved textbook metadata."""
    from models.schemas import TextbookOut
    from agents.rag import build_index as rag_build_index

    meta = _load_json("textbook_metadata.json")
    if not meta:
        return "⚠️ 未找到已解析的教材数据，请先运行 run_pipeline.py"

    textbooks = [TextbookOut(**m) for m in meta]
    if not textbooks:
        return "⚠️ 教材数据为空"

    import asyncio
    loop = asyncio.new_event_loop()
    status = loop.run_until_complete(rag_build_index(textbooks))
    loop.close()
    return f"✅ 索引就绪：{status.indexed_books} 本教材, {status.total_chunks} 个分块"


# ── UI Layout ──────────────────────────────────────────

HEADER = """
<div style="text-align:center;padding:20px 0 10px">
  <h1 style="font-size:1.7rem;font-weight:700;color:#f1f5f9;margin:0">🧠 学科知识整合智能体</h1>
  <p style="color:#64748b;font-size:0.85rem;margin:6px 0 0">
    5-Agent 协作 · 跨教材去重 ≤ 30% · RAG 引用溯源
  </p>
</div>
"""

css = """
.gradio-container { max-width: 1500px !important; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="学科知识整合智能体", theme=gr.themes.Soft()) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        # ── LEFT: Dashboard ──────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### 📊 整合仪表盘")
            dashboard = gr.HTML(load_dashboard())

            btn_refresh = gr.Button("🔄 刷新数据", size="sm")
            btn_refresh.click(fn=load_dashboard, outputs=[dashboard])

        # ── CENTER: Graph ────────────────────────────
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("### 🗺️ 知识图谱")
            graph_data = load_graph_data()
            graph_html = make_graph_html(graph_data)
            graph_display = gr.HTML(graph_html)

            # Hidden textbox for node-click → query linkage
            gr.Textbox(label="", visible=False, elem_id="node-click-box")

        # ── RIGHT: RAG ───────────────────────────────
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 💬 RAG 智能问答")
            rag_input = gr.Textbox(
                label="输入问题",
                placeholder="例如：请解释「动作电位」的概念及其产生机制",
                lines=3,
            )
            btn_ask = gr.Button("🔎 查询", variant="primary", size="sm")
            rag_output = gr.Textbox(label="回答", lines=10, interactive=False)
            rag_cites = gr.Markdown("")

    # ── Events ──────────────────────────────────────

    btn_ask.click(
        fn=rag_query, inputs=[rag_input],
        outputs=[rag_output, rag_cites],
    )

    # Node-click → auto-fill + auto-query
    def on_node_click(node_name: str):
        if not node_name.strip():
            return "", "", ""
        question = f"请详细解释「{node_name}」的概念、定义和作用，并标注来源。"
        return question, "", ""  # trigger rag query separately

    # We use rag_input.change to detect node-click fill
    rag_input.change(
        fn=lambda q: rag_query(q),
        inputs=[rag_input], outputs=[rag_output, rag_cites],
    )

if __name__ == "__main__":
    print("🚀 启动 Gradio 界面...")
    print(f"📂 数据目录: {DATA_DIR}")
    print("📇 重建 FAISS 索引...")
    index_msg = init_rag_index()
    print(f"   {index_msg}")
    print(f"🌐 访问 http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
