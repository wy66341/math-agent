"""API Route definitions matching contest requirements."""
from fastapi import APIRouter, UploadFile, File
from typing import Optional

router = APIRouter(prefix="/api")


# ── P0: 文件上传 ──────────────────────────────────────

@router.post("/upload")
async def upload_textbooks(files: list[UploadFile] = File(...)):
    """上传教材文件（支持 PDF/MD/TXT/DOCX），返回解析状态列表。"""
    pass

@router.get("/textbooks")
async def list_textbooks():
    """返回已上传教材的列表及解析状态。"""
    pass


# ── P0: 知识图谱 ──────────────────────────────────────

@router.post("/kg/build")
async def build_knowledge_graph(book_id: Optional[str] = None):
    """为指定教材（或全部）构建知识图谱。"""
    pass

@router.get("/kg/graph")
async def get_knowledge_graph(book_id: Optional[str] = None):
    """获取知识图谱数据（nodes + edges），供前端 Cytoscape.js 渲染。"""
    pass


# ── P0: 跨教材整合 ───────────────────────────────────

@router.post("/integrate")
async def integrate_knowledge():
    """执行跨教材知识整合，返回整合决策列表 + 压缩统计。"""
    pass

@router.get("/integrate/decisions")
async def get_integration_decisions():
    """获取当前所有整合决策。"""
    pass


# ── P0: RAG 问答 ─────────────────────────────────────

@router.post("/rag/index")
async def build_rag_index():
    """对已上传教材建立向量索引（分块 → Embedding → 存 ChromaDB）。"""
    pass

@router.post("/rag/query")
async def rag_query(question: str, top_k: int = 5):
    """输入问题，返回带引用来源的回答。"""
    pass

@router.get("/rag/status")
async def rag_status():
    """查询索引状态（已索引 X 本教材，共 Y 个 chunk）。"""
    pass


# ── P0: 对话 ──────────────────────────────────────────

@router.post("/dialogue")
async def dialogue(message: str, history: Optional[list] = None):
    """多轮对话，修改整合决策。"""
    pass


# ── P1: 整合报告 ──────────────────────────────────────

@router.get("/report")
async def generate_report():
    """生成整合报告（返回 Markdown）。"""
    pass
