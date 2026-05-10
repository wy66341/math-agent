"""Unified I/O schemas for all 5 Agents."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Parser Agent ──────────────────────────────────────

class ChapterOut(BaseModel):
    chapter_id: str
    title: str
    page_start: int
    page_end: int
    content: str
    char_count: int

class TextbookOut(BaseModel):
    textbook_id: str
    filename: str
    title: str
    total_pages: int
    total_chars: int
    chapters: list[ChapterOut]


# ── Knowledge Extractor ───────────────────────────────

class ImportanceLevel(str, Enum):
    critical = "关键"        # 学科基石，后续知识的前置依赖
    important = "重要"       # 核心主干知识
    supplementary = "补充"   # 拓展/背景知识

class ExtractedKnowledgePoint(BaseModel):
    """A single knowledge point as extracted by the Knowledge Extractor Agent."""
    id: str                                     # textbook-scoped unique ID
    name: str                                   # 知识点名称（≤ 20 字）
    definition: str                             # 定义/说明（50-200 字）
    importance_level: ImportanceLevel           # 重要性分级
    textbook_id: str                            # 来源教材
    chapter: str                                # 来源章节
    page: int                                   # 起始页码
    embedding: Optional[list[float]] = None     # 语义向量（跨教材去重用）

class ExtractRelationType(str, Enum):
    prerequisite = "prerequisite"    # 前置依赖
    parallel = "parallel"            # 并列关系
    contains = "contains"            # 包含关系
    applies_to = "applies_to"        # 应用关系

class ExtractedRelation(BaseModel):
    source: str                     # 源知识点 ID
    target: str                     # 目标知识点 ID
    relation_type: ExtractRelationType
    description: str                # 关系说明

class ExtractionResult(BaseModel):
    """Complete extraction result for one textbook."""
    textbook_id: str
    nodes: list[ExtractedKnowledgePoint]
    edges: list[ExtractedRelation]
    batch_count: int                # 共分了多少批次
    total_cost_tokens: int = 0      # LLM Token 消耗统计


# ── Knowledge Graph Agent (legacy) ────────────────────

class NodeCategory(str, Enum):
    core_concept = "核心概念"
    theorem = "定理/定律"
    method = "方法/技术"
    phenomenon = "现象/过程"

class RelationType(str, Enum):
    prerequisite = "prerequisite"
    parallel = "parallel"
    contains = "contains"
    applies_to = "applies_to"

class KnowledgeNode(BaseModel):
    id: str
    name: str
    definition: str
    category: NodeCategory
    chapter: str
    page: int
    textbook_id: str

class KnowledgeEdge(BaseModel):
    source: str
    target: str
    relation_type: RelationType
    description: str

class KnowledgeGraphOut(BaseModel):
    textbook_id: str
    nodes: list[KnowledgeNode]
    edges: list[KnowledgeEdge]


# ── Integration Agent ─────────────────────────────────

class DecisionAction(str, Enum):
    merge = "merge"      # 多源合并为一个知识点
    keep = "keep"        # 保留唯一来源的知识点
    remove = "remove"    # 删除冗余/碎片知识点

class IntegrationDecision(BaseModel):
    decision_id: str
    action: DecisionAction
    affected_nodes: list[str]          # 涉及的知识点 ID 列表
    affected_names: list[str] = []     # 知识点名称（可读性）
    result_node: Optional[str] = None  # merge 后的节点 ID
    reason: str                        # 决策理由
    confidence: float = Field(ge=0.0, le=1.0)

class IntegrationStats(BaseModel):
    textbook_count: int
    original_nodes: int                 # 整合前总节点数
    merged_nodes: int                   # 整合后节点数
    original_chars: int                 # 整合前总字数
    merged_chars: int                   # 整合后字数
    compression_ratio: float            # 压缩比
    total_decisions: int
    merge_count: int
    keep_count: int
    remove_count: int

class IntegrationOut(BaseModel):
    decisions: list[IntegrationDecision]
    stats: IntegrationStats
    report_markdown: str                # 整合报告 Markdown
    merged_knowledge_graph: dict = {}   # 整合后的图谱数据


# ── RAG Agent ─────────────────────────────────────────

class Citation(BaseModel):
    textbook: str
    chapter: str
    page: int
    relevance_score: float

class RAGQueryIn(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)

class RAGAnswerOut(BaseModel):
    answer: str
    citations: list[Citation]
    source_chunks: list[str]

class RAGStatusOut(BaseModel):
    indexed_books: int
    total_chunks: int
    embedding_model: str


# ── Dialogue Agent ────────────────────────────────────

class DialogueMessage(BaseModel):
    role: str = Field(description="user | assistant")
    content: str

class DialogueIn(BaseModel):
    message: str
    history: list[DialogueMessage] = Field(default_factory=list)

class DialogueOut(BaseModel):
    reply: str
    updated_decisions: Optional[list[IntegrationDecision]] = None
    affected_decision_ids: list[str] = Field(default_factory=list)
