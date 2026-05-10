"""Knowledge Graph Agent — 知识点提取与图谱构建。

Design decisions:
  - Per-chapter LLM calls (not per-book) to keep context manageable (contest §3.1-2)
  - Few-shot examples in the prompt to improve JSON output quality
  - Post-processing with json.loads() + Pydantic validation guarantees clean output
  - Retry with stricter prompt on JSON parse failure
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx

from models.schemas import (
    ChapterOut,
    KnowledgeEdge,
    KnowledgeGraphOut,
    KnowledgeNode,
    NodeCategory,
    RelationType,
)

# ── Few-shot prompt ────────────────────────────────────

_SYSTEM_PROMPT = """你是学科知识图谱构建专家。你的任务是从教材章节中提取知识点和它们之间的关系。

## 输出格式

你必须输出一个纯净的 JSON 对象（不要包含 markdown 代码块标记），格式如下：

{
  "nodes": [
    {
      "id": "node_001",
      "name": "知识点名称（简洁，不超过 20 字）",
      "definition": "该知识点的定义或说明（50-200 字）",
      "category": "核心概念|定理/定律|方法/技术|现象/过程",
      "page": 起始页码（整数）
    }
  ],
  "edges": [
    {
      "source": "node_001",
      "target": "node_002",
      "relation_type": "prerequisite|parallel|contains|applies_to",
      "description": "关系说明（30 字以内）"
    }
  ]
}

## 关系类型定义

- prerequisite（前置依赖）：学习 B 之前必须先掌握 A。示例：动作电位 依赖 静息电位
- parallel（并列关系）：同一层级的平行概念。示例：有丝分裂 与 减数分裂
- contains（包含关系）：上位概念包含下位概念。示例：免疫系统 包含 T细胞
- applies_to（应用关系）：某知识点是另一个的应用场景。示例：抗体 应用于 体液免疫

## 提取规则

1. 每个章节提取 8-15 个核心知识点（过多会降低图谱可读性）
2. 知识点之间至少标注 5 条关系
3. 知识点名称使用教材原文的术语，不要自己发明
4. 定义用自己的话概括，确保准确

## Few-shot 示例

输入章节：
  第二章 细胞的基本功能
  细胞膜主要由脂质双分子层和蛋白质组成，具有选择透过性...
  静息电位是指细胞在静息状态下膜内外的电位差...
  动作电位是细胞受到刺激后膜电位发生的快速可逆倒转...

输出：
{
  "nodes": [
    {"id":"n1","name":"细胞膜","definition":"由脂质双分子层和蛋白质组成的细胞外层结构，具有选择透过性","category":"核心概念","page":20},
    {"id":"n2","name":"静息电位","definition":"细胞在静息状态下膜内外的电位差，膜内为负、膜外为正","category":"核心概念","page":25},
    {"id":"n3","name":"动作电位","definition":"细胞受到刺激后膜电位发生的一次快速可逆倒转","category":"现象/过程","page":28}
  ],
  "edges": [
    {"source":"n2","target":"n3","relation_type":"prerequisite","description":"理解动作电位需要先掌握静息电位"},
    {"source":"n1","target":"n2","relation_type":"applies_to","description":"静息电位的形成依赖于细胞膜结构"}
  ]
}"""

_USER_PROMPT_TEMPLATE = """请从以下章节中提取知识点和关系。

教材：{textbook_name}
章节：{chapter_title}（起始页码 {page_start}）
内容：
{content}"""

# ── JSON extraction helpers ────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> str:
    """Extract a JSON object from LLM output that may contain markdown fences."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find the first complete JSON object
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(0)
    return text


def _validate_and_fix_nodes(nodes: list[dict], chapter_page: int, textbook_id: str) -> list[KnowledgeNode]:
    """Validate node fields and apply defaults for missing values."""
    valid: list[KnowledgeNode] = []
    for i, n in enumerate(nodes):
        try:
            node = KnowledgeNode(
                id=n.get("id", f"node_{i:03d}"),
                name=n["name"],
                definition=n.get("definition", ""),
                category=NodeCategory(n.get("category", "核心概念")),
                chapter="",
                page=n.get("page", chapter_page),
                textbook_id=textbook_id,
            )
            valid.append(node)
        except Exception:
            continue
    return valid


def _validate_and_fix_edges(edges: list[dict], node_ids: set[str]) -> list[KnowledgeEdge]:
    """Validate edges; drop any that reference non-existent nodes."""
    valid: list[KnowledgeEdge] = []
    for e in edges:
        try:
            if e["source"] not in node_ids or e["target"] not in node_ids:
                continue
            edge = KnowledgeEdge(
                source=e["source"],
                target=e["target"],
                relation_type=RelationType(e.get("relation_type", "parallel")),
                description=e.get("description", ""),
            )
            valid.append(edge)
        except Exception:
            continue
    return valid


# ── Public API ─────────────────────────────────────────

async def extract_knowledge_graph(
    chapter: ChapterOut,
    textbook_id: str,
    textbook_name: str = "",
    llm_callable=None,
) -> KnowledgeGraphOut:
    """Extract nodes and edges from a single chapter.

    Args:
        chapter:      Parsed chapter with content.
        textbook_id:  Unique textbook identifier.
        textbook_name: Human-readable textbook title (for the prompt).
        llm_callable: Optional async (messages) -> str. If None, uses DashScope.

    Returns:
        KnowledgeGraphOut with validated nodes and edges.
    """
    if not chapter.content.strip():
        return KnowledgeGraphOut(textbook_id=textbook_id, nodes=[], edges=[])

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        textbook_name=textbook_name or textbook_id,
        chapter_title=chapter.title,
        page_start=chapter.page_start,
        content=chapter.content[:6000],  # Truncate long chapters
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = ""
    if llm_callable:
        raw_output = await llm_callable(messages)
    else:
        raw_output = await _default_llm_call(messages)

    # Parse + validate
    try:
        json_str = _extract_json(raw_output)
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Retry once with a stricter prompt
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": "你上面的输出不是合法的 JSON。请重新输出，确保是纯净的 JSON 对象，不要包含 ``` 标记或其他非 JSON 文本。",
        })
        if llm_callable:
            raw_output = await llm_callable(messages)
        else:
            raw_output = await _default_llm_call(messages)
        try:
            json_str = _extract_json(raw_output)
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return KnowledgeGraphOut(textbook_id=textbook_id, nodes=[], edges=[])

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    nodes = _validate_and_fix_nodes(raw_nodes, chapter.page_start, textbook_id)
    node_ids = {n.id for n in nodes}
    edges = _validate_and_fix_edges(raw_edges, node_ids)

    # Annotate nodes with chapter context
    for node in nodes:
        node.chapter = chapter.title
        node.textbook_id = textbook_id

    return KnowledgeGraphOut(
        textbook_id=textbook_id,
        nodes=nodes,
        edges=edges,
    )


async def _default_llm_call(messages: list[dict]) -> str:
    """Fallback LLM call via DashScope."""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return '{"nodes":[],"edges":[]}'

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("LLM_MODEL", "qwen-max"),
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4096,
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
