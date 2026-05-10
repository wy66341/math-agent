"""Knowledge Extractor Agent — LLM 驱动的知识点提取与关系识别。

Design:
  - Batch processing: large chapters are split into ~4000-char batches to
    prevent LLM context overflow (§ efficiency).
  - Few-shot prompt with structured JSON output.
  - Post-extraction dedup: same-name knowledge points within a textbook are
    merged before being written to the final result.
  - Output fields (name, definition, importance_level) are designed so the
    Integration Agent can perform cross-textbook semantic alignment directly
    on the 'name' + 'definition' fields.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from typing import Optional, Callable

import httpx

from models.schemas import (
    ChapterOut,
    ExtractRelationType,
    ExtractedKnowledgePoint,
    ExtractedRelation,
    ExtractionResult,
    ImportanceLevel,
    TextbookOut,
)

# ── Batch config ───────────────────────────────────────

CHARS_PER_BATCH = 4000        # safe under 8K tokens (Chinese ~1.5 chars/token)
OVERLAP_CHARS = 200           # overlap between adjacent batches to catch cross-boundary concepts


# ── Few-shot prompt ────────────────────────────────────

SYSTEM_PROMPT = """你是学科知识建模专家。你的任务是从教材文本中提取结构化知识点及其关系。

## 输出格式（必须是纯净 JSON）

{
  "nodes": [
    {
      "local_id": "n1",
      "name": "知识点名称（简洁术语，不超过 20 字）",
      "definition": "准确的定义或说明（80-200 字）",
      "importance_level": "关键|重要|补充",
      "page": 起始页码
    }
  ],
  "edges": [
    {
      "source": "n1",
      "target": "n2",
      "relation_type": "prerequisite|parallel|contains",
      "description": "关系说明（30 字以内）"
    }
  ]
}

## 字段说明

- importance_level:
  - "关键"：学科基石概念，是大量后续知识的前置依赖
  - "重要"：核心主干知识点
  - "补充"：拓展性、背景性知识

- relation_type:
  - "prerequisite"（前置依赖）：B 以 A 为基础
    例：动作电位 prerequisite 静息电位
  - "parallel"（并列关系）：同一层级平行概念
    例：有丝分裂 parallel 减数分裂
  - "contains"（包含关系）：上级概念包含下级
    例：免疫系统 contains T细胞
  - "applies_to"（应用关系）：A 是 B 的应用场景或临床关联
    例：抗体 applies_to 体液免疫

## 提取规则

1. 每个批次提取 5-12 个知识点（取决于文本信息密度）
2. 知识点名称必须使用教材中的标准术语
3. 定义须用自己的话概括，确保准确
4. 宁可少提，不编造

## Few-shot 示例

输入文本（页码 28-32）：
  细胞膜主要由脂质双分子层和镶嵌蛋白质组成。膜上的离子通道是控制离子
  进出细胞的关键结构。静息电位是指细胞在静息状态下膜内外的电位差，
  通常膜内为负、膜外为正。钠钾泵（Na+/K+-ATP酶）通过主动转运维持
  细胞内外钠钾离子的浓度梯度。当细胞受到足够强度的刺激时，膜上的电压
  门控钠通道开放，钠离子大量内流，导致膜电位快速去极化乃至反极化，
  这一过程称为动作电位。

输出：
{
  "nodes": [
    {"local_id":"n1","name":"细胞膜","definition":"由脂质双分子层和镶嵌蛋白质构成的细胞外层结构，是物质交换和信号传递的界面","importance_level":"关键","page":28},
    {"local_id":"n2","name":"离子通道","definition":"细胞膜上控制特定离子顺电化学梯度跨膜流动的跨膜蛋白质","importance_level":"重要","page":29},
    {"local_id":"n3","name":"静息电位","definition":"细胞未受刺激时膜内外相对稳定的电位差，膜内为负、膜外为正","importance_level":"关键","page":30},
    {"local_id":"n4","name":"钠钾泵","definition":"利用ATP水解能量将3个钠离子泵出、2个钾离子泵入细胞的跨膜蛋白质，维持离子浓度梯度","importance_level":"重要","page":31},
    {"local_id":"n5","name":"动作电位","definition":"细胞受刺激后膜电位发生的快速去极化-反极化-复极化过程，是兴奋性的标志","importance_level":"关键","page":32}
  ],
  "edges": [
    {"source":"n1","target":"n2","relation_type":"contains","description":"离子通道是细胞膜的组成结构"},
    {"source":"n4","target":"n3","relation_type":"prerequisite","description":"钠钾泵维持的离子梯度是静息电位的基础"},
    {"source":"n3","target":"n5","relation_type":"prerequisite","description":"静息电位是理解动作电位的前提"},
    {"source":"n2","target":"n5","relation_type":"prerequisite","description":"钠通道开放是动作电位去极化的直接原因"},
    {"source":"n4","target":"n5","relation_type":"applies_to","description":"钠钾泵的功能异常直接影响动作电位的产生"}
  ]
}"""


# ── Batch splitter ─────────────────────────────────────

def _split_into_batches(text: str, size: int = CHARS_PER_BATCH, overlap: int = OVERLAP_CHARS) -> list[tuple[str, int]]:
    """Split text into overlapping batches.

    Returns list of (batch_text, char_offset_in_original).
    The char_offset helps map page numbers correctly.
    """
    if len(text) <= size:
        return [(text, 0)]

    batches: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # Try to break at a sentence boundary within the last 15%
        if end < len(text):
            search_start = max(start + int(size * 0.85), start)
            for sep in ("。", "！", "？", "\n\n", "\n", ".", "!", "?"):
                pos = text.rfind(sep, search_start, end)
                if pos > search_start:
                    end = pos + 1
                    break

        batch = text[start:end].strip()
        if batch:
            batches.append((batch, start))
        start = end - overlap if end < len(text) else len(text)

    return batches


# ── JSON extraction ────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(raw: str) -> str:
    """Strip markdown fences and extract the first JSON object."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_RE.search(raw)
    return m.group(0) if m else raw


# ── Main extraction logic ──────────────────────────────

async def extract_from_textbook(
    textbook: TextbookOut,
    llm_callable: Optional[Callable] = None,
    on_batch_complete: Optional[Callable[[int, int, int], None]] = None,
    max_batches: int = 0,
) -> ExtractionResult:
    """Extract knowledge points from an entire textbook using batched LLM calls.

    Args:
        textbook:         Parsed textbook with chapters and content.
        llm_callable:     Async function (messages: list[dict]) -> str.
                           Defaults to DashScope qwen-max.
        on_batch_complete: Optional callback(batch_index, total_batches, tokens_used)
                           for progress reporting.

    Returns:
        ExtractionResult with deduplicated nodes and edges.
    """
    all_batch_prompts: list[tuple[str, str, int]] = []
    # (batch_text, chapter_title, page_offset)

    for chapter in textbook.chapters:
        if not chapter.content.strip():
            continue
        batches = _split_into_batches(chapter.content)
        for batch_text, char_offset in batches:
            # Approximate page within the chapter
            page = chapter.page_start + int(
                (char_offset / max(len(chapter.content), 1)) * (chapter.page_end - chapter.page_start + 1)
            )
            page = max(chapter.page_start, min(page, chapter.page_end))
            all_batch_prompts.append((batch_text, chapter.title, page))

    if not all_batch_prompts:
        return ExtractionResult(textbook_id=textbook.textbook_id, nodes=[], edges=[], batch_count=0)

    if max_batches > 0:
        all_batch_prompts = all_batch_prompts[:max_batches]

    # ── Process all batches ─────────────────────────────
    all_nodes: list[ExtractedKnowledgePoint] = []
    all_edges: list[ExtractedRelation] = []
    local_id_counter = 0
    total_tokens = 0

    for batch_idx, (batch_text, chapter_title, page) in enumerate(all_batch_prompts):
        user_prompt = f"教材：{textbook.title}\n章节：{chapter_title}\n起始页码：{page}\n文本：\n{batch_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Retry logic with exponential backoff (max 5 retries)
        raw = None
        last_err = None
        for attempt in range(5):
            try:
                raw = await _call_llm(messages, llm_callable)
                break
            except (httpx.RemoteProtocolError, httpx.ConnectError,
                    httpx.ReadTimeout, ConnectionError, OSError) as e:
                last_err = e
                wait = 2 ** attempt + random.uniform(0, 1)
                if attempt < 4:
                    print(f"    ⚠️  批次 {batch_idx + 1} 重试 {attempt + 1}/5... ({wait:.1f}s)", flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f"    ❌ 批次 {batch_idx + 1} 5 次重试后仍失败: {e}", flush=True)

        if raw is None:
            # After 5 retries, skip this batch
            if on_batch_complete:
                await on_batch_complete(batch_idx + 1, len(all_batch_prompts), 0)
            continue

        nodes, edges, tokens = _parse_batch_output(raw, textbook.textbook_id, chapter_title, page)

        # Random delay to avoid campus network rate-limiting
        delay = random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)

        # Assign global IDs
        for node in nodes:
            local_id_counter += 1
            node.id = f"{textbook.textbook_id}_n{local_id_counter:04d}"

        # Map local IDs in edges to global IDs
        local_to_global: dict[str, str] = {}
        for node in nodes:
            local_to_global[node.id.rsplit("_", 1)[-1]] = node.id

        for edge in edges:
            src = local_to_global.get(edge.source, edge.source)
            tgt = local_to_global.get(edge.target, edge.target)
            edge.source = src
            edge.target = tgt

        all_nodes.extend(nodes)
        all_edges.extend(edges)
        total_tokens += tokens

        if on_batch_complete:
            await on_batch_complete(batch_idx + 1, len(all_batch_prompts), tokens)

    # ── Intra-textbook dedup ────────────────────────────
    all_nodes = _deduplicate_nodes(all_nodes)
    all_edges = _deduplicate_edges(all_edges)

    return ExtractionResult(
        textbook_id=textbook.textbook_id,
        nodes=all_nodes,
        edges=all_edges,
        batch_count=len(all_batch_prompts),
        total_cost_tokens=total_tokens,
    )


# ── Batch output parser ────────────────────────────────

def _parse_batch_output(
    raw: str,
    textbook_id: str,
    chapter_title: str,
    page: int,
) -> tuple[list[ExtractedKnowledgePoint], list[ExtractedRelation], int]:
    """Parse LLM output into validated nodes and edges.

    Returns (nodes, edges, estimated_tokens).
    """
    nodes: list[ExtractedKnowledgePoint] = []
    edges: list[ExtractedRelation] = []
    estimated_tokens = len(raw) // 2  # rough estimate

    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError:
        return nodes, edges, estimated_tokens

    for item in data.get("nodes", []):
        try:
            nodes.append(ExtractedKnowledgePoint(
                id="",  # assigned later
                name=item.get("name", "")[:20],
                definition=item.get("definition", "")[:300],
                importance_level=ImportanceLevel(item.get("importance_level", "重要")),
                textbook_id=textbook_id,
                chapter=chapter_title,
                page=item.get("page", page),
            ))
        except Exception:
            continue

    node_local_ids = {item.get("local_id", "") for item in data.get("nodes", [])}

    for item in data.get("edges", []):
        try:
            if item["source"] not in node_local_ids or item["target"] not in node_local_ids:
                continue
            edges.append(ExtractedRelation(
                source=item["source"],
                target=item["target"],
                relation_type=ExtractRelationType(item.get("relation_type", "parallel")),
                description=item.get("description", "")[:50],
            ))
        except Exception:
            continue

    return nodes, edges, estimated_tokens


# ── Dedup ──────────────────────────────────────────────

def _deduplicate_nodes(nodes: list[ExtractedKnowledgePoint]) -> list[ExtractedKnowledgePoint]:
    """Merge nodes with identical names within the same textbook.

    Strategy: keep the one with the longest definition, merge page ranges.
    """
    seen: dict[str, ExtractedKnowledgePoint] = {}
    for node in nodes:
        key = node.name.strip()
        if key in seen:
            existing = seen[key]
            if len(node.definition) > len(existing.definition):
                existing.definition = node.definition
            if node.importance_level == ImportanceLevel.critical:
                existing.importance_level = ImportanceLevel.critical
            existing.page = min(existing.page, node.page)
        else:
            seen[key] = node
    return list(seen.values())


def _deduplicate_edges(edges: list[ExtractedRelation]) -> list[ExtractedRelation]:
    """Remove duplicate edges (same source, target, type)."""
    seen = set()
    unique: list[ExtractedRelation] = []
    for e in edges:
        key = (e.source, e.target, e.relation_type.value)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ── LLM call ───────────────────────────────────────────

async def _call_llm(messages: list[dict], callable_fn: Optional[Callable] = None) -> str:
    """Call LLM, either via a provided callable or DashScope."""
    if callable_fn:
        return await callable_fn(messages)

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return '{"nodes":[], "edges":[]}'

    async with httpx.AsyncClient(timeout=180) as client:
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
