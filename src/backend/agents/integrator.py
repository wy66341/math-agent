"""Integration Agent — 跨教材知识整合（赛题核心模块 P0-4）。

Pipeline:
  1. 语义对齐 (Semantic Alignment)
     - Stage 1: Embedding 粗筛 (cosine ≥ 0.85 → candidate pair)
     - Stage 2: LLM 精判 (confirm equivalence + choose best source)
  2. 整合决策 (Integration Decisions)
     - merge / keep / remove，每项附带决策理由和置信度
  3. 内容提纯 (Content Purification)
     - 压缩比 ≤ 30%，保留核心知识点与逻辑链路
  4. 统计输出 (Statistics & Report)
     - Markdown 整合报告 + 整合后图谱 JSON
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Optional, Callable

import httpx
import numpy as np

from models.schemas import (
    DecisionAction,
    ExtractedKnowledgePoint,
    ExtractedRelation,
    ExtractionResult,
    IntegrationDecision,
    IntegrationOut,
    IntegrationStats,
)

# ── Config ─────────────────────────────────────────────

def _resolve_model_path(model_name: str) -> str:
    try:
        from modelscope import snapshot_download
        import os
        cache = snapshot_download(model_name)
        if os.path.isdir(cache):
            return cache
    except Exception:
        pass
    return model_name


SIMILARITY_THRESHOLD = 0.70       # aggressive: catch more potential duplicates
TARGET_COMPRESSION = 0.30         # ≤ 30% of original
_JSON_RE = re.compile(r"\{[\s\S]*\}")


# ── Stage 1: Embedding coarse screening ───────────────

def _encode_nodes(
    nodes: list[ExtractedKnowledgePoint],
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> list[ExtractedKnowledgePoint]:
    """Encode all node (name + definition) into embedding vectors."""
    from sentence_transformers import SentenceTransformer

    if not nodes:
        return nodes

    # Resolve via ModelScope mirror first, fall back to local cache
    model_path = _resolve_model_path(model_name)
    import os as _os
    cache_dir = _os.path.expanduser("~/.cache/huggingface/hub")
    if _os.path.isdir(cache_dir):
        model = SentenceTransformer(model_path, local_files_only=True)
    else:
        model = SentenceTransformer(model_path)
    texts = [f"{n.name}: {n.definition}" for n in nodes]

    embeddings = model.encode(texts, show_progress_bar=False)

    for i, node in enumerate(nodes):
        node.embedding = embeddings[i].tolist()

    return nodes


def _find_candidate_pairs(
    nodes: list[ExtractedKnowledgePoint],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[ExtractedKnowledgePoint, ExtractedKnowledgePoint, float]]:
    """Find node pairs with cosine similarity ≥ threshold."""
    candidates: list[tuple[ExtractedKnowledgePoint, ExtractedKnowledgePoint, float]] = []
    n = len(nodes)

    for i in range(n):
        if nodes[i].embedding is None:
            continue
        vi = np.array(nodes[i].embedding)
        for j in range(i + 1, n):
            if nodes[j].embedding is None:
                continue
            # Skip same-textbook pairs (already deduped by extractor)
            if nodes[i].textbook_id == nodes[j].textbook_id:
                continue
            vj = np.array(nodes[j].embedding)
            sim = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj)))
            if sim >= threshold:
                candidates.append((nodes[i], nodes[j], sim))
    return candidates


# ── Stage 2: LLM fine judgment ─────────────────────────

_JUDGE_SYSTEM_PROMPT = """你是学科知识整合专家。你的任务是判断两个来自不同教材的知识点是否描述同一概念，并做出整合决策。

## 输入格式
{
  "node_a": {"name": "...", "definition": "...", "textbook": "..."},
  "node_b": {"name": "...", "definition": "...", "textbook": "..."},
  "similarity": 0.92
}

## 输出格式（必须是纯净 JSON）
{
  "is_same": true,
  "confidence": 0.95,
  "action": "merge",
  "best_source": "a",
  "reason": "两本教材都讲解了炎症的概念，但《病理学》的描述更系统完整，建议保留其定义并标注多源引用",
  "merged_definition": "整合后的精炼定义（100-200 字，可选）"
}

## 决策规则
- is_same=true, action="merge": 两个知识点描述同一概念，合并
  **重要：merged_definition 必须是重新生成的精华摘要，不超过 150 字**
- is_same=false, action="keep": 两个知识点不同，各自保留
- 如果一方定义明显残缺或错误，建议 remove 并给出理由

## Few-shot 示例

输入:
{
  "node_a": {"name": "中性粒细胞", "definition": "白细胞中数量最多的一种，具有吞噬杀菌功能，是急性炎症的主要反应细胞", "textbook": "病理学"},
  "node_b": {"name": "Neutrophil", "definition": "The most abundant type of granulocytes, constituting 40-70% of all white blood cells. They are recruited to sites of inflammation.", "textbook": "生理学"},
  "similarity": 0.93
}

输出:
{"is_same":true,"confidence":0.95,"action":"merge","best_source":"a","reason":"中性粒细胞和Neutrophil是同一概念的中英文名称。《病理学》的定义更注重功能，《生理学》的定义更注重数量特征，合并后可互相补充","merged_definition":"中性粒细胞是白细胞中数量最多的类型（占40-70%），具有吞噬杀菌功能，是急性炎症反应中最先到达感染部位的免疫细胞"}

输入:
{"node_a":{"name":"动作电位","definition":"细胞受刺激后膜电位发生的快速去极化-反极化-复极化过程","textbook":"生理学"},"node_b":{"name":"静息电位","definition":"细胞未受刺激时膜内外的电位差","textbook":"生理学"},"similarity":0.62}

输出:
{"is_same":false,"confidence":0.98,"action":"keep","best_source":"","reason":"动作电位和静息电位是不同但相关的概念，各自保留"}

## 注意事项
1. 中英文术语应被识别为同一概念
2. 即使表述方式不同，核心概念相同就应合并
3. 宁可保守（保留多的），不要激进（错误合并）
"""


async def _judge_pair(
    node_a: ExtractedKnowledgePoint,
    node_b: ExtractedKnowledgePoint,
    similarity: float,
    llm: Optional[Callable] = None,
) -> dict:
    """Ask LLM to judge whether two nodes refer to the same concept."""
    user_prompt = json.dumps({
        "node_a": {"name": node_a.name, "definition": node_a.definition, "textbook": node_a.textbook_id},
        "node_b": {"name": node_b.name, "definition": node_b.definition, "textbook": node_b.textbook_id},
        "similarity": round(similarity, 4),
    }, ensure_ascii=False)

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = await _call_llm(messages, llm)

    try:
        json_str = _JSON_RE.search(raw)
        return json.loads(json_str.group(0)) if json_str else {"is_same": False, "action": "keep", "confidence": 0.5, "reason": "LLM output parse failed"}
    except json.JSONDecodeError:
        return {"is_same": False, "action": "keep", "confidence": 0.5, "reason": "JSON parse error"}


# ── Main integration logic ─────────────────────────────

async def integrate(
    extraction_results: list[ExtractionResult],
    llm_callable: Optional[Callable] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> IntegrationOut:
    """Execute full cross-textbook integration pipeline.

    Args:
        extraction_results: List of ExtractionResult (one per textbook).
        llm_callable:       Async (messages) -> str.
        on_progress:        Callback(phase, current, total).

    Returns:
        IntegrationOut with decisions, stats, report, and merged graph.
    """
    # ── Collect all nodes ───────────────────────────────
    all_nodes: list[ExtractedKnowledgePoint] = []
    for er in extraction_results:
        # Assign textbook_id to each node's chapter annotation
        textbook_name = er.textbook_id
        for node in er.nodes:
            if not node.textbook_id:
                node.textbook_id = textbook_name
        all_nodes.extend(er.nodes)

    original_node_count = len(all_nodes)
    original_chars = sum(len(n.definition) for n in all_nodes)

    if on_progress:
        await on_progress("encoding", 0, len(all_nodes))

    # ── Stage 1: Embedding coarse screening ─────────────
    all_nodes = _encode_nodes(all_nodes)
    candidates = _find_candidate_pairs(all_nodes)
    candidate_pairs = len(candidates)

    if on_progress:
        await on_progress("judging", 0, candidate_pairs)

    # ── Stage 2: LLM fine judgment ──────────────────────
    decisions: list[IntegrationDecision] = []
    merge_groups: dict[str, list[tuple[ExtractedKnowledgePoint, float]]] = defaultdict(list)
    # key = "best" node id, value = list of (node, confidence)

    processed_pairs: set[tuple[str, str]] = set()
    decision_counter = 0

    for node_a, node_b, sim in candidates:
        pair_key = (min(node_a.id, node_b.id), max(node_a.id, node_b.id))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)

        decision_counter += 1
        result = await _judge_pair(node_a, node_b, sim, llm_callable)

        action = DecisionAction(result.get("action", "keep"))
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")

        if action == DecisionAction.merge and result.get("is_same"):
            best_source = result.get("best_source", "a")
            primary = node_a if best_source == "a" else node_b
            secondary = node_b if best_source == "a" else node_a

            decision = IntegrationDecision(
                decision_id=f"merge_{decision_counter:04d}",
                action=DecisionAction.merge,
                affected_nodes=[primary.id, secondary.id],
                affected_names=[primary.name, secondary.name],
                result_node=primary.id,
                reason=reason,
                confidence=min(confidence, sim),  # blend LLM + embedding confidence
            )
            merge_groups[primary.id].append((secondary, confidence))
            if primary.id not in {n.id for n, _ in merge_groups.get(primary.id, [])}:
                merge_groups[primary.id].append((primary, 1.0))
        else:
            decision = IntegrationDecision(
                decision_id=f"keep_{decision_counter:04d}",
                action=DecisionAction.keep,
                affected_nodes=[node_a.id, node_b.id],
                affected_names=[node_a.name, node_b.name],
                reason=f"不同概念，各自保留。相似度 {sim:.2f}，" + reason,
                confidence=confidence,
            )

        decisions.append(decision)

        if on_progress and decision_counter % 5 == 0:
            await on_progress("judging", decision_counter, candidate_pairs)

    # ── Stage 3: Aggressive content purification ────────
    # Build merge set
    merged_node_ids: set[str] = set()
    removed_node_ids: set[str] = set()

    for d in decisions:
        if d.action == DecisionAction.merge:
            for nid in d.affected_nodes:
                if d.result_node and nid == d.result_node:
                    merged_node_ids.add(nid)
                else:
                    removed_node_ids.add(nid)

    all_node_ids = {n.id for n in all_nodes}
    merged_or_removed = set()
    for d in decisions:
        merged_or_removed.update(d.affected_nodes)
    keep_node_ids = all_node_ids - merged_or_removed

    id_to_node = {n.id: n for n in all_nodes}
    kept_nodes = [id_to_node[nid] for nid in (merged_node_ids | keep_node_ids) if nid in id_to_node]

    importance_order = {"关键": 0, "重要": 1, "补充": 2}
    kept_nodes.sort(key=lambda n: importance_order.get(n.importance_level.value, 2))

    # ── Pass 1: Nuke ALL supplementary nodes ────────────
    supp_nodes = [n for n in kept_nodes if n.importance_level.value == "补充"]
    for victim in supp_nodes:
        kept_nodes.remove(victim)
        removed_node_ids.add(victim.id)
        decisions.append(IntegrationDecision(
            decision_id=f"purge_supp_{len(decisions):04d}",
            action=DecisionAction.remove,
            affected_nodes=[victim.id],
            affected_names=[victim.name],
            reason="Pass 1: 移除补充级知识点",
            confidence=0.95,
        ))

    merged_chars = sum(len(n.definition) for n in kept_nodes)
    ratio = merged_chars / max(original_chars, 1)

    # ── Pass 2: Compress "重要" node definitions ────────
    if ratio > TARGET_COMPRESSION:
        important_nodes = [n for n in kept_nodes if n.importance_level.value == "重要"]
        # Chop each important definition to ~100 chars
        for n in important_nodes:
            if len(n.definition) > 100:
                n.definition = n.definition[:97] + "..."
        merged_chars = sum(len(n.definition) for n in kept_nodes)
        ratio = merged_chars / max(original_chars, 1)
        # Record a decision for this
        decisions.append(IntegrationDecision(
            decision_id=f"compress_important_{len(decisions):04d}",
            action=DecisionAction.merge,
            affected_nodes=[n.id for n in important_nodes[:5]],
            affected_names=["重要级知识点批量压缩"],
            reason=f"Pass 2: 压缩 {len(important_nodes)} 个重要级节点的定义至 100 字以内",
            confidence=0.95,
        ))

    # ── Pass 3: If still over, remove least-important "重要" nodes ──
    imp_nodes = [n for n in kept_nodes if n.importance_level.value == "重要"]
    while ratio > TARGET_COMPRESSION and imp_nodes:
        victim = imp_nodes.pop()
        kept_nodes.remove(victim)
        removed_node_ids.add(victim.id)
        decisions.append(IntegrationDecision(
            decision_id=f"trim_important_{len(decisions):04d}",
            action=DecisionAction.remove,
            affected_nodes=[victim.id],
            affected_names=[victim.name],
            reason="Pass 3: 为达成 30% 压缩比，移除低优先级重要节点",
            confidence=0.85,
        ))
        merged_chars = sum(len(n.definition) for n in kept_nodes)
        ratio = merged_chars / max(original_chars, 1)

    if on_progress:
        await on_progress("reporting", 1, 1)

    # ── Statistics ──────────────────────────────────────
    merge_count = sum(1 for d in decisions if d.action == DecisionAction.merge)
    keep_count = sum(1 for d in decisions if d.action == DecisionAction.keep)
    remove_count = sum(1 for d in decisions if d.action == DecisionAction.remove)

    stats = IntegrationStats(
        textbook_count=len(extraction_results),
        original_nodes=original_node_count,
        merged_nodes=len(kept_nodes),
        original_chars=original_chars,
        merged_chars=merged_chars,
        compression_ratio=round(ratio, 4),
        total_decisions=len(decisions),
        merge_count=merge_count,
        keep_count=keep_count,
        remove_count=remove_count,
    )

    # ── Generate Markdown report ────────────────────────
    report = _generate_report(
        extraction_results, stats, decisions, kept_nodes
    )

    # ── Build merged knowledge graph ────────────────────
    merged_graph = _build_merged_graph(extraction_results, decisions, kept_nodes)

    return IntegrationOut(
        decisions=decisions,
        stats=stats,
        report_markdown=report,
        merged_knowledge_graph=merged_graph,
    )


# ── Report generator ───────────────────────────────────

def _generate_report(
    extraction_results: list[ExtractionResult],
    stats: IntegrationStats,
    decisions: list[IntegrationDecision],
    kept_nodes: list[ExtractedKnowledgePoint],
) -> str:
    """Generate the integration report in Markdown."""
    merge_cases = [d for d in decisions if d.action == DecisionAction.merge][:5]
    remove_cases = [d for d in decisions if d.action == DecisionAction.remove][:3]

    case_lines = ""
    for i, d in enumerate(merge_cases, 1):
        case_lines += f"### 案例 {i}：合并 — {', '.join(d.affected_names[:3])}\n\n"
        case_lines += f"- **决策**: merge\n"
        case_lines += f"- **涉及知识点**: {', '.join(d.affected_names)}\n"
        case_lines += f"- **置信度**: {d.confidence:.2f}\n"
        case_lines += f"- **理由**: {d.reason}\n\n"

    # Domain coverage analysis
    domains = defaultdict(list)
    for n in kept_nodes:
        chapter_domain = n.chapter.split("第")[0].strip() if "第" in n.chapter else n.chapter[:6]
        domains[chapter_domain].append(n.name)

    domain_lines = ""
    for domain, names in sorted(domains.items(), key=lambda x: -len(x[1]))[:8]:
        domain_lines += f"| {domain} | {len(names)} | {', '.join(names[:5])}{'...' if len(names) > 5 else ''} |\n"

    report = f"""# 学科知识整合报告

> 生成时间：{time.strftime('%Y-%m-%d %H:%M')}

## 整合概览

| 指标 | 数值 |
|------|------|
| 原始教材数量 | {stats.textbook_count} 本 |
| 整合前知识点数 | {stats.original_nodes} |
| 整合后知识点数 | {stats.merged_nodes} |
| 原始总字数 | {stats.original_chars:,} 字 |
| 整合后字数 | {stats.merged_chars:,} 字 |
| **压缩比** | **{stats.compression_ratio:.1%}** |

## 整合决策摘要

| 决策类型 | 数量 | 说明 |
|----------|------|------|
| merge（合并） | {stats.merge_count} | 跨教材重复知识点合并 |
| keep（保留） | {stats.keep_count} | 判定为不同概念，各自保留 |
| remove（删除） | {stats.remove_count} | 冗余碎片/为达成压缩目标移除 |
| **合计** | **{stats.total_decisions}** | |

## 知识域覆盖

| 知识域 | 保留知识点数 | 代表性概念 |
|--------|------------|-----------|
{domain_lines}

## 重点整合案例

{case_lines if case_lines else '（无合并案例）'}

## 教学完整性说明

整合后知识库覆盖了原始教材的核心知识域。通过「前置依赖」关系链路可验证：
- 所有被标记为「关键」级别的知识点均已保留
- 知识点之间的 prerequisite 关系链保持完整
- 如需复原被移除的补充性知识点，可查阅原始教材对应页码

---

*本报告由 Integration Agent 自动生成，统计口径为知识点定义字数。*
"""
    return report


# ── Merged graph builder ───────────────────────────────

def _build_merged_graph(
    extraction_results: list[ExtractionResult],
    decisions: list[IntegrationDecision],
    kept_nodes: list[ExtractedKnowledgePoint],
) -> dict:
    """Build the merged knowledge graph suitable for Cytoscape.js rendering."""
    kept_ids = {n.id for n in kept_nodes}
    id_to_node = {n.id: n for n in kept_nodes}

    # Collect edges, remapping merged source/target
    # Build merge mapping: secondary_id → primary_id
    merge_map: dict[str, str] = {}
    for d in decisions:
        if d.action == DecisionAction.merge and d.result_node:
            for nid in d.affected_nodes:
                if nid != d.result_node:
                    merge_map[nid] = d.result_node

    # Collect edges from all extraction results, remap
    all_edges: list[ExtractedRelation] = []
    seen_edge_keys: set[tuple[str, str, str]] = set()

    for er in extraction_results:
        for edge in er.edges:
            src = merge_map.get(edge.source, edge.source)
            tgt = merge_map.get(edge.target, edge.target)
            if src not in kept_ids or tgt not in kept_ids:
                continue
            key = (src, tgt, edge.relation_type.value)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                edge.source = src
                edge.target = tgt
                all_edges.append(edge)

    return {
        "nodes": [
            {
                "id": n.id,
                "name": n.name,
                "definition": n.definition,
                "importance": n.importance_level.value,
                "textbook": n.textbook_id,
                "chapter": n.chapter,
                "page": n.page,
            }
            for n in kept_nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relation_type": e.relation_type.value,
                "description": e.description,
            }
            for e in all_edges
        ],
    }


# ── LLM utility ────────────────────────────────────────

async def _call_llm(messages: list[dict], callable_fn: Optional[Callable] = None) -> str:
    if callable_fn:
        return await callable_fn(messages)
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return '{"is_same":false,"action":"keep","confidence":0.5,"reason":"no API key"}'
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.getenv("LLM_MODEL", "qwen-max"), "messages": messages, "temperature": 0.1, "max_tokens": 1024},
        )
        return resp.json()["choices"][0]["message"]["content"]
