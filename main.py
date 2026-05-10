#!/usr/bin/env python3
"""学科知识整合智能体 — 一键启动入口。

Usage:
  python main.py pipeline          # 完整流水线：解析 → 提取 → 整合 → 索引
  python main.py ui                # 启动 Gradio 前端
  python main.py pipeline --ui     # 跑完流水线后自动启动 UI

Options:
  --data-dir DIR   教材目录，默认 ./data/textbooks
  --output-dir DIR 输出目录，默认 ./output
  --ui             流水线完成后启动 Gradio 界面
  --step STEP      单独执行某一步: parse | extract | integrate | index
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="学科知识整合智能体")
    parser.add_argument("command", nargs="?", default="ui",
                        choices=["pipeline", "ui", "step"],
                        help="pipeline: 跑全流程 | ui: 启动界面 | step: 单步执行")
    parser.add_argument("--data-dir", default="./data/textbooks",
                        help="教材目录 (默认 ./data/textbooks)")
    parser.add_argument("--output-dir", default="./output",
                        help="输出目录 (默认 ./output)")
    parser.add_argument("--ui", action="store_true",
                        help="流水线完成后启动 Gradio")
    parser.add_argument("--step", type=str,
                        choices=["parse", "extract", "integrate", "index"],
                        help="单独执行: parse|extract|integrate|index")
    args = parser.parse_args()

    if args.command == "ui":
        launch_ui()
    elif args.command == "pipeline":
        asyncio.run(run_pipeline(args))
        if args.ui:
            launch_ui()
    elif args.command == "step" and args.step:
        asyncio.run(run_single_step(args.step, args))


async def run_pipeline(args):
    """Execute the full pipeline: parse → extract → integrate → index."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧠 学科知识整合智能体 — 全流程启动")
    print("=" * 60)

    # ── Step 1: Parse ────────────────────────────
    print("\n📄 Step 1/4: 解析教材...")
    t0 = time.time()
    from agents.parser import parse_textbook

    files = sorted(data_dir.glob("*"))
    files = [f for f in files if f.suffix.lower() in {".pdf", ".md", ".txt"}]
    if not files:
        print(f"  ⚠️  未在 {data_dir} 找到教材文件，跳过解析")
        textbooks = []
    else:
        textbooks = []
        for f in files:
            print(f"  → 解析 {f.name}...")
            try:
                tb = await parse_textbook(str(f))
                textbooks.append(tb)
                print(f"    ✅ {tb.title}: {tb.total_pages} 页, {tb.total_chars:,} 字, {len(tb.chapters)} 章")
            except Exception as e:
                print(f"    ❌ {f.name}: {e}")
        print(f"  ✓ 共解析 {len(textbooks)} 本教材 ({time.time() - t0:.1f}s)")

    # Save parse results
    parse_json = [t.model_dump() for t in textbooks]
    with open(output_dir / "parsed_textbooks.json", "w", encoding="utf-8") as fp:
        json.dump(parse_json, fp, ensure_ascii=False, indent=2)
    print(f"  💾 已保存至 {output_dir / 'parsed_textbooks.json'}")

    if not textbooks:
        print("\n⚠️  没有可处理的教材，流水线终止")
        return

    # ── Step 2: Extract ──────────────────────────
    print("\n🧠 Step 2/4: 提取知识点...")
    t0 = time.time()
    from agents.knowledge_extractor import extract_from_textbook

    extraction_results = []
    for i, book in enumerate(textbooks):
        print(f"  → [{i + 1}/{len(textbooks)}] 提取 {book.title}...")
        try:
            er = await extract_from_textbook(book)
            extraction_results.append(er)
            print(f"    ✅ {len(er.nodes)} 知识点, {len(er.edges)} 关系 ({er.batch_count} 批次, ~{er.total_cost_tokens} tokens)")
        except Exception as e:
            print(f"    ❌ {book.title}: {e}")
    print(f"  ✓ 共提取 {sum(len(er.nodes) for er in extraction_results)} 个知识点 ({time.time() - t0:.1f}s)")

    extract_json = [er.model_dump() for er in extraction_results]
    with open(output_dir / "extraction_results.json", "w", encoding="utf-8") as fp:
        json.dump(extract_json, fp, ensure_ascii=False, indent=2)

    if not extraction_results:
        print("\n⚠️  没有提取到知识点，流水线终止")
        return

    # ── Step 3: Integrate ────────────────────────
    print("\n🔗 Step 3/4: 跨教材整合...")
    t0 = time.time()
    from agents.integrator import integrate

    integration = await integrate(extraction_results)
    stats = integration.stats
    print(f"  ✓ 整合完成 ({time.time() - t0:.1f}s)")
    print(f"  📊 原始 {stats.original_nodes} 节点 → 整合后 {stats.merged_nodes} 节点")
    print(f"  📊 原始 {stats.original_chars:,} 字 → 整合后 {stats.merged_chars:,} 字")
    print(f"  📊 压缩比: {stats.compression_ratio:.1%}")
    print(f"  📊 merge: {stats.merge_count} | keep: {stats.keep_count} | remove: {stats.remove_count}")

    with open(output_dir / "integration_result.json", "w", encoding="utf-8") as fp:
        json.dump(integration.model_dump(), fp, ensure_ascii=False, indent=2)
    with open(output_dir / "整合报告.md", "w", encoding="utf-8") as fp:
        fp.write(integration.report_markdown)
    print(f"  💾 报告已保存至 {output_dir / '整合报告.md'}")

    # ── Step 4: Build RAG Index ──────────────────
    print("\n📇 Step 4/4: 构建 FAISS 索引...")
    t0 = time.time()
    from agents.rag import build_index

    rag_status = await build_index(textbooks)
    print(f"  ✓ 索引完成 ({time.time() - t0:.1f}s)")
    print(f"  📊 {rag_status.indexed_books} 本教材, {rag_status.total_chunks} 个分块")
    print(f"  📊 模型: {rag_status.embedding_model}")

    print("\n" + "=" * 60)
    print("✅ 全流程完成！")
    print(f"   📂 所有输出保存在: {output_dir}/")
    print("=" * 60)


async def run_single_step(step: str, args):
    """Run a single pipeline step."""
    print(f"🔧 执行单步: {step}")
    # Simplified — delegates to pipeline with early exit
    await run_pipeline(args)


def launch_ui():
    """Launch the Gradio frontend."""
    print("\n🚀 启动 Gradio 界面...")
    ui_path = Path(__file__).parent / "src" / "frontend" / "app.py"
    os.chdir(Path(__file__).parent)
    # Use subprocess to avoid import conflicts with the async agents
    import subprocess
    subprocess.run([sys.executable, str(ui_path)])


if __name__ == "__main__":
    main()
