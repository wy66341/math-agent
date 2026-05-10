#!/usr/bin/env python3
"""端到端流水线测试：解析 → 提取 → 整合 → 索引。

Usage:
  python run_pipeline.py                    # 处理 data/textbooks/ 下全部 PDF
  python run_pipeline.py --books 2          # 只处理前 2 本
  python run_pipeline.py --skip-extract     # 只跑解析
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


async def main():
    parser = argparse.ArgumentParser(description="学科知识整合智能体 — 流水线测试")
    parser.add_argument("--data-dir", default="./data/textbooks")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--books", type=int, default=0, help="最多处理 N 本 (0=全部)")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-integrate", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="每本教材最多提取 N 个批次 (0=全部，调试用)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find files ─────────────────────────────────
    files = sorted(data_dir.glob("*"))
    files = [f for f in files if f.suffix.lower() in {".pdf", ".md", ".txt"}]
    if args.books:
        files = files[:args.books]
    if not files:
        print("❌ 未找到教材文件")
        return

    print("=" * 65)
    print("🧠 学科知识整合智能体 · 流水线测试")
    print(f"📂 数据目录: {data_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"📚 待处理: {len(files)} 本教材")
    print("=" * 65)

    # ════════════════════════════════════════════════════
    # STEP 1: PARSE
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("📄 STEP 1: 解析教材")
    print("─" * 50)

    from agents.parser import parse_textbook

    t_start = time.time()
    textbooks = []
    for f in files:
        t0 = time.time()
        print(f"  → {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)...", end=" ", flush=True)
        try:
            tb = await parse_textbook(str(f))
            textbooks.append(tb)
            print(f"✅ {tb.total_pages} 页, {tb.total_chars:,} 字, {len(tb.chapters)} 章 ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"❌ {e}")

    t_parse = time.time() - t_start
    print(f"\n  ⏱️  解析耗时: {t_parse:.1f}s")
    print(f"  📊 共解析 {len(textbooks)} 本教材, {sum(t.total_chars for t in textbooks):,} 字")

    # Save
    meta = [t.model_dump() for t in textbooks]
    out_meta = output_dir / "textbook_metadata.json"
    with open(out_meta, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"  💾 {out_meta}")

    if not textbooks:
        print("\n❌ 没有成功解析的教材，流水线终止")
        return

    if args.skip_extract:
        print("\n⏭️  跳过提取和后续步骤")
        return

    # ════════════════════════════════════════════════════
    # STEP 2: EXTRACT (with checkpoint/resume)
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("🧠 STEP 2: 提取知识点 (断点续传模式)")
    print("─" * 50)

    from agents.knowledge_extractor import extract_from_textbook
    from models.schemas import ExtractionResult

    t_start = time.time()
    extraction_results = []

    # Checkpoint: load previously completed books
    temp_file = output_dir / "temp_extraction.json"
    completed_books: set[str] = set()
    if temp_file.exists():
        try:
            saved = json.loads(temp_file.read_text(encoding="utf-8"))
            extraction_results = [ExtractionResult(**s) for s in saved.get("results", [])]
            completed_books = {er.textbook_id for er in extraction_results}
            print(f"  📂 恢复断点: {len(extraction_results)} 本已完成, {len(completed_books)} 本跳过")
        except Exception:
            pass

    total_batches_done = 0
    total_batches_all = 0
    last_log_time = [time.time()]

    async def on_batch(batch_num: int, total: int, tokens: int):
        nonlocal total_batches_done, total_batches_all
        total_batches_done += 1
        now = time.time()
        if now - last_log_time[0] >= 60:
            elapsed = now - t_start
            done = total_batches_done
            remaining = total_batches_all - done
            if done > 0 and remaining > 0:
                rate = done / elapsed
                eta = remaining / rate if rate > 0 else 0
                print(f"  📊 进度 {done}/{total_batches_all} 批次 ({done/max(total_batches_all,1)*100:.1f}%) "
                      f"| 预计剩余 ~{eta/60:.0f} 分钟", flush=True)
            last_log_time[0] = now

    # Count total batches first
    for book in textbooks:
        if book.textbook_id in completed_books:
            continue
        n_batches = 0
        for ch in book.chapters:
            if ch.content.strip():
                n_batches += max(1, len(ch.content) // 4000 + 1)
        total_batches_all += n_batches

    print(f"  📊 总计 ~{total_batches_all} 个批次待处理")

    for i, book in enumerate(textbooks):
        if book.textbook_id in completed_books:
            print(f"  ⏭️  [{i + 1}/{len(textbooks)}] {book.title} (已完成，跳过)")
            continue

        t0 = time.time()
        total_batches_done += (extraction_results[-1].batch_count if extraction_results else 0)
        print(f"  → [{i + 1}/{len(textbooks)}] {book.title} ({book.total_chars:,} 字)...", flush=True)
        try:
            er = await extract_from_textbook(book, on_batch_complete=on_batch, max_batches=args.max_batches)
            extraction_results.append(er)
            # Save checkpoint immediately after each book
            ckpt = {"results": [r.model_dump() for r in extraction_results]}
            temp_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2))
            elapsed = time.time() - t0
            print(f"  ✅ {len(er.nodes)} 知识点, {len(er.edges)} 关系, "
                  f"{er.batch_count} 批次 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ❌ {book.title}: {e}")
            import traceback; traceback.print_exc()
            # Save partial progress anyway
            if extraction_results:
                ckpt = {"results": [r.model_dump() for r in extraction_results]}
                temp_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2))

    t_extract = time.time() - t_start
    total_nodes = sum(len(er.nodes) for er in extraction_results)
    total_edges = sum(len(er.edges) for er in extraction_results)
    print(f"\n  ⏱️  提取耗时: {t_extract:.1f}s")
    print(f"  📊 共提取 {total_nodes} 知识点, {total_edges} 关系")

    # Clean checkpoint after successful completion
    if temp_file.exists():
        temp_file.rename(output_dir / "extraction_checkpoint_done.json")

    # Save initial knowledge graph
    out_kg = output_dir / "initial_knowledge_graph.json"
    kg_data = []
    for er in extraction_results:
        kg_data.append({
            "textbook_id": er.textbook_id,
            "node_count": len(er.nodes),
            "edge_count": len(er.edges),
            "nodes": [n.model_dump() for n in er.nodes],
            "edges": [e.model_dump() for e in er.edges],
        })
    with open(out_kg, "w", encoding="utf-8") as fp:
        json.dump(kg_data, fp, ensure_ascii=False, indent=2)
    print(f"  💾 {out_kg}")

    if not extraction_results:
        print("\n❌ 没有提取到知识点，流水线终止")
        return

    if args.skip_integrate:
        print("\n⏭️  跳过整合和后续步骤")
        return

    # ════════════════════════════════════════════════════
    # STEP 3: INTEGRATE
    # ════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("🔗 STEP 3: 跨教材整合")
    print("─" * 50)

    from agents.integrator import integrate

    t_start = time.time()
    print(f"  → 语义对齐 (两阶段: Embedding 粗筛 + LLM 精判)...")

    async def on_progress(phase: str, current: int, total: int):
        if current % 10 == 0 and total > 0:
            print(f"    {phase}: {current}/{total}", flush=True)

    try:
        integration = await integrate(extraction_results, on_progress=on_progress)
        t_integrate = time.time() - t_start
        s = integration.stats
        print(f"\n  ⏱️  整合耗时: {t_integrate:.1f}s")
        print(f"  📊 原始 {s.original_nodes} 节点 → 整合后 {s.merged_nodes} 节点")
        print(f"  📊 原始 {s.original_chars:,} 字 → 整合后 {s.merged_chars:,} 字")
        print(f"  📊 压缩比: {s.compression_ratio:.1%}")
        print(f"  📊 merge: {s.merge_count} | keep: {s.keep_count} | remove: {s.remove_count}")

        # Save integration report
        out_report = output_dir / "integration_report.md"
        with open(out_report, "w", encoding="utf-8") as fp:
            fp.write(integration.report_markdown)
        print(f"  💾 {out_report}")

        # Save integration result JSON
        out_result = output_dir / "integration_result.json"
        with open(out_result, "w", encoding="utf-8") as fp:
            json.dump(integration.model_dump(), fp, ensure_ascii=False, indent=2)
        print(f"  💾 {out_result}")
    except Exception as e:
        print(f"\n  ❌ 整合失败: {e}")
        import traceback; traceback.print_exc()
        return

    # ════════════════════════════════════════════════════
    # STEP 4: INDEX (FAISS)
    # ════════════════════════════════════════════════════
    if args.skip_index:
        print("\n⏭️  跳过索引步骤")
    else:
        print("\n" + "─" * 50)
        print("📇 STEP 4: 构建 FAISS 索引")
        print("─" * 50)
        t_start = time.time()
        from agents.rag import build_index
        try:
            status = await build_index(textbooks)
            print(f"  ⏱️  索引耗时: {time.time() - t_start:.1f}s")
            print(f"  📊 {status.indexed_books} 本教材, {status.total_chunks} 个分块")
            print(f"  📊 模型: {status.embedding_model}")
        except Exception as e:
            print(f"  ❌ 索引失败: {e}")
            import traceback; traceback.print_exc()

    # ════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("✅ 流水线完成")
    print(f"   ⏱️  总耗时: {t_parse + t_extract + t_integrate:.1f}s")
    print(f"   📂 输出目录: {output_dir.resolve()}")
    print(f"   📄 {out_meta.name}")
    if not args.skip_extract and extraction_results:
        print(f"   📄 {out_kg.name}")
    if not args.skip_integrate:
        print(f"   📄 {out_report.name}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
