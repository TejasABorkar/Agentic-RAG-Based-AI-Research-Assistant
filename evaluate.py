"""Evaluation script for the project review.

Compares three systems on a prepared question set:
  A = LLM only, B = RAG (document only), C = RAG + web fallback
and measures retrieval hit@3, embedding time and response time.

Example:
  python evaluate.py --pdf "data\\your.pdf" --questions questions.json --embedding all-MiniLM-L6-v2 --out results_minilm.csv
"""
import argparse
import csv
import json
import os
import sys
import time
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True)
parser.add_argument("--questions", required=True)
parser.add_argument("--embedding", default=None,
                    help="all-MiniLM-L6-v2 or BAAI/bge-small-en-v1.5")
parser.add_argument("--chunk-size", type=int, default=None)
parser.add_argument("--chunk-overlap", type=int, default=None)
parser.add_argument("--delay", type=float, default=2.0,
                    help="seconds to wait between questions (API limits)")
parser.add_argument("--out", default="results.csv")
args = parser.parse_args()

# These must be set BEFORE importing the src modules
if args.embedding:
    os.environ["EMBEDDING_MODEL"] = args.embedding
if args.chunk_size:
    os.environ["CHUNK_SIZE"] = str(args.chunk_size)
if args.chunk_overlap:
    os.environ["CHUNK_OVERLAP"] = str(args.chunk_overlap)

from src.chunker import chunk_pages  # noqa: E402
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL  # noqa: E402
from src.llm import LLMError, ask_llm  # noqa: E402
from src.pdf_loader import load_pdf  # noqa: E402
from src.qa import answer_question  # noqa: E402
from src.vector_store import _get_model, add_chunks, search_chunks  # noqa: E402


def page_hit(hits, page):
    return any(h["page"] <= page <= h["page_end"] for h in hits)


def timed(fn, *a, **k):
    start = time.time()
    result = fn(*a, **k)
    return result, round(time.time() - start, 2)


print(f"Embedding model: {EMBEDDING_MODEL} | chunk size: {CHUNK_SIZE} "
      f"| overlap: {CHUNK_OVERLAP}")

with open(args.questions, encoding="utf-8") as f:
    questions = json.load(f)

pages = load_pdf(args.pdf)
chunks, metadatas = chunk_pages(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

_get_model()  # load the model first so loading time is not counted
collection = "eval_" + uuid.uuid4().hex
_, embed_time = timed(add_chunks, chunks, metadatas, collection)
print(f"Embedding + storing time: {embed_time}s")

rows = []
for i, item in enumerate(questions, 1):
    q = item["question"]
    in_doc = bool(item.get("in_doc", True))
    exp_page = item.get("page")
    print(f"[{i}/{len(questions)}] {q}")

    # System A: LLM only
    try:
        a_answer, a_time = timed(ask_llm, q)
    except LLMError as e:
        a_answer, a_time = f"ERROR: {e}"[:200], 0

    # Retrieval check (top 3)
    hits, _ = timed(search_chunks, q, 3, collection)
    hit = page_hit(hits, exp_page) if (in_doc and exp_page) else ""

    # System B: RAG only, System C: RAG + web
    b, b_time = timed(answer_question, q, collection, None, False)
    c, c_time = timed(answer_question, q, collection, None, True)

    b_route_ok = (b["source"] == "document") if in_doc else (b["source"] == "none")
    c_route_ok = (c["source"] == "document") if in_doc else (c["source"] == "web")
    b_page_ok = (exp_page in b["pages"]) if (
        in_doc and exp_page and b["source"] == "document") else ""
    c_page_ok = (exp_page in c["pages"]) if (
        in_doc and exp_page and c["source"] == "document") else ""

    rows.append({
        "question": q,
        "in_doc": in_doc,
        "expected_answer": item.get("expected", ""),
        "expected_page": exp_page,
        "retrieval_hit_top3": hit,
        "A_answer": a_answer, "A_time_s": a_time, "A_correct": "",
        "B_answer": b["answer"], "B_source": b["source"], "B_pages": b["pages"],
        "B_route_ok": b_route_ok, "B_page_ok": b_page_ok,
        "B_time_s": b_time, "B_correct": "",
        "C_answer": c["answer"], "C_source": c["source"], "C_pages": c["pages"],
        "C_urls": " ; ".join(s["url"] for s in c["web_sources"]),
        "C_route_ok": c_route_ok, "C_page_ok": c_page_ok,
        "C_time_s": c_time, "C_correct": "",
    })
    time.sleep(args.delay)

with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def avg(key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else 0


hit_rows = [r for r in rows if r["retrieval_hit_top3"] != ""]
hits_total = sum(1 for r in hit_rows if r["retrieval_hit_top3"])
print("\n===== SUMMARY =====")
print(f"Embedding model        : {EMBEDDING_MODEL}")
print(f"Embedding time         : {embed_time}s for {len(chunks)} chunks")
print(f"Retrieval hit@3        : {hits_total}/{len(hit_rows)}")
print(f"Correct routing  B / C : {sum(r['B_route_ok'] for r in rows)}/{len(rows)}"
      f"  /  {sum(r['C_route_ok'] for r in rows)}/{len(rows)}")
print(f"Avg time A / B / C     : {avg('A_time_s')}s / {avg('B_time_s')}s / {avg('C_time_s')}s")
print(f"\nFill the A_correct, B_correct, C_correct columns by hand in {args.out}")