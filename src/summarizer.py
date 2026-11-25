from src.config import SUMMARY_BLOCK_CHARS, SUMMARY_MAX_BLOCKS
from src.llm import ask_llm


def _build_blocks(pages):
    blocks, current = [], ""
    for p in pages:
        current += f"\n[Page {p['page']}]\n{p['text']}"
        if len(current) >= SUMMARY_BLOCK_CHARS:
            blocks.append(current[:SUMMARY_BLOCK_CHARS * 2])
            current = ""
    if current.strip():
        blocks.append(current[:SUMMARY_BLOCK_CHARS * 2])
    return blocks


def _sample_evenly(blocks, n):
    """Pick n blocks spread across the whole document."""
    if len(blocks) <= n:
        return blocks
    indexes = sorted({round(i * (len(blocks) - 1) / (n - 1)) for i in range(n)})
    return [blocks[i] for i in indexes]


def _final_prompt(text):
    return (
        "Write a short summary (5-10 lines) of this document.\n"
        "Cover: main purpose, key points, conclusion.\n\n"
        f"Document:\n{text}\n\nSummary:"
    )


def summarize_document(pages):
    """Summarize the whole document (map-reduce over evenly spaced blocks)."""
    blocks = _sample_evenly(_build_blocks(pages), SUMMARY_MAX_BLOCKS)
    if not blocks:
        return ""
    if len(blocks) == 1:
        return ask_llm(_final_prompt(blocks[0]))

    partial_summaries = []
    for block in blocks:
        partial_summaries.append(
            ask_llm("Summarize this part of a document in 3-4 lines:\n\n" + block)
        )
    return ask_llm(_final_prompt("\n\n".join(partial_summaries)))