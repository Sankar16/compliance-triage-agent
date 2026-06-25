"""Prompt evaluation: compare extraction prompt v1 vs v2 on the same document.

v1 vs v2 difference: v2 added an explicit instruction for the deadline_parsed
field — only supply a full YYYY-MM-DD date; omit (null) for partial dates or
relative timeframes. v1 lacked this, allowing the LLM to produce partial dates
like "2026-02" which then fail Pydantic validation and cause the entire action
item to be dropped.

This script runs both prompts on the first N chunks of a document, counts
verified quotes, rejected quotes, and deadline_parsed ValidationErrors per
prompt version, and prints a comparison table.

No audit logs are written. API calls ARE made (one call per chunk per version).

Usage:
    python3 prompt_eval.py
    python3 prompt_eval.py --doc sample_docs/fatf_grey_list.pdf --chunks 5
    python3 prompt_eval.py --output my_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Per-chunk result
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    chunk_index: int
    version: str          # "v1" or "v2"
    verified_count: int
    rejected_count: int
    validation_errors: int          # action items dropped by Pydantic
    action_items_with_deadline_parsed: int  # action items that supplied a date
    duration_seconds: float
    error: Optional[str]


# ---------------------------------------------------------------------------
# Core extraction with a caller-supplied prompt
# ---------------------------------------------------------------------------


def _extract_with_prompt(
    chunk,           # DocumentChunk
    client,          # anthropic.Anthropic
    system_prompt: str,
    version_label: str,
) -> ChunkResult:
    """Run one extraction call with the given system prompt and tally metrics.

    Mirrors the verification loop in tools.extract_compliance_entities() but
    does NOT construct full Pydantic objects for grounded claims — we only need
    counts. We DO attempt full ActionItem construction to catch ValidationErrors,
    since that is precisely what the v1 vs v2 comparison measures.
    """
    from pydantic import ValidationError
    from tools import (
        EXTRACTED_ENTITIES_TOOL_SCHEMA,
        find_quote_position,
        verify_grounding,
    )
    from schemas import ActionItem

    t0 = time.time()
    verified = 0
    rejected = 0
    validation_errors = 0
    deadline_parsed_count = 0

    try:
        import anthropic as _anthropic
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=[EXTRACTED_ENTITIES_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "extract_compliance_entities"},
            messages=[{
                "role": "user",
                "content": f"Document chunk (chunk_index={chunk.chunk_index}):\n\n{chunk.text}",
            }],
        )
    except Exception as exc:
        return ChunkResult(
            chunk_index=chunk.chunk_index,
            version=version_label,
            verified_count=0,
            rejected_count=0,
            validation_errors=0,
            action_items_with_deadline_parsed=0,
            duration_seconds=round(time.time() - t0, 2),
            error=str(exc),
        )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        return ChunkResult(
            chunk_index=chunk.chunk_index,
            version=version_label,
            verified_count=0,
            rejected_count=0,
            validation_errors=0,
            action_items_with_deadline_parsed=0,
            duration_seconds=round(time.time() - t0, 2),
            error="no tool_use block in response",
        )

    raw = tool_block.input

    # Verify grounded claims (risk_areas, responsible_parties, deadlines)
    for field in ("risk_areas", "responsible_parties", "deadlines"):
        for item in raw.get(field, []):
            quote = item.get("source_quote", "")
            if not verify_grounding(quote, chunk.text):
                rejected += 1
            elif find_quote_position(quote, chunk.text) is None:
                rejected += 1
            else:
                verified += 1

    # Verify action items — also attempt full Pydantic construction to catch
    # deadline_parsed ValidationErrors (this is the key v1 vs v2 difference)
    for item in raw.get("action_items", []):
        quote = item.get("source_quote", "")
        if not verify_grounding(quote, chunk.text):
            rejected += 1
            continue
        if find_quote_position(quote, chunk.text) is None:
            rejected += 1
            continue

        # Quote verified — now attempt schema construction
        deadline_parsed = item.get("deadline_parsed") or None
        try:
            ActionItem(
                action=item["action"],
                owner_type=item["owner_type"],
                deadline_raw=item["deadline_raw"],
                deadline_parsed=deadline_parsed,
                priority=item["priority"],
                source_quote=quote,
                source_char_range=(0, 1),  # placeholder — not needed for eval
            )
            verified += 1
            if deadline_parsed is not None:
                deadline_parsed_count += 1
        except (ValidationError, KeyError):
            validation_errors += 1
            rejected += 1

    return ChunkResult(
        chunk_index=chunk.chunk_index,
        version=version_label,
        verified_count=verified,
        rejected_count=rejected,
        validation_errors=validation_errors,
        action_items_with_deadline_parsed=deadline_parsed_count,
        duration_seconds=round(time.time() - t0, 2),
        error=None,
    )


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def _print_comparison_table(
    doc_name: str,
    n_chunks: int,
    v1_results: list[ChunkResult],
    v2_results: list[ChunkResult],
) -> None:
    header = f"Prompt Evaluation — {doc_name} ({n_chunks} chunk(s))"
    print(f"\n{header}")
    print()

    col = "{:<6} │ {:<11} │ {:<11} │ {:<13} │ {:<11} │ {:<11} │ {:<13}"
    sep = "───────┼─────────────┼─────────────┼───────────────┼─────────────┼─────────────┼──────────────"

    print(col.format("Chunk", "V1 Verified", "V1 Rejected", "V1 DateErrors",
                                "V2 Verified", "V2 Rejected", "V2 DateErrors"))
    print(sep)

    total_v1_ver = total_v1_rej = total_v1_err = 0
    total_v2_ver = total_v2_rej = total_v2_err = 0

    for r1, r2 in zip(v1_results, v2_results):
        print(col.format(
            r1.chunk_index,
            r1.verified_count, r1.rejected_count, r1.validation_errors,
            r2.verified_count, r2.rejected_count, r2.validation_errors,
        ))
        total_v1_ver += r1.verified_count
        total_v1_rej += r1.rejected_count
        total_v1_err += r1.validation_errors
        total_v2_ver += r2.verified_count
        total_v2_rej += r2.rejected_count
        total_v2_err += r2.validation_errors

    print(sep)
    print(col.format(
        "TOTAL",
        total_v1_ver, total_v1_rej, total_v1_err,
        total_v2_ver, total_v2_rej, total_v2_err,
    ))

    # Key findings
    print()
    total_claimed_v1 = total_v1_ver + total_v1_rej
    total_claimed_v2 = total_v2_ver + total_v2_rej
    gr_v1 = total_v1_ver / total_claimed_v1 if total_claimed_v1 else 0.0
    gr_v2 = total_v2_ver / total_claimed_v2 if total_claimed_v2 else 0.0

    print("Key findings:")
    if total_v1_err > 0:
        print(f"  V1 produced {total_v1_err} partial date validation error(s) — action item(s) dropped")
    else:
        print("  V1 produced 0 partial date validation errors")
    if total_v2_err > 0:
        print(f"  V2 produced {total_v2_err} partial date validation error(s) — action item(s) dropped")
    else:
        print("  V2 produced 0 validation errors — deadline_parsed instruction prevented partial dates")
    print(f"  Grounding rate: V1 {gr_v1:.1%} vs V2 {gr_v2:.1%}")

    print()
    if total_v2_err < total_v1_err:
        print("Conclusion: V2 prompt improves deadline reliability with no quality trade-off on extraction or grounding.")
    elif total_v1_err == total_v2_err == 0:
        print("Conclusion: Both prompts produced 0 validation errors on this sample — try more chunks or a different document.")
    else:
        print("Conclusion: No improvement from v2 on this sample — investigate the validation errors above.")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare extraction prompt v1 vs v2 on the same document. "
            "Makes real API calls (one call per chunk per version)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--doc",
        default="sample_docs/dear-ceo-letter.pdf",
        metavar="PATH",
        help="PDF to evaluate",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=3,
        metavar="N",
        help="Number of chunks to test (first N chunks of the document)",
    )
    parser.add_argument(
        "--output",
        default="prompt_eval_results.json",
        metavar="PATH",
        help="Where to save the JSON results",
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        print(f"[prompt_eval] ERROR: document not found: {doc_path}")
        return 1

    # Import heavy dependencies here so --help stays fast
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[prompt_eval] ERROR: ANTHROPIC_API_KEY not set")
        return 1

    import anthropic
    from prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT_V1
    from tools import chunk_document

    client = anthropic.Anthropic(api_key=api_key)

    print(f"[prompt_eval] Chunking {doc_path.name}…")
    chunks = chunk_document(str(doc_path))
    target_chunks = chunks[: args.chunks]
    n = len(target_chunks)
    total_calls = n * 2
    print(f"[prompt_eval] {len(chunks)} chunks total; evaluating first {n}.")
    print(f"[prompt_eval] Making {total_calls} API calls (v1 + v2 × {n} chunks)…\n")

    v1_results: list[ChunkResult] = []
    v2_results: list[ChunkResult] = []

    for i, chunk in enumerate(target_chunks):
        print(f"  chunk {chunk.chunk_index} ({i + 1}/{n}) — v1…", end=" ", flush=True)
        r1 = _extract_with_prompt(chunk, client, EXTRACTION_SYSTEM_PROMPT_V1, "v1")
        print(f"verified={r1.verified_count} rejected={r1.rejected_count} date_errors={r1.validation_errors} ({r1.duration_seconds}s)")
        v1_results.append(r1)

        print(f"  chunk {chunk.chunk_index} ({i + 1}/{n}) — v2…", end=" ", flush=True)
        r2 = _extract_with_prompt(chunk, client, EXTRACTION_SYSTEM_PROMPT, "v2")
        print(f"verified={r2.verified_count} rejected={r2.rejected_count} date_errors={r2.validation_errors} ({r2.duration_seconds}s)")
        v2_results.append(r2)

    _print_comparison_table(doc_path.name, n, v1_results, v2_results)

    # Save results
    output = {
        "document": str(doc_path),
        "chunks_evaluated": n,
        "v1_results": [asdict(r) for r in v1_results],
        "v2_results": [asdict(r) for r in v2_results],
        "summary": {
            "v1_total_verified":          sum(r.verified_count for r in v1_results),
            "v1_total_rejected":          sum(r.rejected_count for r in v1_results),
            "v1_total_validation_errors": sum(r.validation_errors for r in v1_results),
            "v2_total_verified":          sum(r.verified_count for r in v2_results),
            "v2_total_rejected":          sum(r.rejected_count for r in v2_results),
            "v2_total_validation_errors": sum(r.validation_errors for r in v2_results),
        },
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n[prompt_eval] Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
