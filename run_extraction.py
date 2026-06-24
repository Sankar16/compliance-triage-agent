"""Run entity extraction against fatf_grey_list.pdf.

Usage:
    python3 run_extraction.py              # smoke test: first 3 chunks
    python3 run_extraction.py --chunks 5   # first N chunks
    python3 run_extraction.py --all        # all chunks (21 API calls)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

from tools import chunk_document, extract_compliance_entities

PDF_PATH = "sample_docs/fatf_grey_list.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compliance entity extraction on fatf_grey_list.pdf.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chunks", type=int, default=3, metavar="N", help="Number of chunks to process (default: 3)")
    group.add_argument("--all", action="store_true", help="Process all chunks")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env file.")

    client = anthropic.Anthropic(api_key=api_key)

    all_chunks = chunk_document(PDF_PATH)
    chunks_to_run = all_chunks if args.all else all_chunks[: args.chunks]

    label = "all" if args.all else f"first {len(chunks_to_run)}"
    print(f"=== extract_compliance_entities ({label} of {len(all_chunks)} chunks from {PDF_PATH}) ===\n")

    agg_claimed = 0
    agg_verified = 0
    agg_rejected = 0
    zero_verified_chunks: list[tuple[int, str | None]] = []
    still_rejected: list[tuple[int, str | None, str]] = []

    for chunk in chunks_to_run:
        print(f"\n--- chunk_index={chunk.chunk_index} section_title={chunk.section_title!r} ---")

        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capture
        entities = extract_compliance_entities(chunk, client)
        sys.stdout = old_stdout
        captured = capture.getvalue()
        print(captured, end="")

        chunk_verified = 0
        chunk_rejected = 0
        for line in captured.splitlines():
            if "quote(s) verified" in line:
                parts = line.split(":")[-1].strip()
                v_part, r_part = parts.split(",")
                chunk_verified = int(v_part.strip().split()[0])
                chunk_rejected = int(r_part.strip().split()[0])
            if "REJECTED unverifiable" in line or "REJECTED quote" in line:
                quote_repr_start = line.rfind(": ") + 2
                still_rejected.append((chunk.chunk_index, chunk.section_title, line[quote_repr_start:]))

        agg_verified += chunk_verified
        agg_rejected += chunk_rejected
        agg_claimed += chunk_verified + chunk_rejected
        if chunk_verified == 0:
            zero_verified_chunks.append((chunk.chunk_index, chunk.section_title))

        print(json.dumps(entities.model_dump(mode="json"), indent=2))
        sys.stdout.flush()

    print("\n" + "=" * 60)
    print(f"AGGREGATE SUMMARY — {PDF_PATH} ({label})")
    print("=" * 60)
    print(f"Total chunks processed    : {len(chunks_to_run)}")
    print(f"Total claimed quotes      : {agg_claimed}")
    print(f"Total verified quotes     : {agg_verified}")
    print(f"Total rejected quotes     : {agg_rejected}")
    rate = (agg_verified / agg_claimed * 100) if agg_claimed else 0.0
    print(f"Overall verification rate : {rate:.1f}%")

    if zero_verified_chunks:
        print(f"\nChunks with 0 verified extractions ({len(zero_verified_chunks)}):")
        for idx, title in zero_verified_chunks:
            print(f"  chunk_index={idx}  section_title={title!r}")
    else:
        print("\nNo chunks with 0 verified extractions.")

    if still_rejected:
        print(f"\nStill-rejected quotes ({len(still_rejected)}):")
        for idx, title, quote in still_rejected:
            print(f"  chunk_index={idx}  section_title={title!r}")
            print(f"    {quote}")
    else:
        print("\nNo quotes rejected.")


if __name__ == "__main__":
    main()
