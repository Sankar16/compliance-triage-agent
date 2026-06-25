"""Batch compliance triage runner.

Processes multiple PDFs through the compliance triage pipeline in sequence
(one at a time to avoid rate limits). For each document, runs the full
extraction pipeline up to the human checkpoint, optionally simulates a
human approval, and collects results into a summary JSON file.

Usage:
    python3 batch_triage.py sample_docs/
    python3 batch_triage.py doc1.pdf doc2.pdf doc3.pdf
    python3 batch_triage.py sample_docs/ --no-human-sim
    python3 batch_triage.py sample_docs/ --output-dir results/
    python3 batch_triage.py sample_docs/ --document-id-prefix run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _resolve_pdf_paths(raw_paths: list[str]) -> list[Path]:
    """Expand any folder arguments to their contained PDFs; pass files through.

    Returns a deduplicated, ordered list of .pdf paths. Prints a warning and
    skips any non-PDF files and missing paths.
    """
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            pdfs = sorted(p.glob("*.pdf"))
            if not pdfs:
                print(f"[batch] WARNING: no .pdf files found in {p}")
            for pdf in pdfs:
                if pdf not in seen:
                    seen.add(pdf)
                    resolved.append(pdf)
        elif p.suffix.lower() == ".pdf":
            if not p.exists():
                print(f"[batch] WARNING: file not found, skipping: {p}")
                continue
            if p not in seen:
                seen.add(p)
                resolved.append(p)
        else:
            print(f"[batch] WARNING: not a PDF file, skipping: {p}")

    return resolved


def _print_summary_table(results: list[dict]) -> None:
    """Print a Unicode box-drawing results table to stdout."""
    col_widths = {
        "name": max(len("Document"), max((len(r["document_name"]) for r in results), default=0)),
        "domain": max(len("Domain"), max((len(r["domain"] or "—") for r in results), default=0)),
        "urgency": max(len("Urgency"), max((len(r["urgency"] or "—") for r in results), default=0)),
        "status": max(len("Status"), max((len(r["status"] or "—") for r in results), default=0)),
    }

    def _row(*cells, sep="║", pad=" "):
        parts = []
        for text, width in zip(cells, col_widths.values()):
            parts.append(f"{pad}{str(text):<{width}}{pad}")
        return sep + sep.join(parts) + sep

    def _divider(left, mid, right, horiz="═"):
        parts = []
        for width in col_widths.values():
            parts.append(horiz * (width + 2))
        return left + mid.join(parts) + right

    print()
    print(_divider("╔", "╦", "╗"))
    print(_row("Document", "Domain", "Urgency", "Status"))
    print(_divider("╠", "╬", "╣"))
    for r in results:
        print(_row(
            r["document_name"],
            (r["domain"] or "—").upper(),
            (r["urgency"] or "—").upper(),
            r["status"].upper() if r["status"] else "—",
        ))
    print(_divider("╚", "╩", "╝"))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="batch_triage.py",
        description="Run the compliance triage pipeline over multiple PDFs in sequence.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="One or more PDF files, or a single folder containing PDFs.",
    )
    parser.add_argument(
        "--no-human-sim",
        action="store_true",
        default=False,
        help=(
            "Stop at the human-review checkpoint without simulating approval. "
            "Documents will be saved with status=pending_human_review."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="audit_logs",
        metavar="DIR",
        help="Directory for audit logs and the batch summary JSON (default: audit_logs/).",
    )
    parser.add_argument(
        "--document-id-prefix",
        default="batch",
        metavar="PREFIX",
        help="Prefix for generated document IDs, e.g. 'run1' → 'run1-001' (default: batch).",
    )
    args = parser.parse_args()

    pdf_paths = _resolve_pdf_paths(args.paths)
    if not pdf_paths:
        print("[batch] ERROR: no valid PDF files found. Nothing to process.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(pdf_paths)
    batch_id = str(uuid.uuid4())[:8]
    batch_started_at = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*60}")
    print(f"BATCH TRIAGE — {total} document(s)")
    print(f"Batch ID   : {batch_id}")
    print(f"Output dir : {output_dir}")
    print(f"Human sim  : {'disabled (--no-human-sim)' if args.no_human_sim else 'enabled (auto-approve)'}")
    print(f"{'='*60}")

    # Defer the import until after argument parsing so --help works without
    # requiring ANTHROPIC_API_KEY or heavy dependencies in the environment.
    from agent import resume_with_decision, run_triage

    results: list[dict] = []
    total_errors = 0
    batch_wall_start = time.time()

    for i, pdf_path in enumerate(pdf_paths):
        document_id = f"{args.document_id_prefix}-{i + 1:03d}"
        print(f"\n[{i + 1}/{total}] Processing: {pdf_path.name}  (id={document_id})")

        doc_result: dict = {
            "document_id": document_id,
            "document_name": pdf_path.name,
            "domain": None,
            "urgency": None,
            "routing_proposal": None,
            "flag_count": 0,
            "duration_seconds": 0.0,
            "status": None,
            "audit_log_path": None,
            "error": None,
        }

        t0 = time.time()
        try:
            triage_out = run_triage(
                document_path=str(pdf_path),
                document_id=document_id,
            )
            state = triage_out["state"]
            graph = triage_out["graph"]
            thread_id = triage_out["thread_id"]

            tr = state.get("triage_result", {})
            doc_result["domain"] = tr.get("classification", {}).get("compliance_domain")
            doc_result["urgency"] = tr.get("classification", {}).get("urgency_level")
            doc_result["routing_proposal"] = tr.get("routing_proposal", {}).get("recommended_owner")
            doc_result["flag_count"] = len(tr.get("ambiguity_flags", []))
            doc_result["audit_log_path"] = state.get("audit_log_path")
            doc_result["status"] = tr.get("status", "pending_human_review")

            if not args.no_human_sim:
                final_state = resume_with_decision(
                    graph=graph,
                    thread_id=thread_id,
                    action="approved",
                    reviewer_id="batch-auto",
                    final_routing=doc_result["routing_proposal"] or "unknown",
                )
                final_tr = final_state.get("triage_result", {})
                doc_result["status"] = final_tr.get("status", "approved")
                # audit_log_path is updated by record_decision's resave
                doc_result["audit_log_path"] = state.get("audit_log_path")

            print(
                f"  ✓ domain={doc_result['domain']} urgency={doc_result['urgency']} "
                f"flags={doc_result['flag_count']} status={doc_result['status']}"
            )

        except Exception as exc:
            doc_result["status"] = "FAILED"
            doc_result["error"] = str(exc)
            total_errors += 1
            print(f"  ✗ FAILED: {exc}")

        doc_result["duration_seconds"] = round(time.time() - t0, 2)
        results.append(doc_result)

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    _print_summary_table(results)

    total_elapsed = round(time.time() - batch_wall_start, 1)
    successful = total - total_errors
    print(
        f"\nTotal: {total} document(s) | "
        f"Time: {total_elapsed}s | "
        f"Errors: {total_errors}"
    )

    # -------------------------------------------------------------------------
    # Write batch_summary_{timestamp}.json
    # -------------------------------------------------------------------------
    batch_completed_at = datetime.now(timezone.utc).isoformat()
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    summary_path = output_dir / f"batch_summary_{timestamp_str}.json"

    summary = {
        "batch_id": batch_id,
        "started_at": batch_started_at,
        "completed_at": batch_completed_at,
        "total_documents": total,
        "successful": successful,
        "failed": total_errors,
        "documents": results,
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"Batch summary saved to: {summary_path}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
