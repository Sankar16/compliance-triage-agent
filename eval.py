"""Evaluation framework for the Compliance Triage Agent.

Measures three dimensions of quality against human-labeled ground truth:

1. EXTRACTION QUALITY   — grounding_rate, risk_area_coverage, action_item_coverage
2. CLASSIFICATION ACCURACY — domain_match, urgency_match vs ground_truth.yaml labels
3. ROUTING QUALITY      — proposed owner, confidence, flag count

Evaluation runs on existing audit logs — no API calls needed.
All paid pipeline runs are already in audit_logs/; evaluation reads those files.

Usage:
    python3 eval.py
    python3 eval.py --audit-dir audit_logs/ --ground-truth ground_truth.yaml
    python3 eval.py --output my_report.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    document_name: str
    document_id: str

    # Extraction quality
    grounding_rate: float        # verified_quotes / total_claimed_quotes
    total_risk_areas: int
    total_action_items: int
    total_deadlines: int
    risk_area_coverage: float    # % of must_contain_risk_areas found
    action_item_coverage: float  # % of must_contain_action_items found

    # Classification accuracy
    expected_domain: str
    actual_domain: str
    domain_correct: bool
    expected_urgency: str
    actual_urgency: str
    urgency_correct: bool
    classification_confidence: float

    # Routing quality
    expected_owner_role: str
    actual_owner: str
    actual_confidence: float
    flag_count: int

    # Overall
    passed: bool              # domain_correct AND urgency_correct AND grounding_rate >= 0.85
    duration_seconds: float
    error: Optional[str]


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------


def load_ground_truth(path: str = "ground_truth.yaml") -> dict:
    """Load the human-labeled ground truth file.

    Returns a dict keyed by document_name for O(1) lookup.
    Raises FileNotFoundError if the file is missing.
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required: pip install pyyaml"
        )

    gt_path = Path(path)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}\n"
            "Create ground_truth.yaml at the project root to enable evaluation."
        )

    with open(gt_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    documents = raw.get("documents", [])
    return {doc["document_name"]: doc for doc in documents}


# ---------------------------------------------------------------------------
# Extraction evaluation
# ---------------------------------------------------------------------------


def evaluate_extraction(
    triage_result: dict,
    ground_truth_doc: dict,
) -> dict:
    """Check extraction quality against ground truth labels.

    grounding_rate: fraction of extracted items that have a non-empty
    source_quote. Because tools.py drops unverified quotes before saving,
    all items in a valid audit log will have source_quote set — the rate
    reflects how many items the pipeline attempted vs. how many it could verify.

    Coverage checks use case-insensitive substring matching across all
    extracted values in all entities, to be robust against exact phrasing
    differences between the ground truth labels and actual extraction output.
    """
    entities = triage_result.get("all_entities", [])

    total_risk_areas = 0
    total_action_items = 0
    total_deadlines = 0
    total_claimed = 0
    verified = 0

    all_risk_area_values: list[str] = []
    all_action_item_descriptions: list[str] = []

    for entity in entities:
        for ra in entity.get("risk_areas", []):
            total_risk_areas += 1
            total_claimed += 1
            if ra.get("source_quote"):
                verified += 1
            all_risk_area_values.append(ra.get("value", "").lower())

        for ai in entity.get("action_items", []):
            total_action_items += 1
            total_claimed += 1
            if ai.get("source_quote"):
                verified += 1
            # combine action text + deadline_raw for substring search
            combined = (ai.get("action", "") + " " + ai.get("deadline_raw", "")).lower()
            all_action_item_descriptions.append(combined)

        for rp in entity.get("responsible_parties", []):
            total_claimed += 1
            if rp.get("source_quote"):
                verified += 1

        for dl in entity.get("deadlines", []):
            total_deadlines += 1
            total_claimed += 1
            if dl.get("source_quote"):
                verified += 1

    grounding_rate = (verified / total_claimed) if total_claimed > 0 else 0.0

    # Coverage: % of must_contain_* terms found anywhere in extracted values
    must_risk = ground_truth_doc.get("must_contain_risk_areas", [])
    if must_risk:
        found_risk = sum(
            1 for term in must_risk
            if any(term.lower() in val for val in all_risk_area_values)
        )
        risk_area_coverage = found_risk / len(must_risk)
    else:
        risk_area_coverage = 1.0  # no requirements → trivially satisfied

    must_action = ground_truth_doc.get("must_contain_action_items", [])
    if must_action:
        found_action = sum(
            1 for term in must_action
            if any(term.lower() in desc for desc in all_action_item_descriptions)
        )
        action_item_coverage = found_action / len(must_action)
    else:
        action_item_coverage = 1.0

    return {
        "grounding_rate": round(grounding_rate, 4),
        "total_risk_areas": total_risk_areas,
        "total_action_items": total_action_items,
        "total_deadlines": total_deadlines,
        "risk_area_coverage": round(risk_area_coverage, 4),
        "action_item_coverage": round(action_item_coverage, 4),
    }


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


def evaluate_classification(
    triage_result: dict,
    ground_truth_doc: dict,
) -> dict:
    """Check classification accuracy against ground truth labels."""
    cls = triage_result.get("classification", {})

    actual_domain = cls.get("compliance_domain", "")
    actual_urgency = cls.get("urgency_level", "")
    confidence = cls.get("confidence", 0.0)

    expected_domain = ground_truth_doc.get("expected_domain", "")
    expected_urgency = ground_truth_doc.get("expected_urgency", "")

    return {
        "expected_domain": expected_domain,
        "actual_domain": actual_domain,
        "domain_correct": actual_domain.lower() == expected_domain.lower(),
        "expected_urgency": expected_urgency,
        "actual_urgency": actual_urgency,
        "urgency_correct": actual_urgency.lower() == expected_urgency.lower(),
        "classification_confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Single-document evaluation
# ---------------------------------------------------------------------------


def evaluate_single_document(
    audit_log_path: str,
    ground_truth: dict,
) -> EvaluationResult:
    """Load an existing audit log and evaluate it against ground truth.

    Does NOT re-run the pipeline — reads the saved JSON file only.
    No API calls are made.
    """
    import time

    path = Path(audit_log_path)
    t0 = time.time()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return EvaluationResult(
            document_name=path.name,
            document_id="",
            grounding_rate=0.0,
            total_risk_areas=0,
            total_action_items=0,
            total_deadlines=0,
            risk_area_coverage=0.0,
            action_item_coverage=0.0,
            expected_domain="",
            actual_domain="",
            domain_correct=False,
            expected_urgency="",
            actual_urgency="",
            urgency_correct=False,
            classification_confidence=0.0,
            expected_owner_role="",
            actual_owner="",
            actual_confidence=0.0,
            flag_count=0,
            passed=False,
            duration_seconds=round(time.time() - t0, 3),
            error=str(exc),
        )

    triage_result = payload.get("triage_result", {})
    document_name = triage_result.get("document_name", path.stem)
    document_id = triage_result.get("document_id", "")

    # Match against ground truth by document_name
    gt_doc = ground_truth.get(document_name)
    if gt_doc is None:
        return EvaluationResult(
            document_name=document_name,
            document_id=document_id,
            grounding_rate=0.0,
            total_risk_areas=0,
            total_action_items=0,
            total_deadlines=0,
            risk_area_coverage=0.0,
            action_item_coverage=0.0,
            expected_domain="(no ground truth)",
            actual_domain=triage_result.get("classification", {}).get("compliance_domain", ""),
            domain_correct=False,
            expected_urgency="(no ground truth)",
            actual_urgency=triage_result.get("classification", {}).get("urgency_level", ""),
            urgency_correct=False,
            classification_confidence=triage_result.get("classification", {}).get("confidence", 0.0),
            expected_owner_role="(no ground truth)",
            actual_owner=triage_result.get("routing_proposal", {}).get("recommended_owner", ""),
            actual_confidence=triage_result.get("routing_proposal", {}).get("confidence", 0.0),
            flag_count=len(triage_result.get("ambiguity_flags", [])),
            passed=False,
            duration_seconds=round(time.time() - t0, 3),
            error="no matching ground truth entry",
        )

    extraction = evaluate_extraction(triage_result, gt_doc)
    classification = evaluate_classification(triage_result, gt_doc)

    routing = triage_result.get("routing_proposal", {})
    actual_owner = routing.get("recommended_owner", "")
    actual_confidence = routing.get("confidence", 0.0)
    flag_count = len(triage_result.get("ambiguity_flags", []))
    expected_owner_role = gt_doc.get("expected_owner_role", "")

    grounding_rate = extraction["grounding_rate"]
    passed = (
        classification["domain_correct"]
        and classification["urgency_correct"]
        and grounding_rate >= 0.85
    )

    return EvaluationResult(
        document_name=document_name,
        document_id=document_id,
        grounding_rate=grounding_rate,
        total_risk_areas=extraction["total_risk_areas"],
        total_action_items=extraction["total_action_items"],
        total_deadlines=extraction["total_deadlines"],
        risk_area_coverage=extraction["risk_area_coverage"],
        action_item_coverage=extraction["action_item_coverage"],
        expected_domain=classification["expected_domain"],
        actual_domain=classification["actual_domain"],
        domain_correct=classification["domain_correct"],
        expected_urgency=classification["expected_urgency"],
        actual_urgency=classification["actual_urgency"],
        urgency_correct=classification["urgency_correct"],
        classification_confidence=classification["classification_confidence"],
        expected_owner_role=expected_owner_role,
        actual_owner=actual_owner,
        actual_confidence=actual_confidence,
        flag_count=flag_count,
        passed=passed,
        duration_seconds=round(time.time() - t0, 3),
        error=None,
    )


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


def run_evaluation(
    audit_log_dir: str = "audit_logs",
    ground_truth_path: str = "ground_truth.yaml",
) -> list[EvaluationResult]:
    """Find all audit logs, match to ground truth, evaluate each.

    Skips audit logs with no matching ground truth entry (logs a warning).
    Returns one EvaluationResult per matched audit log.
    The most recent log is used when multiple logs exist for the same document.
    """
    gt = load_ground_truth(ground_truth_path)

    log_dir = Path(audit_log_dir)
    if not log_dir.exists():
        print(f"[eval] WARNING: audit log directory not found: {log_dir}")
        return []

    all_logs = sorted(log_dir.glob("*.json"))
    # Exclude batch_summary files
    audit_logs = [p for p in all_logs if not p.name.startswith("batch_summary")]

    if not audit_logs:
        print(f"[eval] WARNING: no audit log JSON files found in {log_dir}")
        return []

    # Group by document_id (filename prefix before first timestamp underscore).
    # Keep only the newest file per document_id — logs are named
    # {document_id}_{timestamp}.json and sort() gives chronological order.
    newest_by_doc_id: dict[str, Path] = {}
    for log_path in audit_logs:
        # Extract document_id: everything before the last _YYYY-MM-DDT... suffix
        stem = log_path.stem
        # Find the timestamp part: look for the last occurrence of _YYYY
        import re
        m = re.search(r"_(\d{4}-\d{2}-\d{2}T)", stem)
        if m:
            doc_id = stem[: m.start()]
        else:
            doc_id = stem
        newest_by_doc_id[doc_id] = log_path  # later in sort = newer

    results: list[EvaluationResult] = []
    for doc_id, log_path in newest_by_doc_id.items():
        result = evaluate_single_document(str(log_path), gt)
        if result.error == "no matching ground truth entry":
            print(
                f"[eval] SKIP {result.document_name!r} — "
                "no entry in ground_truth.yaml (add one to include it in evaluation)"
            )
            continue
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_evaluation_report(results: list[EvaluationResult]) -> None:
    """Print a formatted evaluation report with a Unicode box-drawing table."""
    if not results:
        print("\n[eval] No results to report. Check audit_logs/ and ground_truth.yaml.")
        return

    col_w = {
        "doc":     max(len("Document"),    max(len(r.document_name) for r in results)),
        "domain":  max(len("Domain"),      max(len(r.actual_domain or "") + 4 for r in results)),
        "urgency": max(len("Urgency"),     max(len(r.actual_urgency or "") + 4 for r in results)),
        "ground":  max(len("Grounding"),   len("100.0%")),
    }

    def _cell(text: str, width: int) -> str:
        return f" {text:<{width}} "

    def _divider(left: str, mid: str, right: str, horiz: str = "═") -> str:
        return left + mid.join(horiz * (w + 2) for w in col_w.values()) + right

    def _row(*cells: tuple[str, str]) -> str:
        return "║" + "║".join(_cell(text, w) for (text, w) in zip(cells, col_w.values())) + "║"

    print()
    print(_divider("╔", "╦", "╗"))
    header_title = "  COMPLIANCE TRIAGE AGENT — EVALUATION REPORT"
    total_inner_width = sum(w + 2 for w in col_w.values()) + len(col_w) - 1
    print(f"║{header_title:<{total_inner_width}}║")
    print(_divider("╠", "╦", "╣"))
    print(_row("Document", "Domain", "Urgency", "Grounding"))
    print(_divider("╠", "╬", "╣"))

    for r in results:
        domain_str  = ("✅ " if r.domain_correct  else "❌ ") + r.actual_domain.upper()
        urgency_str = ("✅ " if r.urgency_correct else "❌ ") + r.actual_urgency.upper()[:4]
        ground_str  = f"{r.grounding_rate * 100:.1f}%"
        print(_row(r.document_name, domain_str, urgency_str, ground_str))

    print(_divider("╚", "╩", "╝"))

    # Aggregate stats
    total        = len(results)
    passed       = sum(1 for r in results if r.passed)
    avg_ground   = sum(r.grounding_rate for r in results) / total
    domain_ok    = sum(1 for r in results if r.domain_correct)
    urgency_ok   = sum(1 for r in results if r.urgency_correct)
    avg_flags    = sum(r.flag_count for r in results) / total
    avg_conf     = sum(r.classification_confidence for r in results) / total

    print(f"\nOverall: {passed}/{total} passed")
    print(f"Average grounding rate:    {avg_ground * 100:.1f}%")
    print(f"Domain accuracy:           {domain_ok * 100 // total}% ({domain_ok}/{total})")
    print(f"Urgency accuracy:          {urgency_ok * 100 // total}% ({urgency_ok}/{total})")
    print(f"Avg classification conf:   {avg_conf * 100:.1f}%")
    print(f"Avg ambiguity flags:       {avg_flags:.1f}")
    print()
    print("Threshold for 'good enough to route to human':")
    print("  grounding_rate >= 0.85  AND  domain_correct  AND  urgency_correct")

    # Per-document detail for failures or interesting results
    for r in results:
        if not r.passed or r.risk_area_coverage < 1.0 or r.action_item_coverage < 1.0:
            print(f"\n  ⚠  {r.document_name}")
            if not r.domain_correct:
                print(f"     domain:   expected={r.expected_domain!r}  actual={r.actual_domain!r}")
            if not r.urgency_correct:
                print(f"     urgency:  expected={r.expected_urgency!r}  actual={r.actual_urgency!r}")
            if r.grounding_rate < 0.85:
                print(f"     grounding_rate: {r.grounding_rate:.1%} (below 0.85 threshold)")
            if r.risk_area_coverage < 1.0:
                print(f"     risk_area_coverage: {r.risk_area_coverage:.0%}")
            if r.action_item_coverage < 1.0:
                print(f"     action_item_coverage: {r.action_item_coverage:.0%}")
            if r.error:
                print(f"     error: {r.error}")


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------


def save_evaluation_report(
    results: list[EvaluationResult],
    output_path: str = "evaluation_report.json",
) -> None:
    """Save full evaluation results as JSON."""
    payload = {
        "evaluated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_documents": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "avg_grounding_rate": round(
            sum(r.grounding_rate for r in results) / len(results), 4
        ) if results else 0.0,
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nEvaluation report saved to: {output_path}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Compliance Triage Agent quality against ground truth labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audit-dir",
        default="audit_logs",
        help="Directory containing audit log JSON files",
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.yaml",
        help="Path to ground truth YAML file",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Where to save the evaluation report JSON",
    )
    args = parser.parse_args()

    try:
        results = run_evaluation(args.audit_dir, args.ground_truth)
    except FileNotFoundError as exc:
        print(f"[eval] ERROR: {exc}")
        sys.exit(1)

    print_evaluation_report(results)

    if results:
        save_evaluation_report(results, args.output)
