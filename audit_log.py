"""Compliance audit logging — file-based paper trail for every pipeline run.

Writes JSON files to audit_logs/ (maps to S3/Blob Storage in production —
same structure, different write target). No LLM calls anywhere in this module.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from schemas import AuditEntry, HumanDecision, TriageResult
from prompts import (
    AMBIGUITY_PROMPT_VERSION,
    CLASSIFICATION_PROMPT_VERSION,
    EXTRACTION_PROMPT_VERSION,
    ROUTING_PROMPT_VERSION,
)

PIPELINE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def create_audit_entry(
    document_id: str,
    step_name: str,
    model_used: str,
    prompt_version: str,
    input_token_count: int,
    output_token_count: int,
    duration_ms: int,
    confidence: float | None = None,
) -> AuditEntry:
    """Create a single AuditEntry for one pipeline step."""
    return AuditEntry(
        entry_id=str(uuid.uuid4()),
        document_id=document_id,
        step_name=step_name,
        timestamp=datetime.utcnow(),
        model_used=model_used,
        prompt_version=prompt_version,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        duration_ms=duration_ms,
        confidence=confidence,
    )


def save_triage_result(
    result: TriageResult,
    audit_entries: list[AuditEntry],
    human_decision: HumanDecision | None = None,
    output_dir: str = "audit_logs",
) -> str:
    """Save the complete triage result + audit trail to a JSON file.

    Filename format: {document_id}_{timestamp}.json
    Creates output_dir if it doesn't exist.
    Returns the full path of the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{result.document_id}_{timestamp_str}.json"
    filepath = str(Path(output_dir) / filename)

    payload = {
        "triage_result": result.model_dump(mode="json"),
        "audit_trail": [e.model_dump(mode="json") for e in audit_entries],
        "human_decision": human_decision.model_dump(mode="json") if human_decision else None,
        "saved_at": datetime.utcnow().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"[audit_log] Saved to {filepath}")
    return filepath


def load_triage_result(filepath: str) -> dict:
    """Load a previously saved triage result from disk.

    Returns the raw dict; caller reconstructs Pydantic models if needed.
    Raises FileNotFoundError with a clear message if the file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Audit log not found: {filepath}. "
            "Check the path or use list_audit_logs() to browse available logs."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_audit_logs(
    output_dir: str = "audit_logs",
    document_id: str | None = None,
) -> list[dict]:
    """List all audit log files in output_dir, sorted by saved_at descending.

    If document_id is provided, filter to only that document's logs.
    Returns [] if output_dir doesn't exist or is empty.
    """
    dir_path = Path(output_dir)
    if not dir_path.exists():
        return []

    results = []
    for json_file in dir_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        tr = payload.get("triage_result", {})
        doc_id = tr.get("document_id", "")

        if document_id is not None and doc_id != document_id:
            continue

        classification = tr.get("classification", {})
        routing = tr.get("routing_proposal", {})

        results.append({
            "filepath": str(json_file),
            "document_id": doc_id,
            "saved_at": payload.get("saved_at", ""),
            "status": tr.get("status", ""),
            "domain": classification.get("compliance_domain", ""),
            "urgency": classification.get("urgency_level", ""),
            "proposed_owner": routing.get("recommended_owner", ""),
            "flag_count": len(tr.get("ambiguity_flags", [])),
        })

    results.sort(key=lambda r: r["saved_at"], reverse=True)
    return results


def build_audit_entries_from_state(
    state: dict,
    timing: dict,
) -> list[AuditEntry]:
    """Build AuditEntry list from a completed agent state.

    timing maps step_name -> duration_ms. Token counts are 0 for
    non-LLM steps; extraction timing is 0 because parallel Send nodes
    cannot write back to shared state without race conditions (Phase 1
    known limitation — Phase 2 should aggregate per-chunk timings).
    """
    doc_id = state.get("document_id", "unknown")
    classification = state.get("classification", {})
    routing_proposal = state.get("routing_proposal", {})

    return [
        create_audit_entry(
            document_id=doc_id,
            step_name="ingest_and_chunk",
            model_used="none",
            prompt_version="none",
            input_token_count=0,
            output_token_count=0,
            duration_ms=timing.get("ingest_and_chunk", 0),
        ),
        create_audit_entry(
            document_id=doc_id,
            step_name="extraction",
            model_used="claude-sonnet-4-6",
            prompt_version=EXTRACTION_PROMPT_VERSION,
            input_token_count=0,
            output_token_count=0,
            duration_ms=0,  # parallel Send nodes — not tracked in shared state
        ),
        create_audit_entry(
            document_id=doc_id,
            step_name="merge",
            model_used="none",
            prompt_version="none",
            input_token_count=0,
            output_token_count=0,
            duration_ms=timing.get("run_merge", 0),
        ),
        create_audit_entry(
            document_id=doc_id,
            step_name="classification",
            model_used="claude-haiku-4-5-20251001",
            prompt_version=CLASSIFICATION_PROMPT_VERSION,
            input_token_count=0,
            output_token_count=0,
            duration_ms=timing.get("run_classification", 0),
            confidence=classification.get("confidence"),
        ),
        create_audit_entry(
            document_id=doc_id,
            step_name="flag_ambiguities",
            model_used="claude-haiku-4-5-20251001",
            prompt_version=AMBIGUITY_PROMPT_VERSION,
            input_token_count=0,
            output_token_count=0,
            duration_ms=timing.get("run_flag_ambiguities", 0),
        ),
        create_audit_entry(
            document_id=doc_id,
            step_name="propose_routing",
            model_used="claude-haiku-4-5-20251001",
            prompt_version=ROUTING_PROMPT_VERSION,
            input_token_count=0,
            output_token_count=0,
            duration_ms=timing.get("run_propose_routing", 0),
            confidence=routing_proposal.get("confidence"),
        ),
    ]


# ---------------------------------------------------------------------------
# __main__ — round-trip test (no API calls)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import date
    from schemas import (
        ActionItem,
        AmbiguityFlag,
        AmbiguityType,
        Classification,
        ComplianceDomain,
        DocumentStatus,
        ExtractedEntities,
        GroundedClaim,
        RoutingProposal,
        UrgencyLevel,
    )

    print("=== audit_log.py round-trip test ===\n")

    # --- Mock AuditEntry objects ---
    entry_ingest = create_audit_entry(
        document_id="test-001",
        step_name="ingest_and_chunk",
        model_used="none",
        prompt_version="none",
        input_token_count=0,
        output_token_count=0,
        duration_ms=42,
    )
    entry_extraction = create_audit_entry(
        document_id="test-001",
        step_name="extraction",
        model_used="claude-sonnet-4-6",
        prompt_version=EXTRACTION_PROMPT_VERSION,
        input_token_count=0,
        output_token_count=0,
        duration_ms=0,
    )
    entry_classification = create_audit_entry(
        document_id="test-001",
        step_name="classification",
        model_used="claude-haiku-4-5-20251001",
        prompt_version=CLASSIFICATION_PROMPT_VERSION,
        input_token_count=850,
        output_token_count=120,
        duration_ms=980,
        confidence=0.91,
    )
    audit_entries = [entry_ingest, entry_extraction, entry_classification]

    # --- Mock TriageResult ---
    mock_result = TriageResult(
        document_id="test-001",
        document_name="fatf_grey_list_test.pdf",
        ingested_at=datetime.utcnow(),
        all_entities=[
            ExtractedEntities(
                risk_areas=[
                    GroundedClaim(
                        value="strategic AML deficiencies",
                        source_quote="must remediate its strategic AML deficiencies within 12 months",
                        source_char_range=(89, 150),
                    )
                ],
                action_items=[
                    ActionItem(
                        action="Remediate strategic AML deficiencies",
                        owner_type="National Financial Intelligence Unit",
                        deadline_raw="within 12 months of this statement",
                        deadline_parsed=date(2027, 6, 19),
                        priority=UrgencyLevel.HIGH,
                        source_quote="must remediate its strategic AML deficiencies within 12 months",
                        source_char_range=(89, 150),
                    )
                ],
                responsible_parties=[
                    GroundedClaim(
                        value="National Financial Intelligence Unit",
                        source_quote="National Financial Intelligence Unit serving as the designated responsible authority",
                        source_char_range=(152, 230),
                    )
                ],
                deadlines=[
                    GroundedClaim(
                        value="within 12 months of this statement",
                        source_quote="must remediate its strategic AML deficiencies within 12 months",
                        source_char_range=(89, 150),
                    )
                ],
                source_chunk_index=0,
                extraction_confidence=0.88,
            )
        ],
        classification=Classification(
            compliance_domain=ComplianceDomain.AML,
            urgency_level=UrgencyLevel.HIGH,
            urgency_rationale="Jurisdiction under increased FATF monitoring with a 12-month remediation deadline.",
            confidence=0.91,
            is_cross_domain=False,
            secondary_domains=[],
        ),
        routing_proposal=RoutingProposal(
            recommended_owner="Chief Compliance Officer",
            owner_role="AML Compliance Officer",
            routing_rationale="AML domain with high urgency requires senior oversight.",
            confidence=0.75,
            alternative_owners=["AML Director", "Compliance Manager"],
        ),
        ambiguity_flags=[
            AmbiguityFlag(
                flag_type=AmbiguityType.MISSING_DEADLINE,
                description="Secondary action references follow-up review with no stated timeframe.",
                severity=UrgencyLevel.MEDIUM,
            )
        ],
        audit_trail=audit_entries,
        status=DocumentStatus.PENDING_HUMAN_REVIEW,
    )

    # --- save_triage_result ---
    filepath = save_triage_result(mock_result, audit_entries)

    # --- load_triage_result round-trip ---
    loaded = load_triage_result(filepath)
    assert loaded["triage_result"]["document_id"] == "test-001", "document_id mismatch"
    assert loaded["pipeline_version"] == PIPELINE_VERSION, "pipeline_version mismatch"
    assert len(loaded["audit_trail"]) == 3, "audit_trail length mismatch"
    assert loaded["human_decision"] is None, "expected null human_decision"
    print(f"[round-trip] Loaded back {len(loaded['audit_trail'])} audit entries — OK")

    # --- list_audit_logs ---
    logs = list_audit_logs()
    assert len(logs) >= 1, "expected at least one log entry"
    summary = logs[0]
    print(f"[list_audit_logs] {summary}")

    print("\nAudit log round-trip: PASS")
