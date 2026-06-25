"""Pydantic schemas for the Compliance Triage Agent pipeline.

These models define the data contracts passed between LangGraph nodes:
document ingestion -> entity extraction -> classification -> routing ->
ambiguity detection -> human review -> audit logging.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UrgencyLevel(str, Enum):
    """Severity/urgency tiers used for classification, ambiguity, and audit severity.

    - critical: Immediate regulatory or financial exposure; requires same-day action.
    - high: Material risk with a near-term deadline; requires action within days.
    - medium: Notable risk but not time-critical; should be addressed in the normal cycle.
    - low: Informational or minor risk; no urgent action required.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceDomain(str, Enum):
    """Regulatory/compliance domain a document or action item belongs to."""

    AML = "aml"
    DATA_PRIVACY = "data_privacy"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    KYC = "kyc"
    SANCTIONS = "sanctions"
    REPORTING = "reporting"
    OPERATIONAL_RISK = "operational_risk"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Lifecycle status of a document as it moves through human review."""

    PENDING_HUMAN_REVIEW = "pending_human_review"
    APPROVED = "approved"
    OVERRIDDEN = "overridden"
    ESCALATED = "escalated"
    RETURNED = "returned"


class AmbiguityType(str, Enum):
    """Reasons a document may be flagged as requiring human judgment."""

    MISSING_DEADLINE = "missing_deadline"
    AMBIGUOUS_OWNER = "ambiguous_owner"
    CROSS_DOMAIN = "cross_domain"
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTORY_SIGNALS = "contradictory_signals"
    UNRECOGNIZED_JURISDICTION = "unrecognized_jurisdiction"


# ---------------------------------------------------------------------------
# Document ingestion / chunking model
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A structurally or fixed-size chunked segment of a source document,
    preserving exact location for grounding and audit purposes."""

    chunk_index: int = Field(description="Sequential index of this chunk within the document.")
    text: str = Field(description="The raw text content of this chunk.")
    char_start: int = Field(description="Character offset where this chunk begins in the original full document text.")
    char_end: int = Field(description="Character offset where this chunk ends in the original full document text.")
    page_start: int = Field(description="First PDF page number this chunk overlaps.")
    page_end: int = Field(description="Last PDF page number this chunk overlaps.")
    token_count: int = Field(description="Approximate token count of this chunk's text.")
    chunking_method: str = Field(description="How this chunk was produced: 'structural' or 'fixed_size_fallback'.")
    section_title: Optional[str] = Field(
        default=None, description="Detected section heading for this chunk, if chunking_method is 'structural'."
    )
    detection_confidence: Optional[bool] = Field(
        default=None,
        description="Whether this chunk's section was detected via a high-confidence structural heuristic "
        "(markdown header, all-caps line, or numbered section). None for fixed_size_fallback chunks.",
    )


# ---------------------------------------------------------------------------
# Core extraction models
# ---------------------------------------------------------------------------


class ActionItem(BaseModel):
    """A single actionable task extracted from a document."""

    action: str = Field(description="The action that needs to be taken, in plain language.")
    owner_type: str = Field(
        description="The type/category of party responsible for this action (e.g. 'legal team', 'compliance officer', 'business unit')."
    )
    deadline_raw: str = Field(
        description="The exact deadline language as it appeared in the source document, preserved verbatim."
    )
    deadline_parsed: Optional[date] = Field(
        default=None,
        description="The deadline normalized to a calendar date, if it could be confidently parsed from deadline_raw.",
    )
    priority: UrgencyLevel = Field(description="The urgency level assigned to this specific action item.")
    source_quote: str = Field(
        description="The exact, verbatim sentence or phrase from the source document that supports this action item. Must be an exact substring match of the source text, not a paraphrase."
    )
    source_char_range: tuple[int, int] = Field(
        description="Character offset range (start, end) in the original document text where source_quote appears."
    )


class GroundedClaim(BaseModel):
    """A single extracted claim paired with its exact source grounding."""

    value: str = Field(description="The extracted value (e.g. the risk area name, party name, or deadline phrase).")
    source_quote: str = Field(description="The exact verbatim sentence from the source supporting this claim.")
    source_char_range: tuple[int, int] = Field(
        description="Character offset range (start, end) where source_quote appears in the original document."
    )


class ExtractedEntities(BaseModel):
    """Structured entities extracted from a single chunk of a source document."""

    risk_areas: list[GroundedClaim] = Field(
        description="Risk areas or topics identified in this chunk (e.g. 'transaction monitoring gaps')."
    )
    action_items: list[ActionItem] = Field(
        description="Discrete action items identified in this chunk."
    )
    responsible_parties: list[GroundedClaim] = Field(
        description="Named individuals, roles, or departments mentioned as responsible parties."
    )
    deadlines: list[GroundedClaim] = Field(
        description="Raw deadline phrases found in this chunk, independent of any specific action item."
    )
    source_chunk_index: int = Field(
        description="Index of the source document chunk this extraction was derived from."
    )
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in the accuracy/completeness of this extraction, from 0 to 1.",
    )


# ---------------------------------------------------------------------------
# Merged findings model
# ---------------------------------------------------------------------------


class MergedFindings(BaseModel):
    """Consolidated extraction results across all chunks of a document,
    ready for classification, ambiguity detection, and routing proposal.
    This is produced by merge_findings() and consumed by all downstream
    LLM-calling tools."""

    document_id: str = Field(description="Unique identifier for the source document.")

    all_risk_areas: list[GroundedClaim] = Field(
        description="Deduplicated risk areas across all chunks."
    )
    all_action_items: list[ActionItem] = Field(
        description="All action items across all chunks, preserving source attribution."
    )
    all_responsible_parties: list[GroundedClaim] = Field(
        description="Deduplicated responsible parties across all chunks."
    )
    all_deadlines: list[GroundedClaim] = Field(
        description="All deadlines across all chunks, preserving exact language."
    )

    chunks_processed: int = Field(
        description="Total number of chunks the document was split into."
    )
    chunks_with_content: int = Field(
        description="Number of chunks that returned at least one verified extraction."
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Average extraction_confidence across chunks that had content."
    )

    dominant_urgency_signals: list[str] = Field(
        description=(
            "Urgency-related phrases extracted verbatim from the document "
            "(e.g. 'all deadlines have now expired', 'without delay', "
            "'as soon as possible'). Collected from deadline values and "
            "action item deadline_raw fields. Used by classify_document() "
            "to determine urgency level."
        )
    )

    source_chunk_indices: list[int] = Field(
        description="Indices of chunks that contributed content to this merge."
    )


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------


class Classification(BaseModel):
    """The compliance domain and urgency classification for a document."""

    compliance_domain: ComplianceDomain = Field(
        description="The primary compliance domain this document falls under."
    )
    urgency_level: UrgencyLevel = Field(description="The overall urgency level assigned to the document.")
    urgency_rationale: str = Field(
        description="Explanation of why this urgency level was assigned, citing evidence from the document."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's confidence in this classification, from 0 to 1."
    )
    is_cross_domain: bool = Field(
        description="Whether this document materially touches more than one compliance domain."
    )
    secondary_domains: list[ComplianceDomain] = Field(
        description="Additional compliance domains relevant to this document, if any."
    )


# ---------------------------------------------------------------------------
# Routing model
# ---------------------------------------------------------------------------


class RoutingProposal(BaseModel):
    """A proposed routing decision for who should own/handle this document."""

    recommended_owner: str = Field(
        description="The specific person, team, or role recommended as the primary owner."
    )
    owner_role: str = Field(
        description="The functional role of the recommended owner (e.g. 'AML Compliance Officer')."
    )
    routing_rationale: str = Field(
        description="Explanation of why this owner was recommended, based on the document's content."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's confidence in this routing recommendation, from 0 to 1."
    )
    alternative_owners: list[str] = Field(
        description="Other plausible owners considered as alternatives to the primary recommendation."
    )


# ---------------------------------------------------------------------------
# Ambiguity model
# ---------------------------------------------------------------------------


class AmbiguityFlag(BaseModel):
    """A flag raised when the pipeline cannot confidently resolve some aspect of a document."""

    flag_type: AmbiguityType = Field(description="The category of ambiguity detected.")
    description: str = Field(
        description="Human-readable description of the specific ambiguity encountered."
    )
    severity: UrgencyLevel = Field(
        description="How severe this ambiguity is in terms of risk if left unresolved."
    )
    requires_human_judgment: bool = Field(
        default=True,
        description="Whether this ambiguity requires a human reviewer to resolve. Always True by design.",
    )


# ---------------------------------------------------------------------------
# Audit models
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single audit log record capturing one automated pipeline step for traceability."""

    entry_id: str = Field(description="Unique identifier for this audit entry.")
    document_id: str = Field(description="Identifier of the document this entry pertains to.")
    step_name: str = Field(
        description="Name of the pipeline step that produced this entry (e.g. 'extraction', 'classification')."
    )
    timestamp: datetime = Field(description="UTC timestamp when this step executed.")
    model_used: str = Field(description="Identifier of the LLM model used for this step.")
    prompt_version: str = Field(description="Version identifier of the prompt template used for this step.")
    input_token_count: int = Field(description="Number of input tokens consumed by this step.")
    output_token_count: int = Field(description="Number of output tokens generated by this step.")
    duration_ms: int = Field(description="Wall-clock duration of this step in milliseconds.")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score produced by this step, if applicable.",
    )


class HumanDecision(BaseModel):
    """A record of a human reviewer's decision on a document, including any override."""

    decision_id: str = Field(description="Unique identifier for this human decision record.")
    document_id: str = Field(description="Identifier of the document this decision pertains to.")
    reviewer_id: str = Field(description="Identifier of the human reviewer who made this decision.")
    timestamp: datetime = Field(description="UTC timestamp when the decision was made.")
    action: DocumentStatus = Field(description="The resulting document status after this decision.")
    original_proposal: RoutingProposal = Field(
        description="The system-generated routing proposal that the reviewer evaluated."
    )
    final_routing: str = Field(description="The final routing destination chosen by the reviewer.")
    override_reason: Optional[str] = Field(
        default=None,
        description="Reviewer's explanation for overriding the original proposal, if applicable.",
    )


# ---------------------------------------------------------------------------
# Master output model
# ---------------------------------------------------------------------------


class TriageResult(BaseModel):
    """The complete output of the triage pipeline for a single document, combining
    extraction, classification, routing, ambiguity flags, and the full audit trail.
    """

    document_id: str = Field(description="Unique identifier for the source document.")
    document_name: str = Field(description="Original filename or title of the source document.")
    ingested_at: datetime = Field(description="UTC timestamp when the document was ingested into the pipeline.")
    all_entities: list[ExtractedEntities] = Field(
        description="Extracted entities from each chunk of the source document."
    )
    classification: Classification = Field(description="The compliance domain and urgency classification.")
    routing_proposal: RoutingProposal = Field(description="The proposed routing decision for this document.")
    ambiguity_flags: list[AmbiguityFlag] = Field(
        description="Ambiguities detected during processing that require human judgment."
    )
    audit_trail: list[AuditEntry] = Field(
        description="Chronological log of all automated pipeline steps performed on this document."
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING_HUMAN_REVIEW,
        description="Current lifecycle status of this document in the review process.",
    )


if __name__ == "__main__":
    dummy_source_text = (
        "The FATF identified the Republic of Numeria as a jurisdiction under increased "
        "monitoring. Numeria must remediate its strategic AML deficiencies within 12 months "
        "of this statement, with the National Financial Intelligence Unit serving as the "
        "designated responsible authority."
    )

    dummy_action_item = ActionItem(
        action="Remediate strategic AML deficiencies identified by the FATF.",
        owner_type="National Financial Intelligence Unit",
        deadline_raw="within 12 months of this statement",
        deadline_parsed=date(2027, 6, 19),
        priority=UrgencyLevel.HIGH,
        source_quote="Numeria must remediate its strategic AML deficiencies within 12 months "
        "of this statement, with the National Financial Intelligence Unit serving as the "
        "designated responsible authority.",
        source_char_range=(89, 268),
    )

    dummy_risk_area = GroundedClaim(
        value="strategic AML deficiencies",
        source_quote="Numeria must remediate its strategic AML deficiencies within 12 months "
        "of this statement.",
        source_char_range=(89, 188),
    )

    dummy_responsible_party = GroundedClaim(
        value="National Financial Intelligence Unit",
        source_quote="the National Financial Intelligence Unit serving as the designated "
        "responsible authority.",
        source_char_range=(190, 268),
    )

    dummy_deadline = GroundedClaim(
        value="within 12 months of this statement",
        source_quote="Numeria must remediate its strategic AML deficiencies within 12 months "
        "of this statement.",
        source_char_range=(89, 188),
    )

    dummy_entities = ExtractedEntities(
        risk_areas=[dummy_risk_area],
        action_items=[dummy_action_item],
        responsible_parties=[dummy_responsible_party],
        deadlines=[dummy_deadline],
        source_chunk_index=0,
        extraction_confidence=0.87,
    )

    dummy_classification = Classification(
        compliance_domain=ComplianceDomain.AML,
        urgency_level=UrgencyLevel.HIGH,
        urgency_rationale="Document places a jurisdiction under increased FATF monitoring with a defined remediation deadline.",
        confidence=0.91,
        is_cross_domain=False,
        secondary_domains=[],
    )

    dummy_routing = RoutingProposal(
        recommended_owner="Jane Doe",
        owner_role="AML Compliance Officer",
        routing_rationale="Subject matter and remediation requirement fall under this role's mandate.",
        confidence=0.85,
        alternative_owners=["Compliance Director"],
    )

    dummy_ambiguity_flag = AmbiguityFlag(
        flag_type=AmbiguityType.MISSING_DEADLINE,
        description="Secondary action item references a follow-up review with no stated timeframe.",
        severity=UrgencyLevel.MEDIUM,
    )

    dummy_audit_entry = AuditEntry(
        entry_id="audit-0001",
        document_id="doc-0001",
        step_name="classification",
        timestamp=datetime(2026, 6, 18, 12, 0, 0),
        model_used="claude-sonnet-4-6",
        prompt_version="v1",
        input_token_count=1200,
        output_token_count=150,
        duration_ms=850,
        confidence=0.91,
    )

    dummy_result = TriageResult(
        document_id="doc-0001",
        document_name="fatf_grey_list_numeria.pdf",
        ingested_at=datetime(2026, 6, 18, 11, 55, 0),
        all_entities=[dummy_entities],
        classification=dummy_classification,
        routing_proposal=dummy_routing,
        ambiguity_flags=[dummy_ambiguity_flag],
        audit_trail=[dummy_audit_entry],
    )

    print(json.dumps(dummy_result.model_dump(mode="json"), indent=2))
