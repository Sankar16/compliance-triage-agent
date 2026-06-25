"""
Compliance Triage Agent — LangGraph Graph

PIPELINE:
                    [ingest_and_chunk]
                           ↓
              fan_out via Send API (parallel)
         ↙         ↙         ↙         ↘
[extract_0] [extract_1] [extract_2] ... [extract_N]
         ↘         ↘         ↘         ↙
                    [run_merge]
                           ↓
                  [run_classification]
                           ↓
                  [run_flag_ambiguities]
                           ↓
                  [run_propose_routing]
                           ↓
              [assemble_triage_result]
                           ↓
                  ⏸ HUMAN CHECKPOINT ⏸
                  (interrupt_before here)
                  status = pending_human_review
                           ↓
                  (human resumes graph)
                           ↓
                   [record_decision]
                           ↓
                          END

HITL ENFORCEMENT:
- Graph compiles with interrupt_before=["record_decision"]
- TriageResult.status defaults to PENDING_HUMAN_REVIEW (schema level)
- No node routes the document — propose_routing() only proposes
- Human decision required to transition status to any other value

PARALLELISM:
- Chunk extraction: Send API fans out one task per chunk, all run
  concurrently, results collected before merge
- classify, flag, propose run sequentially after merge (classify →
  flag → propose) to ensure each step has its dependency available
"""

from __future__ import annotations

import operator
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import anthropic as anthropic_module
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from schemas import (
    AmbiguityFlag,
    Classification,
    DocumentChunk,
    DocumentStatus,
    ExtractedEntities,
    MergedFindings,
    RoutingProposal,
    TriageResult,
)
from tools import (
    chunk_document,
    classify_document,
    extract_compliance_entities,
    flag_ambiguities,
    merge_findings,
    propose_routing,
)
from audit_log import build_audit_entries_from_state, save_triage_result

load_dotenv()
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    # Input
    document_path: str
    document_id: str

    # Populated by ingest_and_chunk
    chunks: list  # list[DocumentChunk] as dicts

    # Populated by parallel extraction — Annotated+operator.add lets
    # concurrent Send nodes append without overwriting each other
    all_entities: Annotated[list, operator.add]

    # Populated by run_merge
    merged: dict  # MergedFindings serialized

    # Populated sequentially: classify → flag → propose
    classification: dict  # Classification serialized
    flags: Annotated[list, operator.add]  # list[AmbiguityFlag] serialized

    # Populated by run_propose_routing
    routing_proposal: dict  # RoutingProposal serialized

    # Final output
    triage_result: dict  # TriageResult serialized

    # Human decision (set when graph is resumed after interrupt)
    human_decision: dict | None

    # Timing: step_name -> duration_ms (populated by sequential nodes)
    timing: dict

    # Path to the saved audit log file
    audit_log_path: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def ingest_and_chunk(state: AgentState) -> dict:
    t0 = time.time()
    chunks = chunk_document(state["document_path"])
    duration_ms = int((time.time() - t0) * 1000)
    print(f"[ingest_and_chunk] {len(chunks)} chunks created")
    return {
        "chunks": [c.model_dump(mode="json") for c in chunks],
        "timing": {**state.get("timing", {}), "ingest_and_chunk": duration_ms},
    }


def fan_out_extraction(state: AgentState) -> list[Send]:
    """Conditional edge: fan one Send per chunk to extract_one_chunk."""
    return [
        Send("extract_one_chunk", {"chunk": c, "document_id": state["document_id"]})
        for c in state["chunks"]
    ]


@retry(
    retry=retry_if_exception_type(anthropic_module.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(4),
    reraise=False,
)
def _extract_with_retry(chunk, client):
    return extract_compliance_entities(chunk, client)


def extract_one_chunk(state: AgentState) -> dict:
    chunk = DocumentChunk(**state["chunk"])
    try:
        entities = _extract_with_retry(chunk, _client)
        print(f"[extract] chunk {chunk.chunk_index} done")
        return {"all_entities": [entities.model_dump(mode="json")]}
    except Exception as exc:
        print(f"[extract] WARNING: chunk {chunk.chunk_index} failed after 4 attempts — skipping")
        return {"all_entities": []}


def run_merge(state: AgentState) -> dict:
    t0 = time.time()
    chunks = [DocumentChunk(**c) for c in state["chunks"]]
    entities = [ExtractedEntities(**e) for e in state["all_entities"]]
    merged = merge_findings(chunks, entities, state["document_id"])
    duration_ms = int((time.time() - t0) * 1000)
    print(f"[merge] {merged.chunks_with_content} chunks had content")
    return {
        "merged": merged.model_dump(mode="json"),
        "timing": {**state.get("timing", {}), "run_merge": duration_ms},
    }


def run_classification(state: AgentState) -> dict:
    t0 = time.time()
    findings = MergedFindings(**state["merged"])
    classification = classify_document(findings, _client)
    duration_ms = int((time.time() - t0) * 1000)
    print(f"[classify] {classification.compliance_domain} / {classification.urgency_level}")
    return {
        "classification": classification.model_dump(mode="json"),
        "timing": {**state.get("timing", {}), "run_classification": duration_ms},
    }


def run_flag_ambiguities(state: AgentState) -> dict:
    t0 = time.time()
    findings = MergedFindings(**state["merged"])
    classification = Classification(**state["classification"])
    flags = flag_ambiguities(findings, classification, _client)
    duration_ms = int((time.time() - t0) * 1000)
    print(f"[flag] {len(flags)} ambiguity flags raised")
    return {
        "flags": [f.model_dump(mode="json") for f in flags],
        "timing": {**state.get("timing", {}), "run_flag_ambiguities": duration_ms},
    }


def run_propose_routing(state: AgentState) -> dict:
    t0 = time.time()
    findings = MergedFindings(**state["merged"])
    classification = Classification(**state["classification"])
    flags = [AmbiguityFlag(**f) for f in state["flags"]]
    proposal = propose_routing(findings, classification, flags, _client)
    duration_ms = int((time.time() - t0) * 1000)
    return {
        "routing_proposal": proposal.model_dump(mode="json"),
        "timing": {**state.get("timing", {}), "run_propose_routing": duration_ms},
    }


def assemble_triage_result(state: AgentState) -> dict:
    timing = state.get("timing", {})
    audit_entries = build_audit_entries_from_state(state, timing)

    result = TriageResult(
        document_id=state["document_id"],
        document_name=Path(state["document_path"]).name,
        ingested_at=datetime.utcnow(),
        all_entities=[ExtractedEntities(**e) for e in state["all_entities"]],
        classification=Classification(**state["classification"]),
        routing_proposal=RoutingProposal(**state["routing_proposal"]),
        ambiguity_flags=[AmbiguityFlag(**f) for f in state["flags"]],
        audit_trail=audit_entries,
        status=DocumentStatus.PENDING_HUMAN_REVIEW,
    )
    print(f"[assemble] TriageResult ready — status: {result.status}, flags: {len(result.ambiguity_flags)}")
    filepath = save_triage_result(result, audit_entries)
    return {"triage_result": result.model_dump(mode="json"), "audit_log_path": filepath}


def record_decision(state: AgentState) -> dict:
    if not state.get("human_decision"):
        raise RuntimeError("record_decision called without human_decision in state")

    decision = state["human_decision"]
    triage = dict(state["triage_result"])
    triage["status"] = decision["action"]
    print(f"[record_decision] status updated to {decision['action']}")

    # Reconstruct HumanDecision with original_proposal filled from state
    from schemas import HumanDecision as HumanDecisionModel
    human_decision_obj = HumanDecisionModel(
        decision_id=decision.get("decision_id", ""),
        document_id=state["document_id"],
        reviewer_id=decision.get("reviewer_id", ""),
        timestamp=decision.get("timestamp", datetime.utcnow().isoformat()),
        action=decision["action"],
        original_proposal=RoutingProposal(**state["routing_proposal"]),
        final_routing=decision.get("final_routing", ""),
        override_reason=decision.get("override_reason"),
    )

    # Rebuild TriageResult with updated status for the resave
    final_result = TriageResult(**{**state["triage_result"], "status": decision["action"]})
    audit_entries = [
        AuditEntry(**e) for e in state["triage_result"].get("audit_trail", [])
    ]
    save_triage_result(final_result, audit_entries, human_decision=human_decision_obj)

    return {"triage_result": triage}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> object:
    builder = StateGraph(AgentState)

    builder.add_node("ingest_and_chunk", ingest_and_chunk)
    builder.add_node("extract_one_chunk", extract_one_chunk)
    builder.add_node("run_merge", run_merge)
    builder.add_node("run_classification", run_classification)
    builder.add_node("run_flag_ambiguities", run_flag_ambiguities)
    builder.add_node("run_propose_routing", run_propose_routing)
    builder.add_node("assemble_triage_result", assemble_triage_result)
    builder.add_node("record_decision", record_decision)

    builder.add_edge(START, "ingest_and_chunk")
    builder.add_conditional_edges("ingest_and_chunk", fan_out_extraction, ["extract_one_chunk"])
    builder.add_edge("extract_one_chunk", "run_merge")
    builder.add_edge("run_merge", "run_classification")
    builder.add_edge("run_classification", "run_flag_ambiguities")
    builder.add_edge("run_flag_ambiguities", "run_propose_routing")
    builder.add_edge("run_propose_routing", "assemble_triage_result")
    builder.add_edge("assemble_triage_result", "record_decision")
    builder.add_edge("record_decision", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["record_decision"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_triage(
    document_path: str,
    document_id: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Run the triage pipeline up to the human checkpoint.

    Returns {"state": ..., "thread_id": ..., "graph": ...} so the caller
    can inspect the triage result and then call resume_with_decision().
    """
    import uuid

    graph = build_graph()

    if document_id is None:
        document_id = str(uuid.uuid4())[:8]
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "document_path": document_path,
        "document_id": document_id,
        "chunks": [],
        "all_entities": [],
        "merged": {},
        "classification": {},
        "flags": [],
        "routing_proposal": {},
        "triage_result": {},
        "human_decision": None,
        "timing": {},
        "audit_log_path": "",
    }

    final_state = graph.invoke(initial_state, config)
    return {"state": final_state, "thread_id": thread_id, "graph": graph}


def resume_with_decision(
    graph: object,
    thread_id: str,
    action: str,
    reviewer_id: str,
    override_reason: str | None = None,
    final_routing: str | None = None,
) -> dict:
    """Resume the graph after human review with a decision.

    action must be one of the DocumentStatus enum values:
    "approved", "overridden", "escalated", "returned".
    """
    import uuid

    config = {"configurable": {"thread_id": thread_id}}

    human_decision = {
        "decision_id": str(uuid.uuid4())[:8],
        "document_id": None,   # filled from state inside record_decision
        "reviewer_id": reviewer_id,
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "original_proposal": None,  # filled from state inside record_decision
        "final_routing": final_routing or "pending",
        "override_reason": override_reason,
    }

    # Inject the decision into the checkpointed state, then resume from the
    # interrupt point (invoke with None — LangGraph picks up from checkpoint)
    graph.update_state(config, {"human_decision": human_decision})
    return graph.invoke(None, config)


# ---------------------------------------------------------------------------
# __main__ — end-to-end demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json

    print("=== Compliance Triage Agent — End-to-End Run ===")
    print("Document: sample_docs/fatf_grey_list.pdf")
    print("Running pipeline up to human checkpoint...\n")

    result = run_triage(
        document_path="sample_docs/fatf_grey_list.pdf",
        document_id="demo-001",
    )

    state = result["state"]
    thread_id = result["thread_id"]
    graph = result["graph"]

    triage = state.get("triage_result", {})
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — AWAITING HUMAN REVIEW")
    print("=" * 60)
    print(f"Document ID:     {triage.get('document_id')}")
    print(f"Status:          {triage.get('status')}")
    print(f"Domain:          {triage.get('classification', {}).get('compliance_domain')}")
    print(f"Urgency:         {triage.get('classification', {}).get('urgency_level')}")
    print(f"Proposed owner:  {triage.get('routing_proposal', {}).get('recommended_owner')}")
    print(f"Routing conf:    {triage.get('routing_proposal', {}).get('confidence')}")
    print(f"Ambiguity flags: {len(triage.get('ambiguity_flags', []))}")

    print("\n[Simulating human reviewer approving the routing...]\n")

    final_state = resume_with_decision(
        graph=graph,
        thread_id=thread_id,
        action="approved",
        reviewer_id="demo-reviewer-001",
        final_routing=triage.get("routing_proposal", {}).get("recommended_owner", "unknown"),
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — HUMAN DECISION RECORDED")
    print("=" * 60)
    final_triage = final_state.get("triage_result", {})
    print(f"Final status: {final_triage.get('status')}")
    print(f"Reviewer:     demo-reviewer-001")
