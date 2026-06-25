# Compliance Triage Agent — Design Document

## 1. Architecture Overview

The pipeline runs as a linear sequence of stages: PDF ingestion -> chunking ->
entity extraction (per chunk, run in parallel across chunks) -> merge ->
classification -> ambiguity flagging -> routing proposal -> human checkpoint
(LangGraph interrupt).

**The agent never auto-routes.** Every output is `status = pending_human_review`.
This is enforced in two independent places, deliberately redundant with each
other:
- At the schema level: `TriageResult.status` defaults to
  `DocumentStatus.PENDING_HUMAN_REVIEW`, so it is not possible to construct a
  valid `TriageResult` that claims to already be approved or routed.
- Architecturally: no node in the graph executes a routing decision. The
  `routing_proposal` node only *proposes* an owner and rationale; the actual
  status transition (`approved` / `overridden` / `escalated` / `returned`)
  only happens via a `HumanDecision` recorded after a human reviews the
  proposal at the LangGraph interrupt.

**Stack and reasoning:**
- **Claude Sonnet** — 200k context window (fits full regulatory documents
  without aggressive truncation), strong long-document reasoning, cheaper
  input tokens than GPT-4o for the volume of text this pipeline processes.
- **LangGraph** — the state machine model maps directly onto the
  human-in-the-loop checkpoint requirement (an explicit `interrupt` node is a
  first-class concept, not a bolted-on workaround), and it has native
  parallel branch support via the `Send` API for fanning extraction out
  across chunks.
- **LangSmith** — observability and audit trail. Compliance work requires
  being able to reconstruct exactly what the model saw and produced at each
  step; LangSmith gives this without hand-rolling a tracing layer.
- **No RAG in Phase 1** — a single document needs comprehensive analysis, not
  retrieval. Retrieval introduces a real failure mode for compliance work:
  silently missing a relevant passage because it scored low on similarity is
  itself a compliance risk, not just an accuracy nit. RAG is deferred to
  Phase 2, where it becomes genuinely useful for cross-document
  policy-library queries ("has this requirement appeared in any prior
  filing?").

**Implementation status (as of 2026-06-24):** all six tools and `agent.py`
are implemented and validated. Tools: `chunk_document`,
`extract_compliance_entities`, `merge_findings`, `classify_document`,
`flag_ambiguities`, `propose_routing`. Graph: `agent.py` (LangGraph
StateGraph with Send API parallel fan-out, MemorySaver checkpointer,
`interrupt_before=["record_decision"]` HITL checkpoint). End-to-end demo
run confirmed on `fatf_grey_list.pdf`: AML/CRITICAL, Chief Compliance
Officer proposed, human decision recorded. Remaining: `audit_log.py`
(compliance paper trail).

## 2. Schema Design Rationale

- **`GroundedClaim`** — every extracted `risk_area`, `responsible_party`, and
  `deadline` is paired with an exact verbatim `source_quote` plus a character
  offset range, not just a page number. Page-level citation ("this came from
  page 4") is not precise enough for a compliance reviewer to verify a claim
  against the source text quickly. Exact-quote grounding lets a future UI
  highlight the precise supporting text and lets a reviewer confirm or reject
  a claim in seconds instead of re-reading a page.

- **`ActionItem.deadline_raw` vs `deadline_parsed`** — real regulatory
  deadlines are often not expressible as a calendar date. Phrases actually
  observed in the FATF grey-list test document include "within agreed
  timeframes," "as soon as possible," and "all deadlines have now expired."
  Storing only a parsed date would force the model to either hallucinate a
  fake date to satisfy the field or silently drop the deadline entirely.
  Storing both preserves the original regulatory language verbatim (required
  for audit) while still enabling calendar/scheduling use downstream on the
  subset of deadlines that ARE confidently parseable.

- **`AmbiguityFlag.requires_human_judgment`** defaults to `True` and is not
  meant to ever be `False` in practice. This encodes the human-in-the-loop
  constraint into the data model itself, not just into orchestration logic —
  so the constraint is visible to anyone auditing the schema alone, without
  having to trace through the graph code to confirm it.

- **`RoutingProposal.owner_role` alongside `recommended_owner`** — stored
  specifically to survive personnel changes. A named routing target ("Jane
  Doe") goes stale the moment that person changes roles or leaves; a
  functional role ("AML Compliance Officer") does not, and lets a reviewer
  reroute sensibly even if the named individual is no longer correct.

- **`TriageResult.status` defaults to `PENDING_HUMAN_REVIEW`** — restated
  here because it is the single most load-bearing default in the schema: the
  agent cannot construct a valid `TriageResult` that claims to be already
  approved or routed. Any code path that tries to skip this requires
  explicitly overriding a default, which is a much harder mistake to make by
  accident than the reverse.

## 3. Chunking Strategy — Design Decisions and Real Findings

The chunker uses a hybrid approach: structural detection first (markdown-style
headers, all-caps short lines, sparse numbered sections), falling back to
fixed-size token chunking with overlap when no reliable structure is found.
This was chosen because regulatory documents are semi-structured — pure
fixed-size chunking risks splitting an action item from its deadline across a
chunk boundary, which damages extraction quality on exactly the data that
matters most for this pipeline.

Chunk size config: `target_chunk_size = 1500` tokens, `min_chunk_size = 200`
tokens, `overlap_fallback = 150` tokens (applied only on the fixed-size
fallback path). 1500 tokens is large enough to hold a full structural section
in most regulatory documents tested so far, while staying small enough to
avoid the "lost in the middle" effect that degrades extraction quality on very
long contexts. A simple ~4 characters/token approximation is used throughout;
no tokenizer dependency has been added yet.

### Real bugs found through testing against two structurally different real documents

This section is written as a narrative, in the order the bugs were actually
found, because it is genuine evidence of iterative design validation against
real documents — not a hypothetical risk analysis written in advance.

**1. Running header/footer noise**
- *Observed:* Every chunk from the FATF grey-list PDF was polluted with
  browser-print artifacts: timestamps like "6/19/26, 6:42 PM," a repeated
  page title ("Jurisdictions under Increased Monitoring - 13 February 2026"),
  and "N/total" page counters like "15/21."
- *Why it happened:* The document was printed from a browser, which injects
  these artifacts into the text layer of every page.
- *Fix:* Added `strip_running_artifacts()`, which removes any normalized line
  appearing on more than ~40% of pages (and on at least 3 pages), plus
  dedicated regexes for timestamp and "N/total" shapes regardless of
  repetition count. Runs before any chunking, since all downstream char
  offsets must be computed against the cleaned text.
- *Revealed by:* FATF grey-list document. One of these artifacts ("15/21")
  had also been misdetected as a section title, corrupting the Syria section
  boundary — a second-order effect of the same root cause.

**2. Numbered-paragraph over-segmentation**
- *Observed:* `detect_structural_sections()` produced 177 chunks on the FATF
  Stablecoins report, where roughly 15–25 real sections were expected
  (Executive Summary, Background, Threat Actors, Vulnerabilities,
  Recommendations, Annexes, etc.).
- *Why it happened:* The numbered-section regex matched every numbered
  paragraph ("10. The report explores...", "11. The methodology...") as if
  each were a new section header, when in this document numbering marks
  paragraphs, not sections.
- *Fix:* Added a density check — numbered lines are only treated as section
  markers if they make up under 5% of all non-blank lines in the document.
  Above that threshold, numbering is assumed to be paragraph-level and is
  ignored for section detection, falling back to markdown headers / all-caps
  lines / weak title-guess heuristics (and ultimately fixed-size chunking if
  even those are too sparse).
- *Revealed by:* FATF Stablecoins report (42 pages, dense numbered
  paragraphs throughout).

**3. Page-footer pattern artifacts**
- *Observed:* Page-number decorations like "4 |," "| 5," and "36 |" were
  misdetected as section titles in the Stablecoins report.
- *Why it happened:* These differ per page (the number changes), so the
  repetition-based stripping in `strip_running_artifacts()` — which catches
  *identical* repeated strings — could not catch them. This required a
  pattern-based fix rather than a frequency-based one.
- *Fix:* Added a dedicated `page_footer_pattern` regex
  (`^(\d{1,3}\s*\|\s*|\|\s*\d{1,3}\s*)$`) applied alongside the timestamp and
  page-of-total patterns in `strip_running_artifacts()`.
- *Revealed by:* FATF Stablecoins report.

**4. Small-span merging silently destroying real content**
- *Observed:* `merge_small_spans()` initially merged any chunk shorter than
  `min_chunk_size` into its neighbor, regardless of *why* it was short. On
  the FATF grey-list document (~21 distinct country sections, some
  genuinely brief), this silently absorbed 6 real, legitimate country
  entries — Angola, Bulgaria, Kuwait, Monaco, Syria, and one more — into a
  neighboring country's chunk, discarding their `section_title` entirely.
- *Why it happened:* The merge logic treated "small" as synonymous with
  "noise," when in this document a short section is often a real, complete
  entry (a country with a brief FATF statement), not a fragment.
- *Why this was the most dangerous bug found:* It failed silently rather
  than visibly — no error, no crash, no chunk count anomaly that would
  obviously stand out. It would have made those countries' compliance
  content unattributable and unsearchable downstream, which is exactly the
  failure mode a compliance pipeline cannot tolerate.
- *Fix:* Added a `high_confidence` flag to structural detection
  (`detect_structural_sections()` now returns it per candidate — True for
  markdown headers, all-caps lines, and numbered sections when the density
  check allows them; False for the weak "short line + blank + paragraph
  follows" heuristic). `merge_small_spans()` now never merges a
  `high_confidence=True` span, regardless of size.
- *Revealed by:* FATF grey-list document, by manually cross-checking
  detected section titles against the known list of 21 countries on the
  list.

**5. Unicode/accented-character gap**
- *Observed:* "CÔTE D'IVOIRE" was not detected as a section header at all.
- *Why it happened:* `all_caps_pattern`'s character class
  (`[A-Z][A-Z0-9 \-,/&]`) was ASCII-only and did not match the accented `Ô`
  or the curly apostrophe `'` used in the source text.
- *Fix:* Extended the character class to
  `[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 \-,/&'’]`, covering common accented Latin
  capitals and both straight and curly apostrophes.
- *A second, parallel bug this exposed:* the project's own test harness had
  the same class of bug. Its ground-truth country-list comparison did a
  plain ASCII `.lower()` substring match, so even after detection started
  working, `"côte d'ivoire"` (accented, as actually detected) still did not
  match `"Cote d'Ivoire"` (the ASCII reference name) and was reported as
  "missing" despite being correctly found. Fixed with a normalize-before-
  compare helper using `unicodedata.normalize("NFKD", ...)` applied to both
  sides of the comparison.
- *Revealed by:* FATF grey-list document.

**Process lesson:** during this work, an AI-reported code change was at one
point *not* actually applied to the file on disk — the agent narrated the
change as done, but a later `grep` for the expected new identifier returned
zero matches. Going forward, verify file state directly (grep for a
distinctive new identifier and show the output) rather than trusting a "done"
summary on its own. This is now a standing convention — see `CLAUDE.md`.

### Extraction validation across document types

`extract_compliance_entities()` was validated against both sample documents
after the full extraction pipeline was complete:

- **Document 1 (FATF grey list, browser-printed):** full 21-chunk run.
  400/400 quotes verified, 0 rejected, 100% grounding verification rate.
  Six chunks (Angola, Bulgaria, Côte d'Ivoire, Kuwait, Monaco, Syria)
  produced zero extractions with `extraction_confidence=0.1` — correct
  behaviour, as these are the near-empty country stub entries in the
  source document.
- **Document 2 (FATF Stablecoins report, formally typeset):** 3-chunk spot
  check. 4/4 quotes verified on content chunks. Low-content chunks (table
  of contents, abbreviations list) produced `extraction_confidence` of
  0.35–0.45 and minimal useful extraction — appropriate behaviour, since
  these chunks contain no actionable compliance content.
- Full extraction validation on Document 2 is deferred to Phase 2
  alongside `section_title` reliability improvements for formally typeset
  documents (see Section 4).

## 4. Known Limitations / Phase 2 Candidates

- **`section_title` is unreliable on densely-typeset PDFs with footnotes and
  citations.** Observed on the Stablecoins report: detected titles included
  "Blog," a bare URL/file-path fragment, and a bracketed footnote reference —
  none of which are real section names. Chunk boundaries and content remain
  intact and usable for extraction; only the `section_title` label is
  sometimes wrong or meaningless on this document type. Mitigation:
  `detection_confidence=False` chunks' titles are treated as advisory only;
  downstream audit trail relies primarily on `page_start`/`page_end` and
  `char_start`/`char_end`, not `section_title`, for these chunks.

- **`verify_grounding()` is exact-substring match only** — no fuzzy matching
  for minor whitespace/punctuation differences yet. This is flagged as a
  known future improvement in the code where it was written, but has not yet
  been exercised against real LLM output, since no extraction tool exists yet
  at time of writing.

- **Full extraction validation on formally typeset documents (Document 2)
  is scoped to Phase 2.** The 3-chunk spot check confirms extraction works
  correctly on prose content chunks. However, `section_title` metadata is
  unreliable on this document type (detected titles include bare URLs,
  footnote references, and single-word fragments), and low-content chunks
  (table of contents, abbreviations list) produce appropriately low
  confidence scores but minimal useful extraction. A full 21-chunk run on
  this document is deferred until `section_title` reliability improvements
  are tackled in Phase 2.

- **`routing_rules.yaml` is a static file with placeholder owner names.**
  In production this should integrate with the firm's HR/org system to
  validate that named owners still exist and hold the relevant role.
  Routing staleness (owner left the firm) is a known risk documented in
  the failure mode analysis.

- **Confidence floor of 0.30 means the system always produces a routing
  proposal even under maximum uncertainty.** In Phase 2, consider adding
  a hard escalation threshold — if confidence falls below 0.35, skip the
  routing proposal entirely and escalate directly to Chief Compliance
  Officer with a mandatory human decision flag.

- **Rate limiting:** parallel chunk extraction can trigger API rate limits
  (429 errors) when processing documents with many chunks. Implemented
  retry with exponential backoff (tenacity, max 4 attempts, 1–60 second
  wait). In production, consider adding a semaphore to limit concurrent
  extraction calls to 5–10 at a time, reducing rate limit frequency
  without sacrificing parallelism.

- **Extraction timing not captured in audit log:** parallel Send nodes
  cannot write back to shared AgentState without race conditions, so
  `duration_ms=0` is recorded for the extraction step. Phase 2 should
  aggregate per-chunk timings (e.g. by tracking them in `all_entities`
  entries or via a dedicated `Annotated[list, operator.add]` field).

- **Token counts not captured in audit log (Phase 1):** `input_token_count`
  and `output_token_count` are recorded as 0 for all steps. The Anthropic
  SDK returns token counts in the raw API response; wiring these into audit
  entries requires threading the response object out of each tool function
  and up to the audit layer. Deferred to Phase 2 to keep tool signatures
  clean.

- **Evaluation ground truth is manually labeled for 2 documents only.**
  Production evaluation requires 50+ labeled documents to be statistically
  meaningful. The evaluation framework (`eval.py` + `ground_truth.yaml`) is
  designed to scale — add entries to `ground_truth.yaml` as more documents
  are processed and reviewed.

**6. Typographic quote and hyphenation-artifact normalization (follow-up to whitespace normalization)**
- *Observed:* After the whitespace-normalization pass was in place, real LLM
  extraction testing revealed two further classes of rejected quotes that were
  formatting artifacts, not hallucinations: (a) curly/typographic apostrophes
  in PDF source text (e.g. `country’s`) vs. straight apostrophes in LLM
  output (`country's`), and (b) PDF line-wrap hyphenation artifacts
  (e.g. `risk-\nbased`) which after whitespace collapse became `risk- based`,
  not matching an LLM-emitted `risk-based`.
- *Fix:* Extended `normalize_whitespace()` to a four-step pipeline applied to
  both sides before comparison: (1) replace typographic quotes with straight
  equivalents, (2) collapse `hyphen + whitespace` to bare hyphen, (3) collapse
  remaining whitespace runs, (4) strip. Still strict substring match — no
  fuzzy/wording tolerance added.

## 5. Decision Log

| Date | Decision | Reasoning | Alternative(s) rejected |
|---|---|---|---|
| 2026-06-20 | Chose Claude Sonnet over GPT-4o | Larger context window, cheaper input tokens, comparable tool-calling reliability | GPT-4o (viable alternative, slightly more expensive, smaller context window) |
| 2026-06-20 | Deferred RAG to Phase 2 | Single-document comprehensive read benefits from full coverage over retrieval; retrieval failure is a compliance risk | RAG-first design (rejected for Phase 1 scope) |
| 2026-06-20 | Chose LangGraph over raw SDK loop or CrewAI | State machine model matches HITL checkpoint requirement natively, industry-standard for this pattern | Raw SDK (more manual, harder to get HITL interrupt semantics right), CrewAI (built for multi-agent role-play, overkill for this linear pipeline) |
| 2026-06-20 | Chose hybrid structural+fixed-size chunking over pure fixed-size | Semantic coherence matters more for extraction quality on regulatory text | Pure fixed-size (simpler but cuts content arbitrarily), pure semantic/embedding-based chunking (overkill for single-document analysis, adds embedding dependency for marginal gain) |
| 2026-06-24 | Extended `normalize_whitespace()` to cover typographic quote normalization and PDF hyphenation artifacts | Real LLM extraction testing revealed remaining rejections after whitespace normalization; inspected and confirmed both were formatting artifacts not hallucinations; fixed since they were cheap and well-understood | Deferred (would have left known false rejections in production) |
| 2026-06-24 | Did not deduplicate action_items and deadlines in `merge_findings()` | Compliance context: false negatives (missing an obligation) worse than false positives (seeing a similar obligation twice); deduplication risks losing nuance in multi-country regulatory documents | Deduplication (rejected: risks silently dropping distinct obligations that share similar wording) |
| 2026-06-24 | Used focused text summary (not full JSON dump) as input to `classify_document()` | ~400 tokens vs ~4000 tokens per call; classifier only needs urgency signals, deadlines, risk areas, and parties — not full grounding metadata | Full JSON dump (rejected: token cost, context noise, no quality benefit for classification task) |
| 2026-06-24 | Used `claude-haiku-4-5` for `classify_document()` and `flag_ambiguities()` jurisdiction check | Classification and jurisdiction lookup are simpler reasoning tasks than extraction; Haiku is ~10x cheaper and sufficient for these tasks | `claude-sonnet` for all steps (rejected: unnecessary cost for simpler tasks) |
| 2026-06-24 | `flag_ambiguities()` uses mostly pure Python logic with one LLM call only for jurisdiction check | Deterministic conditions (missing deadline, cross-domain, low confidence, contradictory signals) don't need LLM reasoning; only semantic recognition of unknown regulatory bodies requires it | LLM for all checks (rejected: adds cost and non-determinism to checks that can be done reliably with code) |
| 2026-06-24 | Confidence calculation moved from LLM to deterministic Python in `propose_routing()` | LLMs are unreliable at arithmetic; compliance audit trail requires explainable, reproducible scores; formula is `max(0.30, classification.confidence - 0.10 per ambiguity flag)` | LLM-calculated confidence (rejected: not auditable, not reproducible, returned wrong value in testing) |
| 2026-06-24 | Routing rules stored in `routing_rules.yaml`, not hardcoded in code | Routing targets change as staff change; code changes for personnel changes are a maintenance risk; YAML file can be updated without touching agent logic | Hardcoded routing (rejected: brittle, requires code deployment for org changes) |
| 2026-06-24 | `propose_routing()` uses `claude-haiku-4-5` with focused text summary input | Routing is a lookup + rationale task, not complex reasoning; Haiku sufficient and ~10x cheaper than Sonnet | Sonnet for routing (rejected: unnecessary cost for this task complexity) |
| 2026-06-24 | `resume_with_decision()` uses `graph.update_state()` then `invoke(None)` to resume from interrupt | `invoke({"human_decision": ...})` starts a fresh invocation, re-running the full pipeline. The correct LangGraph HITL resume pattern injects state into the checkpoint with `update_state`, then resumes from the interrupt point with `invoke(None)`. Discovered and fixed during end-to-end demo run. | `invoke(partial_state)` (rejected: re-runs entire pipeline, wastes tokens and latency, confirmed broken in testing) |
| 2026-06-24 | Added retry with exponential backoff (tenacity) for chunk extraction | Parallel Send API fan-out triggers rate limits on documents with many chunks; without retry, temporary 429s cause permanent data loss for affected chunks | No retry (rejected: data loss risk unacceptable for compliance pipeline), synchronous extraction (rejected: ~10x slower) |
| 2026-06-24 | Jurisdiction check updated to exclude sovereign nations explicitly | Initial implementation flagged every country name as unrecognized (17 flags on FATF grey list), drowning out genuinely useful flags for unknown private entities; prompt now instructs the LLM that sovereign nations are always recognized | Removing the check entirely (rejected: still useful for non-country entities), relying on the existing "treat as recognized" clause (rejected: LLM was not applying it reliably to country names) |
| 2026-06-25 | Audit log writes JSON to `audit_logs/` (file-based for Phase 1) | Compliance paper trail is required; JSON files are human-readable, regulator-inspectable, and structurally identical to what an S3/Blob write would produce — same payload, different target in production | Database write (rejected: adds infra dependency with no benefit for single-document Phase 1), LangSmith only (rejected: external service, not self-contained for compliance audit) |
| 2026-06-25 | `save_triage_result()` called twice per document — once pre-checkpoint (pending status) and once post-decision (final status) | The pre-checkpoint save ensures a record exists even if the human review step fails or times out; the post-decision save overwrites it with the final status and HumanDecision record | Single save only at end (rejected: no record if pipeline crashes before human review) |
| 2026-06-25 | Moved confidence thresholds to `routing_rules.yaml` | Compliance managers should be able to tune sensitivity without code changes; different firms have different risk tolerances for routing confidence | Hardcoded thresholds (rejected: requires code deployment to adjust firm-specific settings) |
| 2026-06-25 | Evaluation runs on existing audit logs, not re-running the pipeline | Re-running would cost API tokens and introduce variability; audit logs capture the exact output to evaluate — evaluation is free if audit logs exist | Re-run evaluation (rejected: expensive, non-deterministic, would make evaluation a luxury rather than a routine check) |
