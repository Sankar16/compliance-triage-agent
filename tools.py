"""Document ingestion and chunking tools for the Compliance Triage Agent pipeline.

This module turns a source PDF into a list of grounded DocumentChunk objects,
preserving exact character offsets and page numbers so later extraction steps
can cite verbatim quotes back to their location in the original document.
"""

from __future__ import annotations

import re
import unicodedata

import anthropic
import fitz  # PyMuPDF
from pydantic import ValidationError

from prompts import CLASSIFICATION_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT, JURISDICTION_CHECK_PROMPT
from schemas import (
    ActionItem,
    AmbiguityFlag,
    AmbiguityType,
    Classification,
    ComplianceDomain,
    DocumentChunk,
    ExtractedEntities,
    GroundedClaim,
    MergedFindings,
    UrgencyLevel,
)

CHUNKING_CONFIG = {
    "target_chunk_size": 1500,
    "min_chunk_size": 200,
    "overlap_fallback": 150,
    "strategy": "structural_with_fixed_fallback",
}


def extract_text_with_pages(pdf_path: str) -> tuple[str, list[tuple[int, int, int]]]:
    """Extract all text from a PDF as one continuous string, plus a mapping
    of (page_number, char_start, char_end) so any character offset in the
    full text can be traced back to the PDF page it came from.
    """
    doc = fitz.open(pdf_path)
    full_text_parts: list[str] = []
    page_map: list[tuple[int, int, int]] = []
    cursor = 0

    for page_number in range(len(doc)):
        page_text = doc[page_number].get_text()
        char_start = cursor
        full_text_parts.append(page_text)
        cursor += len(page_text)
        char_end = cursor
        page_map.append((page_number + 1, char_start, char_end))

    doc.close()
    return "".join(full_text_parts), page_map


def strip_running_artifacts(
    full_text: str, page_map: list[tuple[int, int, int]]
) -> tuple[str, list[tuple[int, int, int]]]:
    """Detect and remove text lines that repeat near-identically across many
    pages (running headers/footers, browser-print timestamps, 'page X/Y'
    artifacts).

    Must run before detect_structural_sections() and fixed_size_chunk(),
    since downstream char offsets depend on the cleaned text being the new
    ground truth for offset calculations.
    """
    timestamp_pattern = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?$", re.IGNORECASE
    )
    page_of_total_pattern = re.compile(r"^\d{1,3}\s*/\s*\d{1,3}$")
    page_footer_pattern = re.compile(r"^(\d{1,3}\s*\|\s*|\|\s*\d{1,3}\s*)$")

    lines = full_text.splitlines(keepends=True)

    def page_for_offset(offset: int) -> int:
        for page_number, p_start, p_end in page_map:
            if p_start <= offset < p_end:
                return page_number
        return page_map[-1][0] if page_map else 1

    line_records: list[tuple[str, str, int, int]] = []  # (raw_line, normalized, char_start, page)
    cursor = 0
    for line in lines:
        normalized = line.strip().lower()
        line_records.append((line, normalized, cursor, page_for_offset(cursor)))
        cursor += len(line)

    total_pages = max((p for _, _, _, p in line_records), default=1)

    pages_seen_by_norm: dict[str, set[int]] = {}
    for _, normalized, _, page in line_records:
        if not normalized:
            continue
        pages_seen_by_norm.setdefault(normalized, set()).add(page)

    repeated_norms = {
        norm
        for norm, pages in pages_seen_by_norm.items()
        if len(pages) >= 3 and (len(pages) / total_pages) > 0.4
    }

    kept_lines: list[str] = []
    for raw_line, normalized, _char_start, _page in line_records:
        stripped = raw_line.strip()
        if normalized in repeated_norms:
            continue
        if stripped and (
            timestamp_pattern.match(stripped)
            or page_of_total_pattern.match(stripped)
            or page_footer_pattern.match(stripped)
        ):
            continue
        kept_lines.append(raw_line)

    cleaned_text = "".join(kept_lines)

    # Rebuild page_map char offsets against the cleaned text by re-walking
    # the kept lines and re-using each line's original page number.
    kept_with_pages: list[tuple[str, int]] = []
    cursor = 0
    for raw_line, normalized, _char_start, page in line_records:
        stripped = raw_line.strip()
        if normalized in repeated_norms:
            continue
        if stripped and (
            timestamp_pattern.match(stripped)
            or page_of_total_pattern.match(stripped)
            or page_footer_pattern.match(stripped)
        ):
            continue
        kept_with_pages.append((raw_line, page))

    updated_page_map: list[tuple[int, int, int]] = []
    cursor = 0
    current_page = None
    page_start = 0
    for raw_line, page in kept_with_pages:
        if current_page is None:
            current_page = page
            page_start = cursor
        elif page != current_page:
            updated_page_map.append((current_page, page_start, cursor))
            current_page = page
            page_start = cursor
        cursor += len(raw_line)
    if current_page is not None:
        updated_page_map.append((current_page, page_start, cursor))

    if not updated_page_map:
        updated_page_map = [(1, 0, len(cleaned_text))]

    return cleaned_text, updated_page_map


def detect_structural_sections(full_text: str) -> list[tuple[str, int, int, bool]]:
    """Detect likely section boundaries using heuristics (no LLM call).

    Looks for markdown-style headers, all-caps short lines, numbered
    sections, and title-like lines followed by a blank line and a paragraph.
    Returns an empty list if fewer than 2 sections are found, signaling
    that the caller should fall back to fixed-size chunking.

    Each returned tuple is (section_title, char_start, char_end, high_confidence).
    high_confidence is True for markdown headers, all-caps lines, and numbered
    sections (when the density check allows treating them as sections); it is
    False for the weak "short line + blank + paragraph follows" heuristic.
    """
    lines = full_text.splitlines(keepends=True)
    line_offsets: list[tuple[int, int]] = []
    cursor = 0
    for line in lines:
        line_offsets.append((cursor, cursor + len(line)))
        cursor += len(line)

    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")
    numbered_pattern = re.compile(r"^(?:Section\s+\d+|Article\s+\d+|\d+\.)\s*[:.]?\s*(.+)$", re.IGNORECASE)
    all_caps_pattern = re.compile(r"^[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 \-,/&'’]{3,80}$")

    non_blank_lines = [line.strip() for line in lines if line.strip()]
    numbered_line_count = sum(1 for line in non_blank_lines if numbered_pattern.match(line))
    numbered_density = (numbered_line_count / len(non_blank_lines)) if non_blank_lines else 0.0
    treat_numbered_as_sections = numbered_density <= 0.05

    candidates: list[tuple[str, int, bool]] = []  # (title, char_start, high_confidence)

    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue

        char_start = line_offsets[idx][0]

        header_match = header_pattern.match(stripped)
        if header_match:
            candidates.append((header_match.group(2).strip(), char_start, True))
            continue

        numbered_match = numbered_pattern.match(stripped)
        if treat_numbered_as_sections and numbered_match and len(stripped) < 120:
            candidates.append((stripped, char_start, True))
            continue

        if all_caps_pattern.match(stripped) and len(stripped.split()) <= 12:
            candidates.append((stripped, char_start, True))
            continue

        is_short_title = len(stripped) < 80 and not stripped.endswith((".", ",", ";", ":"))
        has_blank_after = idx + 1 < len(lines) and not lines[idx + 1].strip()
        has_paragraph_after = idx + 2 < len(lines) and lines[idx + 2].strip()
        if is_short_title and has_blank_after and has_paragraph_after:
            candidates.append((stripped, char_start, False))
            continue

    if len(candidates) < 2:
        return []

    sections: list[tuple[str, int, int, bool]] = []
    for i, (title, char_start, high_confidence) in enumerate(candidates):
        char_end = candidates[i + 1][1] if i + 1 < len(candidates) else len(full_text)
        sections.append((title, char_start, char_end, high_confidence))

    return sections


def fixed_size_chunk(text: str, start_offset: int, target_size: int, overlap: int) -> list[tuple[int, int]]:
    """Chunk `text` by approximate token count (~4 chars/token) with overlap.

    Returns (char_start, char_end) tuples relative to the original full
    document, using `start_offset` to translate local offsets.
    """
    chars_per_token = 4
    target_chars = target_size * chars_per_token
    overlap_chars = overlap * chars_per_token

    if len(text) <= target_chars:
        return [(start_offset, start_offset + len(text))]

    chunks: list[tuple[int, int]] = []
    pos = 0
    text_len = len(text)
    while pos < text_len:
        end = min(pos + target_chars, text_len)
        chunks.append((start_offset + pos, start_offset + end))
        if end >= text_len:
            break
        pos = end - overlap_chars
        if pos < 0:
            pos = end

    return chunks


def merge_small_spans(
    spans: list[tuple[int, int, str, str | None, bool]], full_text: str, min_chars: int
) -> list[tuple[int, int, str, str | None, bool]]:
    """Merge consecutive spans where a span's text is shorter than min_chars
    into the NEXT span (extending that next span's start backward to include
    the small one), preserving the next span's section_title, method, and
    high_confidence flag. If the small span is the LAST one in the list, merge
    it into the PREVIOUS span instead. Never produce a span longer than ~3x
    target_chunk_size*4 chars as a result of merging — if a merge would exceed
    that, leave the small span standalone rather than merging.

    A span is NEVER merged (regardless of size) if its high_confidence flag is
    True — those represent confident structural detections (e.g. a country's
    FATF entry) and silently absorbing them into a neighbor would discard
    real, attributable content. Only low-confidence or fixed_size_fallback
    spans are eligible to be merged away.
    """
    if not spans:
        return spans

    max_merged_chars = CHUNKING_CONFIG["target_chunk_size"] * 4 * 3

    merged = list(spans)
    i = 0
    while i < len(merged):
        char_start, char_end, method, section_title, high_confidence = merged[i]
        span_len = char_end - char_start
        if high_confidence or span_len >= min_chars or len(merged) == 1:
            i += 1
            continue

        if i + 1 < len(merged):
            next_start, next_end, next_method, next_title, next_confidence = merged[i + 1]
            combined_len = next_end - char_start
            if combined_len <= max_merged_chars:
                merged[i + 1] = (char_start, next_end, next_method, next_title, next_confidence)
                del merged[i]
                continue
        if i > 0:
            prev_start, prev_end, prev_method, prev_title, prev_confidence = merged[i - 1]
            combined_len = char_end - prev_start
            if combined_len <= max_merged_chars:
                merged[i - 1] = (prev_start, char_end, prev_method, prev_title, prev_confidence)
                del merged[i]
                continue

        i += 1

    return merged


def chunk_document(pdf_path: str) -> list[DocumentChunk]:
    """Orchestrate chunking: structural sections where detectable, falling
    back to fixed-size chunking for the whole document or for any
    oversized section.
    """
    full_text, page_map = extract_text_with_pages(pdf_path)
    full_text, page_map = strip_running_artifacts(full_text, page_map)
    sections = detect_structural_sections(full_text)
    target_size = CHUNKING_CONFIG["target_chunk_size"]
    overlap = CHUNKING_CONFIG["overlap_fallback"]

    def pages_for_range(char_start: int, char_end: int) -> tuple[int, int]:
        overlapping_pages = [
            page_number
            for page_number, p_start, p_end in page_map
            if p_start < char_end and p_end > char_start
        ]
        if not overlapping_pages:
            return 1, 1
        return min(overlapping_pages), max(overlapping_pages)

    # (char_start, char_end, method, section_title, high_confidence)
    spans: list[tuple[int, int, str, str | None, bool]] = []

    if sections:
        for title, sec_start, sec_end, high_confidence in sections:
            sec_text = full_text[sec_start:sec_end]
            if len(sec_text) <= target_size * 4:
                spans.append((sec_start, sec_end, "structural", title, high_confidence))
            else:
                sub_ranges = fixed_size_chunk(sec_text, sec_start, target_size, overlap)
                for sub_start, sub_end in sub_ranges:
                    spans.append((sub_start, sub_end, "fixed_size_fallback", title, False))
    else:
        full_ranges = fixed_size_chunk(full_text, 0, target_size, overlap)
        for sub_start, sub_end in full_ranges:
            spans.append((sub_start, sub_end, "fixed_size_fallback", None, False))

    min_chars = CHUNKING_CONFIG["min_chunk_size"] * 4
    spans = merge_small_spans(spans, full_text, min_chars)

    chunks: list[DocumentChunk] = []
    for idx, (char_start, char_end, method, section_title, high_confidence) in enumerate(spans):
        chunk_text = full_text[char_start:char_end]
        page_start, page_end = pages_for_range(char_start, char_end)
        chunks.append(
            DocumentChunk(
                chunk_index=idx,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                page_start=page_start,
                page_end=page_end,
                token_count=len(chunk_text) // 4,
                chunking_method=method,
                section_title=section_title,
                detection_confidence=high_confidence if method == "structural" else None,
            )
        )

    return chunks


def normalize_whitespace(text: str) -> str:
    """Normalize text for grounding comparison.

    Pipeline (order matters):
    1. Replace typographic/curly quotes with straight equivalents so LLM
       output and PDF-extracted source text agree on apostrophe/quote style.
    2. Collapse PDF hyphenation artifacts (hyphen + whitespace, e.g.
       "risk-\\nbased") to a bare hyphen, re-joining broken words as an LLM
       would naturally write them.
    3. Collapse all remaining whitespace runs to a single space.
    4. Strip leading/trailing whitespace.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"-\s+", "-", text)
    return re.sub(r"\s+", " ", text).strip()


if False:
    assert normalize_whitespace("country’s risks") == normalize_whitespace("country's risks")
    assert normalize_whitespace("risk-\nbased approach") == normalize_whitespace("risk-based approach")
    assert normalize_whitespace("  multiple   spaces  ") == "multiple spaces"


def verify_grounding(quote: str, source_text: str) -> bool:
    """Check whether `quote` is an exact substring of `source_text`, after
    whitespace-normalizing both sides.

    This stays a strict substring check — not fuzzy matching. PDF text
    extraction introduces mid-sentence line breaks from visual page
    wrapping, and an LLM naturally reproduces a quote without those
    line breaks; normalizing whitespace on both sides accounts for that
    without tolerating any actual wording/character differences.
    """
    return normalize_whitespace(quote) in normalize_whitespace(source_text)


def find_quote_position(quote: str, source_text: str) -> tuple[int, int] | None:
    """Locate the real (start, end) character offsets of `quote` within the
    ORIGINAL, non-normalized `source_text`, tolerating whitespace
    differences (e.g. PDF line-wrap breaks) between the two.

    Builds a regex from the normalized quote where each whitespace run
    becomes `\\s+` and each straight apostrophe/double-quote becomes a
    character class matching both straight and typographic variants —
    because verify_grounding() normalizes both sides before comparing, but
    we must search the ORIGINAL source_text (which may still contain curly
    quotes) to recover the correct character offsets.

    Returns None if no match is found; caller must treat this the same as
    a failed verification — log and drop, never fabricate an offset.
    """
    normalized_quote = normalize_whitespace(quote)
    if not normalized_quote:
        return None

    def _token_pattern(word: str) -> str:
        escaped = re.escape(word)
        escaped = escaped.replace("'", "['\\u2018\\u2019]")
        escaped = escaped.replace('"', '["\\u201c\\u201d]')
        escaped = escaped.replace('\\-', '\\-\\s*')
        return escaped

    pattern = r"\s+".join(_token_pattern(word) for word in normalized_quote.split(" "))
    match = re.search(pattern, source_text)
    if match is None:
        return None
    return match.start(), match.end()


# ---------------------------------------------------------------------------
# Entity extraction (first LLM-calling tool)
# ---------------------------------------------------------------------------

_GROUNDED_CLAIM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {
            "type": "string",
            "description": "The extracted value (e.g. the risk area name, party name, or deadline phrase).",
        },
        "source_quote": {
            "type": "string",
            "description": "The exact, verbatim substring copied from the provided chunk text that supports this claim.",
        },
    },
    "required": ["value", "source_quote"],
}

_ACTION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "The action that needs to be taken, in plain language."},
        "owner_type": {
            "type": "string",
            "description": "The type/category of party responsible for this action (e.g. 'legal team', 'compliance officer', 'business unit').",
        },
        "deadline_raw": {
            "type": "string",
            "description": "The exact deadline language as it appeared in the source document, preserved verbatim.",
        },
        "deadline_parsed": {
            "type": "string",
            "description": "The deadline normalized to an ISO 8601 calendar date (YYYY-MM-DD), only if confidently parseable from deadline_raw. Omit if not.",
        },
        "priority": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low"],
            "description": "The urgency level assigned to this specific action item.",
        },
        "source_quote": {
            "type": "string",
            "description": "The exact, verbatim sentence or phrase from the chunk text that supports this action item.",
        },
    },
    "required": ["action", "owner_type", "deadline_raw", "priority", "source_quote"],
}

EXTRACTED_ENTITIES_TOOL_SCHEMA = {
    "name": "extract_compliance_entities",
    "description": "Extract structured compliance entities (risk areas, action items, responsible parties, deadlines) from a chunk of a regulatory document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_areas": {"type": "array", "items": _GROUNDED_CLAIM_ITEM_SCHEMA},
            "action_items": {"type": "array", "items": _ACTION_ITEM_SCHEMA},
            "responsible_parties": {"type": "array", "items": _GROUNDED_CLAIM_ITEM_SCHEMA},
            "deadlines": {"type": "array", "items": _GROUNDED_CLAIM_ITEM_SCHEMA},
            "extraction_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Model's confidence in the accuracy/completeness of this extraction, from 0 to 1.",
            },
        },
        "required": ["risk_areas", "action_items", "responsible_parties", "deadlines", "extraction_confidence"],
    },
}


def extract_compliance_entities(chunk: DocumentChunk, client: anthropic.Anthropic) -> ExtractedEntities:
    """Call Claude (via forced tool use) to extract structured compliance
    entities from a single document chunk, then verify every claimed
    source_quote against the chunk's actual text before trusting it.

    Quotes that fail verification are dropped (not fabricated, not crashed
    on) and reported via a print summary, since silently keeping an
    unverifiable quote in a compliance pipeline is unacceptable.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        tools=[EXTRACTED_ENTITIES_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "extract_compliance_entities"},
        messages=[
            {
                "role": "user",
                "content": f"Document chunk (chunk_index={chunk.chunk_index}):\n\n{chunk.text}",
            }
        ],
    )

    tool_use_block = next(block for block in response.content if block.type == "tool_use")
    raw = tool_use_block.input

    verified_count = 0
    rejected_count = 0

    def _ground_claim(raw_claim: dict) -> GroundedClaim | None:
        nonlocal verified_count, rejected_count
        quote = raw_claim.get("source_quote", "")
        if not verify_grounding(quote, chunk.text):
            print(
                f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
                f"REJECTED unverifiable quote: {quote!r}"
            )
            rejected_count += 1
            return None
        position = find_quote_position(quote, chunk.text)
        if position is None:
            print(
                f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
                f"REJECTED quote (normalized match passed but position lookup failed): {quote!r}"
            )
            rejected_count += 1
            return None
        verified_count += 1
        local_start, local_end = position
        absolute_range = (chunk.char_start + local_start, chunk.char_start + local_end)
        return GroundedClaim(
            value=raw_claim["value"],
            source_quote=chunk.text[local_start:local_end],
            source_char_range=absolute_range,
        )

    def _ground_action_item(raw_item: dict) -> ActionItem | None:
        nonlocal verified_count, rejected_count
        quote = raw_item.get("source_quote", "")
        if not verify_grounding(quote, chunk.text):
            print(
                f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
                f"REJECTED unverifiable action_item quote: {quote!r}"
            )
            rejected_count += 1
            return None
        position = find_quote_position(quote, chunk.text)
        if position is None:
            print(
                f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
                f"REJECTED action_item quote (normalized match passed but position lookup failed): {quote!r}"
            )
            rejected_count += 1
            return None
        verified_count += 1
        local_start, local_end = position
        absolute_range = (chunk.char_start + local_start, chunk.char_start + local_end)
        deadline_parsed = raw_item.get("deadline_parsed") or None
        try:
            return ActionItem(
                action=raw_item["action"],
                owner_type=raw_item["owner_type"],
                deadline_raw=raw_item["deadline_raw"],
                deadline_parsed=deadline_parsed,
                priority=raw_item["priority"],
                source_quote=chunk.text[local_start:local_end],
                source_char_range=absolute_range,
            )
        except ValidationError as exc:
            print(
                f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
                f"DROPPED action_item (ValidationError on {raw_item.get('action', '')!r}): {exc}"
            )
            rejected_count += 1
            return None

    risk_areas = [c for c in (_ground_claim(r) for r in raw.get("risk_areas", [])) if c is not None]
    responsible_parties = [
        c for c in (_ground_claim(r) for r in raw.get("responsible_parties", [])) if c is not None
    ]
    deadlines = [c for c in (_ground_claim(r) for r in raw.get("deadlines", [])) if c is not None]
    action_items = [
        a for a in (_ground_action_item(r) for r in raw.get("action_items", [])) if a is not None
    ]

    print(
        f"[extract_compliance_entities] chunk_index={chunk.chunk_index}: "
        f"{verified_count} quote(s) verified, {rejected_count} quote(s) rejected"
    )

    return ExtractedEntities(
        risk_areas=risk_areas,
        action_items=action_items,
        responsible_parties=responsible_parties,
        deadlines=deadlines,
        source_chunk_index=chunk.chunk_index,
        extraction_confidence=raw["extraction_confidence"],
    )


_DOMAIN_ENUM = ["aml", "data_privacy", "capital_requirements", "kyc", "sanctions", "reporting", "operational_risk", "other"]

CLASSIFICATION_TOOL_SCHEMA = {
    "name": "classify_document",
    "description": "Classify a compliance document by its primary domain and urgency level, based on the provided extracted findings summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "compliance_domain": {
                "type": "string",
                "enum": _DOMAIN_ENUM,
                "description": "The primary compliance domain this document falls under.",
            },
            "urgency_level": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
                "description": "Overall urgency level assigned to the document.",
            },
            "urgency_rationale": {
                "type": "string",
                "description": "Explanation of the urgency level, citing specific phrases from the provided signals, deadlines, or risk areas.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Model's confidence in this classification, from 0 to 1.",
            },
            "is_cross_domain": {
                "type": "boolean",
                "description": "Whether the document materially touches more than one compliance domain.",
            },
            "secondary_domains": {
                "type": "array",
                "items": {"type": "string", "enum": _DOMAIN_ENUM},
                "description": "Additional compliance domains materially present in the document, if any.",
            },
        },
        "required": ["compliance_domain", "urgency_level", "urgency_rationale", "confidence", "is_cross_domain", "secondary_domains"],
    },
}


def classify_document(
    findings: MergedFindings,
    client: anthropic.Anthropic,
) -> Classification:
    """Call Claude (forced tool use) to classify a MergedFindings object by
    compliance domain and urgency level.

    Builds a focused text summary of the findings rather than dumping the
    full JSON, to minimize tokens and surface exactly what the classifier
    needs. Uses claude-haiku-4-5 (cheaper/faster than Sonnet — classification
    is simpler than extraction).
    """
    urgency_block = (
        "\n".join(f"- {s}" for s in findings.dominant_urgency_signals)
        if findings.dominant_urgency_signals
        else "None detected"
    )
    deadline_values = [d.value for d in findings.all_deadlines][:15]
    deadlines_block = "\n".join(f"- {v}" for v in deadline_values) if deadline_values else "None"
    risk_values = [r.value for r in findings.all_risk_areas][:15]
    risks_block = "\n".join(f"- {v}" for v in risk_values) if risk_values else "None"
    party_values = [p.value for p in findings.all_responsible_parties][:10]
    parties_block = "\n".join(f"- {v}" for v in party_values) if party_values else "None"

    user_message = (
        f"Compliance Domain Classification Request\n\n"
        f"Document ID: {findings.document_id}\n"
        f"Chunks with content: {findings.chunks_with_content} of {findings.chunks_processed}\n"
        f"Overall extraction confidence: {findings.overall_confidence:.2f}\n\n"
        f"URGENCY SIGNALS (exact phrases from document):\n{urgency_block}\n\n"
        f"DEADLINES (verbatim from document):\n{deadlines_block}\n\n"
        f"RISK AREAS identified:\n{risks_block}\n\n"
        f"RESPONSIBLE PARTIES:\n{parties_block}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=CLASSIFICATION_SYSTEM_PROMPT,
        tools=[CLASSIFICATION_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "classify_document"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_use_block = next(block for block in response.content if block.type == "tool_use")
    raw = tool_use_block.input

    try:
        result = Classification(
            compliance_domain=ComplianceDomain(raw["compliance_domain"]),
            urgency_level=UrgencyLevel(raw["urgency_level"]),
            urgency_rationale=raw["urgency_rationale"],
            confidence=raw["confidence"],
            is_cross_domain=raw["is_cross_domain"],
            secondary_domains=[ComplianceDomain(d) for d in raw.get("secondary_domains", [])],
        )
    except ValidationError as exc:
        raise RuntimeError(
            f"[classify_document] Classification validation failed for document "
            f"{findings.document_id!r}: {exc}"
        ) from exc

    print(
        f"[classify_document] domain={result.compliance_domain} "
        f"urgency={result.urgency_level} confidence={result.confidence:.2f}"
    )
    return result


_URGENCY_KEYWORDS = frozenset([
    "expired", "delay", "immediate", "urgent", "soon", "swiftly",
    "critical", "without delay", "as soon as possible", "overdue",
])

_YEAR_PATTERN = re.compile(r"\b\d{4}\b")


def merge_findings(
    chunks: list[DocumentChunk],
    all_entities: list[ExtractedEntities],
    document_id: str,
) -> MergedFindings:
    """Consolidate per-chunk extraction results into a single MergedFindings
    object. Pure Python — no LLM call.

    Deduplicates risk_areas and responsible_parties by normalized value
    (case-insensitive, whitespace-normalized). Preserves all action_items
    and deadlines without deduplication. Collects urgency-signaling phrases
    from deadline values and action_item deadline_raw fields.
    """
    seen_risk_keys: set[str] = set()
    seen_party_keys: set[str] = set()
    seen_urgency_keys: set[str] = set()

    all_risk_areas: list[GroundedClaim] = []
    all_action_items: list[ActionItem] = []
    all_responsible_parties: list[GroundedClaim] = []
    all_deadlines: list[GroundedClaim] = []
    dominant_urgency_signals: list[str] = []
    source_chunk_indices: list[int] = []

    confidence_sum = 0.0
    chunks_with_content = 0

    def _is_urgency_signal(phrase: str) -> bool:
        lower = phrase.lower()
        if any(kw in lower for kw in _URGENCY_KEYWORDS):
            return True
        if _YEAR_PATTERN.search(phrase):
            return True
        return False

    for entities in all_entities:
        has_content = bool(
            entities.risk_areas or entities.action_items or entities.deadlines
        )
        if has_content:
            chunks_with_content += 1
            confidence_sum += entities.extraction_confidence
            source_chunk_indices.append(entities.source_chunk_index)

        for claim in entities.risk_areas:
            key = normalize_whitespace(claim.value).lower()
            if key not in seen_risk_keys:
                seen_risk_keys.add(key)
                all_risk_areas.append(claim)

        all_action_items.extend(entities.action_items)

        for claim in entities.responsible_parties:
            key = normalize_whitespace(claim.value).lower()
            if key not in seen_party_keys:
                seen_party_keys.add(key)
                all_responsible_parties.append(claim)

        all_deadlines.extend(entities.deadlines)

        urgency_candidates = [d.value for d in entities.deadlines] + [
            a.deadline_raw for a in entities.action_items
        ]
        for phrase in urgency_candidates:
            normalized = normalize_whitespace(phrase)
            key = normalized.lower()
            if key not in seen_urgency_keys and _is_urgency_signal(normalized):
                seen_urgency_keys.add(key)
                dominant_urgency_signals.append(normalized)

    overall_confidence = confidence_sum / chunks_with_content if chunks_with_content else 0.0

    return MergedFindings(
        document_id=document_id,
        all_risk_areas=all_risk_areas,
        all_action_items=all_action_items,
        all_responsible_parties=all_responsible_parties,
        all_deadlines=all_deadlines,
        chunks_processed=len(chunks),
        chunks_with_content=chunks_with_content,
        overall_confidence=overall_confidence,
        dominant_urgency_signals=dominant_urgency_signals,
        source_chunk_indices=source_chunk_indices,
    )


JURISDICTION_CHECK_TOOL_SCHEMA = {
    "name": "identify_unrecognized_jurisdictions",
    "description": "Identify party names not matching known regulatory bodies",
    "input_schema": {
        "type": "object",
        "properties": {
            "unrecognized_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Party names not matching any known regulatory body or jurisdiction",
            }
        },
        "required": ["unrecognized_names"],
    },
}

_VAGUE_DEADLINE_PHRASES = [
    "not specified",
    "as agreed",
    "ongoing",
    "to be determined",
    "tbd",
    "not stated",
    "implement its fatf action plan",
]

_HIGH_URGENCY_SIGNALS = frozenset(["expired", "without delay", "immediately", "overdue"])
_LOW_URGENCY_SIGNALS = frozenset(["informational", "no action required", "for information only", "awareness only"])

_AMBIGUOUS_OWNER_PHRASES = [
    "or their",
    "designated representative",
    "or equivalent",
    "competent authority",
    "relevant authority",
    "or delegate",
]


def flag_ambiguities(
    findings: MergedFindings,
    classification: Classification,
    client: anthropic.Anthropic,
) -> list[AmbiguityFlag]:
    """Detect ambiguities in MergedFindings + Classification that require
    human judgment before routing.

    Runs five deterministic checks (pure Python) and one lightweight LLM
    call for unrecognized jurisdiction detection. Never crashes — the
    jurisdiction check failure is caught and skipped.
    """
    flags: list[AmbiguityFlag] = []

    # CHECK 1 — missing_deadline
    if findings.all_action_items:
        vague_count = sum(
            1
            for item in findings.all_action_items
            if any(phrase in item.deadline_raw.lower() for phrase in _VAGUE_DEADLINE_PHRASES)
        )
        total = len(findings.all_action_items)
        if vague_count / total > 0.5:
            flags.append(AmbiguityFlag(
                flag_type=AmbiguityType.MISSING_DEADLINE,
                description=(
                    f"{vague_count} of {total} action items have vague or unspecified deadlines. "
                    "Human reviewer must determine actual deadline before prioritizing."
                ),
                severity=classification.urgency_level,
            ))

    # CHECK 2 — ambiguous_owner
    for party in findings.all_responsible_parties:
        value_lower = party.value.lower()
        if any(phrase in value_lower for phrase in _AMBIGUOUS_OWNER_PHRASES):
            flags.append(AmbiguityFlag(
                flag_type=AmbiguityType.AMBIGUOUS_OWNER,
                description=(
                    f"Responsible party '{party.value}' is non-specific. "
                    "Human reviewer must identify the actual owner before routing."
                ),
                severity=UrgencyLevel.HIGH,
            ))

    # CHECK 3 — cross_domain
    if classification.is_cross_domain:
        secondary = ", ".join(d.value for d in classification.secondary_domains)
        flags.append(AmbiguityFlag(
            flag_type=AmbiguityType.CROSS_DOMAIN,
            description=(
                f"Document spans multiple compliance domains: "
                f"{classification.compliance_domain.value} (primary) + "
                f"{secondary} (secondary). "
                "Single-team routing may be insufficient — consider joint review."
            ),
            severity=UrgencyLevel.HIGH,
        ))

    # CHECK 4 — low_confidence
    if classification.confidence < 0.70:
        flags.append(AmbiguityFlag(
            flag_type=AmbiguityType.LOW_CONFIDENCE,
            description=(
                f"Classification confidence is {classification.confidence:.0%}. "
                f"Domain assignment ({classification.compliance_domain.value}) should be "
                "verified by human reviewer before routing."
            ),
            severity=UrgencyLevel.MEDIUM,
        ))

    # CHECK 5 — contradictory_signals
    signals_lower = [s.lower() for s in findings.dominant_urgency_signals]
    has_high = any(kw in sig for sig in signals_lower for kw in _HIGH_URGENCY_SIGNALS)
    has_low = any(kw in sig for sig in signals_lower for kw in _LOW_URGENCY_SIGNALS)
    if has_high and has_low:
        flags.append(AmbiguityFlag(
            flag_type=AmbiguityType.CONTRADICTORY_SIGNALS,
            description=(
                "Document contains both high-urgency and low-urgency signals. "
                "Human reviewer must determine actual priority level."
            ),
            severity=UrgencyLevel.HIGH,
        ))

    # CHECK 6 — unrecognized_jurisdiction (one LLM call)
    party_names = [p.value for p in findings.all_responsible_parties]
    if party_names:
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=JURISDICTION_CHECK_PROMPT,
                tools=[JURISDICTION_CHECK_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "identify_unrecognized_jurisdictions"},
                messages=[{
                    "role": "user",
                    "content": "Responsible party names to check:\n" + "\n".join(f"- {n}" for n in party_names),
                }],
            )
            tool_block = next(b for b in response.content if b.type == "tool_use")
            unrecognized = tool_block.input.get("unrecognized_names", [])
            for name in unrecognized:
                flags.append(AmbiguityFlag(
                    flag_type=AmbiguityType.UNRECOGNIZED_JURISDICTION,
                    description=(
                        f"'{name}' is not a recognized regulatory body or jurisdiction in our "
                        "routing rules. Human reviewer must determine appropriate owner before routing."
                    ),
                    severity=UrgencyLevel.HIGH,
                ))
        except Exception as exc:
            print(f"[flag_ambiguities] WARNING: jurisdiction check failed, skipping: {exc}")

    print(f"[flag_ambiguities] {len(flags)} flag(s) raised: {[f.flag_type.value for f in flags]}")
    return flags


EXPECTED_GREY_LIST_COUNTRIES = [
    "Algeria", "Angola", "Bolivia", "Bulgaria", "Cameroon", "Cote d'Ivoire",
    "Democratic Republic of the Congo", "Haiti", "Kenya", "Kuwait", "Lao PDR",
    "Lebanon", "Monaco", "Namibia", "Nepal", "Papua New Guinea", "South Sudan",
    "Syria", "Venezuela", "Vietnam", "Yemen",
]

PAGE_FOOTER_SHAPE_PATTERN = re.compile(r"^(\d{1,3}\s*\|\s*|\|\s*\d{1,3}\s*)$")


def _print_chunking_summary(sample_path: str) -> None:
    result_chunks = chunk_document(sample_path)

    structural_count = sum(1 for c in result_chunks if c.chunking_method == "structural")
    fallback_count = sum(1 for c in result_chunks if c.chunking_method == "fixed_size_fallback")
    avg_tokens = (
        sum(c.token_count for c in result_chunks) / len(result_chunks) if result_chunks else 0
    )
    repeated_header_like = sum(
        1
        for c in result_chunks
        if c.section_title and "targeted report" in c.section_title.lower()
    )
    tiny_chunks = sum(1 for c in result_chunks if c.token_count < 50)

    print(f"=== {sample_path} ===")
    print(f"Total chunks: {len(result_chunks)}")
    print(f"Structural: {structural_count}, Fixed-size fallback: {fallback_count}")
    print(f"Average token count per chunk: {avg_tokens:.1f}")
    print(f"Chunks with repeated-header-looking section_title: {repeated_header_like}")
    print(f"Chunks under 50 tokens: {tiny_chunks}")

    def _normalize_for_match(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

    section_titles = [c.section_title for c in result_chunks if c.section_title]
    normalized_titles = [_normalize_for_match(title) for title in section_titles]
    missing_countries = [
        country
        for country in EXPECTED_GREY_LIST_COUNTRIES
        if not any(_normalize_for_match(country) in title for title in normalized_titles)
    ]
    if missing_countries:
        print(f"Missing expected countries ({len(missing_countries)}/{len(EXPECTED_GREY_LIST_COUNTRIES)}): {missing_countries}")
    else:
        print(f"All {len(EXPECTED_GREY_LIST_COUNTRIES)} expected countries found.")

    page_footer_shaped = sum(
        1
        for c in result_chunks
        if c.section_title and PAGE_FOOTER_SHAPE_PATTERN.match(c.section_title.strip())
    )
    print(f"Chunks with page-footer-shaped section_title: {page_footer_shaped}")
    print()

    for chunk in result_chunks:
        print(
            f"[{chunk.chunk_index}] ({chunk.chunking_method}) section='{chunk.section_title}' "
            f"confidence={chunk.detection_confidence} pages={chunk.page_start}-{chunk.page_end}"
        )
        print(f"    {chunk.text[:100]!r}")
        print()


if __name__ == "__main__":
    import json

    sample_paths = [
        "sample_docs/fatf_grey_list.pdf",
        "sample_docs/targeted-report-on-stablecoins-and-unhosted-wallets.pdf.coredownload.inline-2.pdf",
    ]

    for sample_path in sample_paths:
        _print_chunking_summary(sample_path)

    # ------------------------------------------------------------------
    # merge_findings() smoke test — no API calls, mock entities only
    # ------------------------------------------------------------------
    print("=== merge_findings() smoke test (mock data, no API calls) ===\n")

    grey_list_chunks = chunk_document("sample_docs/fatf_grey_list.pdf")

    def _claim(value: str, quote: str, start: int) -> GroundedClaim:
        return GroundedClaim(value=value, source_quote=quote, source_char_range=(start, start + len(quote)))

    def _action(action: str, owner: str, deadline_raw: str, priority: str, quote: str, start: int) -> ActionItem:
        from schemas import UrgencyLevel
        return ActionItem(
            action=action,
            owner_type=owner,
            deadline_raw=deadline_raw,
            priority=UrgencyLevel(priority),
            source_quote=quote,
            source_char_range=(start, start + len(quote)),
        )

    # chunk 0 — Algeria: 2 risk areas (one will be duplicated in chunk 4), 1 action item, 2 parties, 1 deadline
    entities_0 = ExtractedEntities(
        risk_areas=[
            _claim("strategic AML deficiencies", "work remains to address its strategic AML deficiencies", 100),
            _claim("terrorism financing gaps", "financial sanctions for terrorism financing", 200),
        ],
        action_items=[
            _action(
                "Implement risk-based supervision of DNFBPs",
                "supervisory authority",
                "all deadlines have now expired",
                "critical",
                "implementing risk-based supervision of DNFBPs",
                300,
            ),
        ],
        responsible_parties=[
            _claim("FATF", "work with the FATF and ESAAMLG", 400),
            _claim("ESAAMLG", "work with the FATF and ESAAMLG", 400),
        ],
        deadlines=[
            _claim("all deadlines have now expired", "all deadlines have now expired", 500),
        ],
        source_chunk_index=0,
        extraction_confidence=0.88,
    )

    # chunk 4 — Cameroon: 3 risk areas (first is a duplicate of chunk 0's first), 2 action items, 1 party, 2 deadlines
    entities_4 = ExtractedEntities(
        risk_areas=[
            _claim("strategic AML deficiencies", "address its strategic AML deficiencies as soon as possible", 1000),  # duplicate
            _claim("NPO oversight gaps", "risk-based monitoring of NPOs to prevent abuse for TF purposes", 1100),
            _claim("beneficial ownership transparency", "ensuring beneficial ownership information is accurate", 1200),
        ],
        action_items=[
            _action(
                "Designate an authority for AML/CFT supervision of all DNFBPs",
                "government authority",
                "as soon as possible",
                "high",
                "designating an authority for AML/CFT supervision of all DNFBPs",
                1300,
            ),
            _action(
                "Demonstrate sustained increase in TF investigations and prosecutions",
                "law enforcement",
                "without delay",
                "critical",
                "demonstrating a sustained increase in the number of TF investigations",
                1400,
            ),
        ],
        responsible_parties=[
            _claim("GABAC", "work with the FATF and GABAC", 1500),
        ],
        deadlines=[
            _claim("as soon as possible", "continue to implement its action plan as soon as possible", 1600),
            _claim("without delay", "implement targeted financial sanctions without delay", 1700),
        ],
        source_chunk_index=4,
        extraction_confidence=0.85,
    )

    # chunk 6 — DRC: 1 risk area, 1 action item, 1 party, 1 deadline
    entities_6 = ExtractedEntities(
        risk_areas=[
            _claim("ML investigation gaps", "demonstrating an increase in ML investigations and prosecutions", 2000),
        ],
        action_items=[
            _action(
                "Strengthen effectiveness of AML/CFT regime",
                "government / competent authorities",
                "within agreed timeframes",
                "high",
                "strengthen the effectiveness of its AML/CFT regime",
                2100,
            ),
        ],
        responsible_parties=[
            _claim("FATF", "work with the FATF and GABAC to strengthen", 2200),  # duplicate party — should dedup
        ],
        deadlines=[
            _claim("within agreed timeframes", "continue to work within agreed timeframes", 2300),
        ],
        source_chunk_index=6,
        extraction_confidence=0.82,
    )

    # All other chunks: empty
    empty_chunk_indices = [i for i in range(len(grey_list_chunks)) if i not in (0, 4, 6)]
    mock_all_entities: list[ExtractedEntities] = []
    for chunk in grey_list_chunks:
        if chunk.chunk_index == 0:
            mock_all_entities.append(entities_0)
        elif chunk.chunk_index == 4:
            mock_all_entities.append(entities_4)
        elif chunk.chunk_index == 6:
            mock_all_entities.append(entities_6)
        else:
            mock_all_entities.append(ExtractedEntities(
                risk_areas=[], action_items=[], responsible_parties=[], deadlines=[],
                source_chunk_index=chunk.chunk_index, extraction_confidence=0.1,
            ))

    merged = merge_findings(grey_list_chunks, mock_all_entities, document_id="fatf_grey_list_mock")
    print(json.dumps(merged.model_dump(mode="json"), indent=2))

    total_raw_risk_areas = sum(len(e.risk_areas) for e in mock_all_entities)
    duplicates_removed = total_raw_risk_areas - len(merged.all_risk_areas)
    print(f"\nDeduplication check: {len(merged.all_risk_areas)} unique risk areas from "
          f"{total_raw_risk_areas} total ({duplicates_removed} duplicates removed)")
    print(f"Urgency signals found: {merged.dominant_urgency_signals}")
    print(f"Chunks with content: {merged.chunks_with_content} of {merged.chunks_processed} total")

    # ------------------------------------------------------------------
    # classify_document() smoke test — one real Haiku API call
    # ------------------------------------------------------------------
    import os
    from dotenv import load_dotenv

    print("\n=== classify_document() smoke test (1 Haiku API call) ===\n")

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env file.")

    anthropic_client = anthropic.Anthropic(api_key=api_key)

    # Build a realistic mock MergedFindings directly — no chunk_document() call
    mock_merged = MergedFindings(
        document_id="fatf_grey_list_2026",
        all_risk_areas=[
            GroundedClaim(value="strategic AML deficiencies", source_quote="address its strategic AML deficiencies", source_char_range=(100, 140)),
            GroundedClaim(value="terrorism financing gaps", source_quote="financial sanctions for terrorism financing", source_char_range=(200, 243)),
            GroundedClaim(value="NPO oversight gaps", source_quote="risk-based monitoring of NPOs to prevent abuse for TF purposes", source_char_range=(300, 362)),
            GroundedClaim(value="beneficial ownership transparency", source_quote="ensuring beneficial ownership information is accurate", source_char_range=(400, 453)),
            GroundedClaim(value="ML investigation and prosecution deficiencies", source_quote="demonstrating an increase in ML investigations and prosecutions", source_char_range=(500, 563)),
        ],
        all_action_items=[
            ActionItem(
                action="Implement risk-based supervision of DNFBPs",
                owner_type="supervisory authority",
                deadline_raw="all deadlines have now expired",
                priority=UrgencyLevel.CRITICAL,
                source_quote="implementing risk-based supervision of DNFBPs",
                source_char_range=(600, 645),
            ),
            ActionItem(
                action="Designate an authority for AML/CFT supervision of all DNFBPs",
                owner_type="government authority",
                deadline_raw="as soon as possible",
                priority=UrgencyLevel.HIGH,
                source_quote="designating an authority for AML/CFT supervision of all DNFBPs",
                source_char_range=(700, 762),
            ),
            ActionItem(
                action="Demonstrate increase in TF investigations without delay",
                owner_type="law enforcement",
                deadline_raw="without delay",
                priority=UrgencyLevel.CRITICAL,
                source_quote="demonstrating a sustained increase in the number of TF investigations",
                source_char_range=(800, 869),
            ),
        ],
        all_responsible_parties=[
            GroundedClaim(value="FATF", source_quote="work with the FATF and ESAAMLG", source_char_range=(900, 930)),
            GroundedClaim(value="ESAAMLG", source_quote="work with the FATF and ESAAMLG", source_char_range=(900, 930)),
            GroundedClaim(value="GABAC", source_quote="work with the FATF and GABAC", source_char_range=(1000, 1028)),
        ],
        all_deadlines=[
            GroundedClaim(value="all deadlines have now expired", source_quote="all deadlines have now expired", source_char_range=(1100, 1130)),
            GroundedClaim(value="as soon as possible", source_quote="continue to implement its action plan as soon as possible", source_char_range=(1200, 1257)),
            GroundedClaim(value="without delay", source_quote="implement targeted financial sanctions without delay", source_char_range=(1300, 1351)),
        ],
        chunks_processed=21,
        chunks_with_content=15,
        overall_confidence=0.86,
        dominant_urgency_signals=[
            "all deadlines have now expired",
            "as soon as possible",
            "without delay",
        ],
        source_chunk_indices=list(range(15)),
    )

    classification = classify_document(mock_merged, anthropic_client)
    print(json.dumps(classification.model_dump(mode="json"), indent=2))

    # ------------------------------------------------------------------
    # flag_ambiguities() smoke test — 1 Haiku API call for jurisdiction check
    # ------------------------------------------------------------------
    print("\n=== flag_ambiguities() smoke test (1 Haiku API call) ===\n")

    from schemas import AmbiguityFlag  # already imported at module level, explicit here for clarity

    ambiguity_findings = MergedFindings(
        document_id="fatf_grey_list_ambiguity_test",
        all_risk_areas=[
            GroundedClaim(value="strategic AML deficiencies", source_quote="address its strategic AML deficiencies", source_char_range=(0, 40)),
        ],
        all_action_items=[
            ActionItem(
                action="Implement risk-based supervision of DNFBPs",
                owner_type="supervisory authority",
                deadline_raw="implement its FATF action plan",
                priority=UrgencyLevel.HIGH,
                source_quote="implementing risk-based supervision of DNFBPs",
                source_char_range=(100, 145),
            ),
            ActionItem(
                action="Demonstrate increase in ML investigations",
                owner_type="law enforcement",
                deadline_raw="implement its FATF action plan",
                priority=UrgencyLevel.HIGH,
                source_quote="demonstrating an increase in ML investigations",
                source_char_range=(200, 246),
            ),
            ActionItem(
                action="Strengthen effectiveness of AML/CFT regime",
                owner_type="government",
                deadline_raw="within agreed timeframes",
                priority=UrgencyLevel.MEDIUM,
                source_quote="strengthen the effectiveness of its AML/CFT regime",
                source_char_range=(300, 350),
            ),
        ],
        all_responsible_parties=[
            GroundedClaim(
                value="competent authority or their designated representative",
                source_quote="competent authority or their designated representative",
                source_char_range=(400, 454),
            ),
            GroundedClaim(value="FATF", source_quote="work with the FATF", source_char_range=(500, 519)),
        ],
        all_deadlines=[
            GroundedClaim(value="all deadlines have now expired", source_quote="all deadlines have now expired", source_char_range=(600, 630)),
        ],
        chunks_processed=21,
        chunks_with_content=10,
        overall_confidence=0.80,
        dominant_urgency_signals=[
            "all deadlines have now expired",
            "informational update only",
        ],
        source_chunk_indices=list(range(10)),
    )

    ambiguity_classification = Classification(
        compliance_domain=ComplianceDomain.AML,
        urgency_level=UrgencyLevel.CRITICAL,
        urgency_rationale="All deadlines have expired.",
        confidence=0.65,
        is_cross_domain=True,
        secondary_domains=[ComplianceDomain.KYC],
    )

    ambiguity_flags = flag_ambiguities(ambiguity_findings, ambiguity_classification, anthropic_client)
    for flag in ambiguity_flags:
        print(json.dumps(flag.model_dump(mode="json"), indent=2))
    print(f"\n{len(ambiguity_flags)} flags raised: {[f.flag_type.value for f in ambiguity_flags]}")
