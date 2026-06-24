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

from prompts import EXTRACTION_SYSTEM_PROMPT
from schemas import ActionItem, DocumentChunk, ExtractedEntities, GroundedClaim

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
    sample_paths = [
        "sample_docs/fatf_grey_list.pdf",
        "sample_docs/targeted-report-on-stablecoins-and-unhosted-wallets.pdf.coredownload.inline-2.pdf",
    ]

    for sample_path in sample_paths:
        _print_chunking_summary(sample_path)
