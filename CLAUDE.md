# CLAUDE.md — Compliance Triage Agent

## Project Context

This is a Compliance Triage Agent built with LangGraph and the Claude API. It
ingests a regulatory document (PDF), extracts structured compliance entities
(risk areas, action items, deadlines, responsible parties), classifies it by
domain and urgency, and proposes — but never executes — a routing decision,
stopping at a mandatory human checkpoint before anything is considered final.
Constraints: human-in-the-loop is non-negotiable (the agent must never
auto-route), no greenfield infra (build on existing LangGraph/Claude/LangSmith
stack, not custom orchestration), full audit trail required for every
automated step, and a 4-week Phase 1 scope. This is a learning-focused build —
explanations and rationale matter as much as working code.

## File Responsibilities

- `schemas.py` — data shapes only. No logic, no LLM calls.
- `tools.py` — agent capabilities (functions the agent can call). No
  orchestration. Its `__main__` runs chunking validation only (no API calls,
  cheap to run at any time).
- `run_extraction.py` — standalone extraction runner. Accepts `--chunks N`
  (default 3, cheap smoke test) or `--all` (full 21-chunk run). The only
  entry point that makes real API calls during development.
- `prompts.py` — prompt templates only.
- `agent.py` — LangGraph orchestration, the loop, the human checkpoint.
- `audit_log.py` — compliance logging.

## Established Conventions

- **Always verify your own changes.** After editing a file, grep for a
  distinctive new identifier (function name, variable, constant) you just
  added and show the output BEFORE declaring a task complete. Do not just
  narrate what you did — prove the file on disk actually contains it. (This
  is not hypothetical: an earlier session reported a change as done that
  had not actually landed; see DESIGN.md section 3.)
- **When you discover a bug or limitation outside the current task's stated
  scope, flag it explicitly and ask before fixing it.** Do not silently
  expand scope.
- **When testing against sample documents, test against ALL files currently
  in `sample_docs/`, not just one.** This project has been burned before by
  heuristics that worked on one document and failed differently on another.
- **Grounding fields (`source_quote`, `source_char_range`) must be exact
  verbatim substrings of the source text — never paraphrased.** This is
  enforced via `verify_grounding()`, now implemented and exercised against
  real extraction output.

## Known Limitations (do not silently "fix" without asking)

- `section_title` is unreliable (advisory only) on densely-typeset PDFs with
  footnotes/citations. This is accepted, documented behavior, not a bug to
  chase further without discussion — see DESIGN.md section 4.
- `verify_grounding()` is **whitespace-normalized exact match**, not raw
  exact match (corrected from the original raw-exact-match design after real
  LLM extraction testing showed ~84% of claimed quotes were rejected purely
  because PDF text extraction introduces mid-sentence line breaks from
  visual page wrapping, which an LLM naturally reproduces without). Both the
  quote and the source text are run through `normalize_whitespace()`
  (collapse any whitespace run to a single space) before the substring
  check — this is still a strict substring match, not fuzzy matching; no
  wording, character, or punctuation differences are tolerated.
  `find_quote_position()` then locates the quote's real offsets in the
  *original* (non-normalized) text via a whitespace-tolerant regex, so
  `source_char_range` still points at exact original-text positions. Known
  remaining gaps, observed but intentionally not auto-fixed: curly vs.
  straight apostrophe mismatches (e.g. source `’` vs. LLM-emitted `'`), and
  hyphenation-across-linebreak artifacts (e.g. source `risk-\nbased`
  normalizing to `risk- based`, which doesn't match an LLM-emitted
  `risk-based`).

## Current Status

All six tools complete and validated. Next: `agent.py` (LangGraph graph
with parallel extraction via Send API and human checkpoint interrupt),
then `audit_log.py`, then end-to-end demo run on real document.
See DESIGN.md for full history and rationale.

## Chunking Config Reference

`target_chunk_size=1500` tokens, `min_chunk_size=200` tokens,
`overlap_fallback=150` tokens (fixed-size fallback only). ~4 chars/token
approximation used throughout, no tokenizer dependency added yet.

---

Keep both files up to date as the project progresses — DESIGN.md gets a new
entry under section 3/4/5 whenever a real design decision or bug is
made/found in future tasks; this file's "Current Status" section gets
updated at the end of each task.
