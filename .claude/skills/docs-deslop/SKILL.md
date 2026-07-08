---
name: docs-deslop
description: >-
  Condense and clarify an operational/research doc (SKILL.md, ops.md, tracker, README,
  agent_log) by an editor-subagent dispatched with fresh context and a list of file paths
  to review. The editor MOVES information to a concept-ordered taxonomy with functional
  subsections, then cuts paragraph-by-paragraph: stale gotchas + their corrections,
  deprecated features, rationalizations/justifications for actions (docs say WHAT to do,
  not WHY), partially-redundant sections, and flowery language. CONDENSE AND CLARIFY ONLY
  — never add or elaborate. Backs up every doc to ~/Documents/slop_docs/ before editing
  (dated + original filename), then leans toward OVER-condensing (backups exist). Returns
  a per-doc percent-shortened count + a brief overview of what was compressed for the
  supervisor to review. Use when docs have grown bloated/stale/repetitive, when a sweep
  flags doc bloat, or when the supervisor says "deslop these docs".
---

# docs-deslop

A **doc-editor** subagent that takes a list of doc paths and **condenses + reorganizes**
each one — without ever adding content. Docs in this repo accumulate cruft: stale gotchas
left in place long after the fix shipped, deprecated features still described at length,
rationalizations for why a step exists, sections that say the same thing three ways, and
flowery prose around a two-line procedure. This skill is the cleanup pass that reclaims
that space.

> **You are a SUBAGENT EDITOR. Your job is to CONDENSE AND CLARIFY — never to add,
> elaborate, or insert new information.** You do not introduce facts, commands, flags,
> or guidance that was not already in the document. You may move, merge, tighten, and cut
> — that is all. If you find yourself writing a new explanation, stop: that is out of
> scope. Flag the gap in your return summary instead and let the supervisor decide whether
> to author it.

## Inputs

From the dispatching supervisor you receive:

| Input | Required | Notes |
|---|---|---|
| **List of doc paths** to edit | yes | absolute or repo-relative; one or many |
| Scope constraints (optional) | no | e.g. "only the Guardrails section", "leave the worked example" — if none given, the whole doc is in scope |

If a path does not exist or is not a doc (`.md`/`.txt`), report it back unchanged with a
note — do not guess or create files.

---

## 0. Back up the original (FIRST, before any edit)

**Before touching a doc, save a dated copy of the verbatim original** to
`/Users/benjaminfeuer/Documents/slop_docs/`. This is non-negotiable — the whole skill
licenses aggressive cuts on the basis that the original is recoverable.

Naming: **`<YYYY-MM-DD_HHMM>_<original-filename>`** — date+time prefix, original filename
preserved (including extension). Example: editing `ops/leonardo/ops.md` on 2026-07-08 at
14:32 → `/Users/benjaminfeuer/Documents/slop_docs/2026-07-08_1432_ops.md`.

```bash
STAMP=$(date +%Y-%m-%d_%H%M)
cp "<doc-path>" "/Users/benjaminfeuer/Documents/slop_docs/${STAMP}_$(basename "<doc-path>")"
```

Do this **once per doc**, before the first edit. If two docs share a basename, the date
prefix disambiguates them. Confirm the backup landed (the `cp` above either succeeds or
errors loudly — no silent skip).

---

## 1. First pass — reorganize (move, don't write)

Read the whole doc end to end first. Then **move** existing content into a concept-ordered
structure. The goal is a document a reader can **scan and reference** — find the right
section fast, read only what they need.

**Order by CONCEPT, not chronology.** Most docs accrete in the order things were learned
("first we hit X, then Y, then the fix for X…"). That serves the author, not the reader.
Reorder so that the structure reflects **what the content IS**, not when it was added. The
single exception: a doc that is **inherently chronological** (a launch runbook whose steps
are a temporal sequence; a bring-up ladder; a retry/chain-restart flow) — there, the
sequence IS the content and you preserve it.

**Organize into well-organized subsections with a natural taxonomy by function.** Group
related material under a heading that names the *function* of the material (access,
invocation, gotchas, guardrails, worked examples), not the *incidents* that produced it.
Use the taxonomy already present in the repo's best docs as a template:
`When to use` → `What it does`/`Resources` → `How to invoke`/numbered procedure →
`Gotchas`/`Guardrails` → `Worked example`. Merge scattered discussions of the same topic
into one place; a reader should never have to assemble an answer from three sections.

**During reorganization:**
- **Move** paragraphs and sections to their conceptually-correct home.
- **Merge** sections that cover the same function into one.
- **Promote** buried essential facts (e.g. a load-bearing prerequisite mentioned in a
  footnote) to where a reader will actually see them.
- **Do NOT** write new connective prose to smooth the moves. If a merge leaves a seam, let
  the condense pass (§2) tighten it. No new sentences.

At the end of this pass the doc should read in a logical concept order with clean
subsections — but it is likely still wordy. That is what the next pass fixes.

---

## 2. Second pass — condense and clean (paragraph by paragraph)

Go through the reorganized doc **paragraph by paragraph** and cut. This is where the bulk
of the reduction happens. Lean toward **over-condensing** — the backup in `slop_docs/`
exists precisely so you can be aggressive; a doc that comes back too tight is a quick
un-edit, while one that comes back too loose wastes the whole pass.

**Candidates for removal or tightening:**

| Cut | What it looks like | Why |
|---|---|---|
| **Stale gotchas + their corrections** | "NOTE: X used to break, but the fix in commit Y…"; a whole paragraph about a bug that's long fixed and whose fix is now the normal behavior | If the corrected behavior is now the default/only path, the history is dead weight. Keep the *current* rule, drop the story of the bug. |
| **Deprecated features** | Steps/flags/paths that no longer exist or are explicitly superseded; "legacy" paths called out as legacy | If it's deprecated, describing how to use it is actively harmful. Remove unless the doc's job is a migration guide. |
| **Rationalizations + justifications** | "We do X *because*…", "the reason for this is…", paragraphs of *why* a step exists | **Docs say WHAT to do, not WHY.** Cut the rationale, keep the instruction. (The rare exception: a non-obvious rule that a reader will *violate* without understanding it — there, one clause of "why," not a paragraph.) |
| **Partially redundant sections** | Two sections that restate the same rule; a procedure that repeats a caveat already stated once globally; a worked example whose lesson already appears as a guardrail | Keep the single strongest statement; cut the rest. |
| **Flowery language** | Hedging ("it might be worth considering whether…"), throat-clearing intros/conclusions, emphatic adverbs, metaphor, rhetorical questions | Replace with the direct statement. The reader wants the rule, not the mood. |

**Condensing rules:**
- **Preserve every load-bearing technical fact:** exact commands, flag names, paths,
  thresholds, IDs, version numbers, the literal strings to grep for. These are the point
  of the doc. Condense the prose *around* them, never the facts themselves.
- **Preserve the doc's frontmatter** (the `---` YAML block) and top-level structure
  conventions — edit the body, not the metadata.
- **Keep code blocks and tables** unless their content is fully redundant with prose
  elsewhere (then cut the weaker of the two — usually the prose restating the table).
- **A bullet is tighter than a paragraph.** A list of conditions reads faster than the
  same conditions in sentences. Convert where it loses nothing.
- **When in doubt, cut and let the supervisor restore.** Asymmetry: an over-cut paragraph
  is a 30-second restore from the backup; an under-cut doc is a wasted dispatch.

---

## 3. Return — percent shortened + overview

For **each** doc you edited, report back to the supervisor in this shape:

```
DOCS-DESLOP — <doc-path>

Reduction: <X%> shorter   (before: <B> words/lines → after: <A>)
Backup:    /Users/benjaminfeuer/Documents/slop_docs/<stamp>_<filename>

Reorganized:
  - <1-line: what moved where, e.g. "merged three scattered 'Gotchas' mentions into one §Gotchas">
  - <1-line: concept reorders, e.g. "moved 'When to use' above 'What it does'">

Condensed / removed:
  - <1-line per category of cut, e.g. "dropped the fixed-bug history in §3 (now default behavior)">
  - "removed rationale paragraphs from the launch procedure (kept the steps)"
  - "tightened flowery intros in §1, §4"

Flagged (NOT edited — for supervisor decision):
  - <any gap you noticed but did NOT author, e.g. "§Eval references a flag --foo not defined anywhere; may need authoring">
```

The **percent-shortened** is the headline number (compute it as
`round((before − after) / before × 100)` on whichever unit — words or lines — is cleaner;
state which). The **overview** is a short list of *what categories of slop were removed*
and *what was reorganized* — enough for the supervisor to spot-check one or two cuts
against the backup without re-reading the whole doc.

If a doc came back **little changed** (< ~10% shorter), say so explicitly — that doc was
already tight, and the supervisor should know the pass found little to cut rather than
assume it was thorough. If a doc came back **much shorter** (> ~50%), flag it for closer
review — aggressive cuts warrant a second look.

---

## Guardrails

- **CONDENSE AND CLARIFY ONLY. Never add or elaborate.** This is the whole license and the
  whole constraint. You move, merge, tighten, and cut. You do not write new content. When
  you spot a genuine gap, put it in the "Flagged" section of your return — do not fill it.
- **The backup is mandatory and first.** No edit before the dated `slop_docs/` copy lands.
  The aggressive condensing in §2 is only safe because §0 ran.
- **Order by concept, not chronology** — unless the doc is inherently chronological (a
  runbook, a sequence, a ladder). When in doubt, the reader's lookup task wins over the
  author's narrative.
- **Preserve load-bearing facts.** Commands, flags, paths, thresholds, IDs, literal grep
  strings, version pins — these are non-negotiable. Condense prose, never facts.
- **One doc at a time.** Back up → reorganize → condense → report, then move to the next.
  Don't batch edits across docs (a half-edited doc is hard to reason about).
- **Don't touch frontmatter or the repo's structural conventions** (`---` YAML, `# <name>`
  H1, the `When to use` / `Gotchas` / `Guardrails` taxonomy). Edit the body within the
  existing skeleton; restructure subsections, not the doc type.
- **Over-condense, don't under-condense.** The asymmetry is deliberate: the backup makes
  over-cuts cheap to reverse; under-cuts waste the dispatch.
