# Project Notes

## Purpose

- This repo is CS336 assignment work. It exists for the user to learn by
  implementing things by hand, not for shipping a product.

## Code authorship (important)

- The user writes the implementation code. Do NOT edit, refactor, or "fix"
  source files unless the user explicitly asks the agent to modify code.
- Default agent role is diagnose and explain: identify the root cause, cite
  file paths and line numbers, and describe the fix in prose (a minimal
  illustrative snippet in the chat is fine). Then stop.
- Read-only investigation is always allowed: running tests, `git diff`, greps,
  reading files, timing experiments.
- Even when the fix looks obvious, ask before touching the code
  (e.g. "要我直接改吗？").
- Exception: agent-owned scratch files (notes, this `AGENTS.md`) may be edited
  when asked.

## Test organization

- Keep the course-provided test suite in `tests/` separate from tests added for
  this repository's own regressions and tooling.
- Put repository-authored tests in `project_tests/`. Do not add files under
  `tests/` unless the user explicitly asks to modify the course test suite.
