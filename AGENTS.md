# AGENTS.md — panelary

## Prime directive

**This project's first customer is AgenticFinance.**

Its job right now is to build software — SDKs, APIs, MCP servers, harnesses — that the
AgenticFinance assessment engine actually calls. Becoming an independent product comes
later, and only on the strength of what that work proves.

Panelary is the quantitative engine behind **Quantitative & Research Infrastructure** (STRATEGY §7). It is already the most mature asset here — 3,007 tests, four independent leak-safety layers — and the engine does not import it at all yet.

## The rule that keeps this honest

> **No new public surface without a named caller in the engine, and a test in the engine
> that exercises it.**

A beautiful SDK nobody calls is the failure mode. If the engine does not need it yet, it is
not the next thing to build. This is also what makes an eventual spin-off credible: the
interface arrives documented, tested, and exercised by a caller that notices when it breaks.

## Why AgenticFinance is a good first customer

It is demanding in exactly the ways that make software good: point-in-time correctness,
provenance on every value, deterministic re-runs, sealed ground truth, and disclosed
affiliation. An interface built to satisfy that is stronger than one built to a guess about
what the market wants.

## What to build first

**`CompileResult.to_json()` so a leakage finding can become report evidence; pipeline-level `audit()` so a whole feature set sweeps in one call; an as-of join and calendar-aware embargo (both are claimed by STRATEGY §7 and do not exist — there is no bitemporal concept anywhere in the package).**

Consumed by: `truepoint/quant/` — leakage findings become `Evidence(kind=STATIC_ANALYSIS)` in an assessment report
Caller: `truepoint/src/truepoint/quant/`

## Non-negotiables

- Never funnel. Where this project's capability appears in an assessment recommendation, it
  is disclosed as affiliated and sits alongside alternatives we do not sell (STRATEGY §3).
- Never expose sealed ground-truth questions, seeds or answer keys (STRATEGY §21).
- The customer never needs to know this project's name to buy an assessment (STRATEGY §28).

Full context: `../STRATEGY.md` (§30 is the amendment this file implements) and `../CLAUDE.md`.
