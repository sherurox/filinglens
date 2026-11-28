# Analyst-crew rubric (FL-M6, Done-when D1)

Written and committed before the crew has produced any answer. It fixes how every answer to the analyst questions (`gold/agents/questions.{dev,test}.jsonl`, made by `filinglens.agents.questions`) is scored. The test run (30 questions) is scored once, under this rubric and the rule file `bench/m6_crew_rule.json` committed before it.

## What is scored

The crew's final output for one question: the answer text, its citations (each a filing URL plus the excerpt it points to), the figures it states, and its status (`answered` or `blocked`). A **claim** is one sentence of the answer text. A **figure** is any number in the answer except fiscal years, dates, and citation markers. "FY*y*" in this rubric and in the question files means the fiscal year that ended in calendar year *y* (the year of the 10-K's report date); the question text says "the fiscal year that ended in *y*" for the same reason, because a company whose year ends in late January calls that year by the previous number.

## The five items (each pass / fail)

| Item | How scored | Passes when |
|---|---|---|
| **R1 Figures correct** | automatic | Every reference figure and derived value the question asks for appears in the answer and matches its reference (numbers checker rule: the stated precision and scale; either the last-reported or the first-reported value), and no other figure in the answer fails the numbers checker. For a figure the question's reference lists as **missing**, the answer states no value for it. |
| **R2 Every claim cited** | automatic | Every claim carries at least one citation; every cited URL is a filing URL of an excerpt the crew actually retrieved for this question; a claim about a company's FY *y* 10-K cites that company's FY *y* filing. |
| **R3 Covers every part** | by hand | Each part of the question is addressed: each figure asked for, and the risk-theme part for each company and year named. For a missing figure, saying it is unavailable counts as addressing it. |
| **R4 Faithful to the excerpts** | by hand | What the answer says about the risk theme is supported by the excerpts it cites, read in full. A "change" claim (new, expanded, reduced, unchanged) must match the two years' excerpts. The question's `risk_evidence` (the 10-K chunks that match the theme, and how many) is the scorer's aid, not a required citation list. |
| **R5 No unsupported claims** | by hand | Nothing in the answer comes from outside the excerpts and figures: no outside facts, forecasts, advice or speculation presented as fact. |

**Task success** for a question = R1 and R2 and R3 and R4 and R5 all pass.

**What R1's "asks for" means, per template** (fixed 28 Nov 2026, before any dev answer was scored and before the test run): `metric_margin_risk` asks for revenue, operating income and the operating margin; `peer_compare` for the two net margins; `growth` for the revenue growth; `risk_change_metric` for the change, which may be stated as the change itself (growth, or margin points) or as both years' values (revenue in both years, or the margin in both years). Figures a question stores only as inputs (e.g. the revenue behind a margin) are not required.

## Blocked answers and unanswerable questions

- **Answerable question, answer blocked** (the numbers checker or a guardrail stopped it): task failure. It is counted separately as `blocked`, because a block is safer than a wrong figure; the report gives both numbers.
- **Unanswerable question** (`answerable: false`, template `missing_data`): success when the answer says the missing figure is unavailable and states no value for it (R1, R3), and any part that is answerable (the risk-theme part, when its 10-K exists) passes R2, R4 and R5. A block that names the missing figure also counts as success; a block for any other reason does not.

## How the hand items are scored

1. The scoring script writes one sheet per run (`bench/agent_score_<split>.csv`): question, answer, status, every cited excerpt in full, the automatic R1/R2 results, and the question's `risk_evidence`.
2. The user scores R3, R4 and R5 as 1 or 0 for every row, with a one-line note for every 0. Each row is scored once; a score is not changed after the totals are seen.
3. Only the sheet's 0/1 values and notes are read back; the script computes the totals.

## What is reported

- Task success: count and share of the questions, with a Wilson 95% interval; separately for answerable and unanswerable questions.
- Pass rate of each item R1–R5, and the number of blocked answers.
- The same by template.
- Everything from the committed run files; no number is typed in by hand.

## Known limits of this rubric

- R4 and R5 are one person's judgment, scored once, without a second rater.
- The spec's rule is strict: every figure in an answer must match `fct_financials`, or the answer is blocked. So a number quoted from excerpt text (say, a percentage of revenue from one customer) also blocks the answer, and the writer is told to state no number except the figures it is given.
- The questions come from templates, so their wording is more uniform than real analyst questions.
