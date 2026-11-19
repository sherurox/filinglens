# Experiments

Every number here comes from a committed file in `bench/` (named next to it) and the command recorded inside that file. Test splits are run once per final setting; settings are chosen on dev.

## FL-M2: labels, tables and extraction

### Deviation from the spec: silver labels (decided 19 Nov 2026)

The spec asks for a hand-labeled gold set and a human test-retest kappa. This milestone uses **silver labels** instead: the 330 sampled risk-factor paragraphs were labeled by a local open-weight model following the committed guide (`docs/LABELING_GUIDE.md` v1.0, condensed in `docs/labeling/silver_prompt.md`). Consequences:

- `gold/risk_paragraphs.*.jsonl` is a silver set, not a gold set.
- The agreement number below is between two models, not a human's reliability.
- Extraction F1 against these labels measures agreement with the silver labeler, not correctness.
- The table ground truth is not affected: it comes from the HTML each page image is rendered from.

### Silver labels and labeler agreement

Labelers: `ibm-granite/granite-4.1-3b` (FP8, primary) and `Qwen/Qwen3-4B-AWQ` (check), greedy decoding constrained to a JSON schema (`bench/silver_ibm-granite-granite-4-1-3b.json`, `bench/silver_qwen-qwen3-4b-awq.json`).

| | Granite 4.1 3B | Qwen3-4B-AWQ |
|---|---|---|
| Answers following the rules | 327 / 330 | 328 / 330 |
| Severity 2 share (of valid answers) | 268 / 327 | 285 / 328 |
| Time for 330 paragraphs, incl. model load | 91.1 s | 310.1 s (incl. download) |
| Peak GPU memory (display included) | 7,695 MiB | 7,759 MiB |

Silver set (`gold/MANIFEST.json`): 327 paragraphs kept (3 dropped: the primary answer broke the rules), split by company into 191 train / 71 dev / 65 test.

Agreement between the two labelers on the 326 paragraphs both answered validly (`bench/kappa.json`, 2,000 bootstrap resamples):

| Field | Observed agreement | Cohen's kappa | 95% CI |
|---|---|---|---|
| Category (11 classes) | 0.733 | 0.697 | 0.644–0.747 |
| Severity (1–3) | 0.813 | 0.303 | 0.168–0.422 |
| Severity, quadratic weights | 0.813 | 0.343 | 0.216–0.454 |
| First-seen year, quadratic weights | 0.785 | 0.781 | 0.695–0.852 |

Reading: category agreement is substantial; severity has high raw agreement but low kappa because both models answer "2" for most paragraphs (chance agreement is already high), so severity labels carry little information. The most frequent category disagreements are competition_demand vs people (10) and competition_demand vs operations_supply (9).

### Table extraction: cell accuracy on rendered 10-K tables

Pages: 50 real tables from 10-K HTML, rendered by Chromium at letter width (2x), one company per page, 10 dev / 40 test (`bench/table_pages.json`). Ground truth: each table's logical rows (label + numbers in reading order) from the same HTML. Metric (`src/filinglens/extract/table_metrics.py`): a true cell is (row, k-th number); rows are matched by label; cell accuracy = correct true cells / all true cells (micro), with precision = correct / predicted cells.

| Method | Dev accuracy | **Test accuracy (micro)** | Test macro | Test precision | Rows matched (test) | Source |
|---|---|---|---|---|---|---|
| Tesseract 5.5.0, psm 4 (chosen on dev over psm 6: 0.8224, psm 11: 0.8454) | 0.8684 | **0.7254** | 0.7534 | 0.8266 | 0.908 | `bench/tables_ocr_*_psm*.json` |
| Qwen3-VL-2B-Instruct (FP8), image only, prompt v2 | 0.6053 | **0.5351** | 0.5819 | 0.4938 | 0.618 | `bench/tables_vlm_image_*_p2.json` |
| Qwen3-VL-2B-Instruct (FP8), image + OCR rows as hint, prompt v2 | 0.4474 | **0.4619** | 0.4584 | 0.4040 | 0.562 | `bench/tables_vlm_image_ocr_*_p2.json` |

Test: 40 pages, 1,708 true cells. The VLM returned valid JSON for 38 of 40 test pages in both variants; 287.2 s (image) and 399.4 s (image+OCR) for 40 pages including model load; peak GPU memory 7,889 MiB.

Findings:
- On clean, rendered pages, plain OCR with a position-based row/cell grouping beats a 2B vision-language model by about 19 points of cell accuracy, and is far faster (CPU only; seconds per page are in the bench files).
- The VLM reads digits well; its errors are structural: emitting each cell as its own row, or adding the title and year headers as rows. Prompt v2 (what a row is, plus a worked example) addressed most of this on dev.
- Giving the VLM the OCR rows as a hint made it worse: it tends to copy OCR's mistakes and row breaks.
- Dev was a poor predictor of test for OCR (0.8684 vs 0.7254): 10 dev pages are too few. Settings were still chosen on dev only.

History (kept for honesty, not comparable with the table above):
- The first page draw (before a ground-truth fix for "—%" nil cells, which changed which tables qualified) gave OCR psm 4 dev 0.9457 / test 0.6884, and VLM prompt v1 dev 0.2287 (image) / 0.5116 (image+OCR) (`bench/tables_vlm_*_dev_p1_run.json` holds the v1 run logs).
- The ground-truth fix redrew the 50 pages; all numbers in the table above are on the current draw.

Limits: pages are rendered from HTML, so they are cleaner than scanned annual reports; one table per page; 2B model in FP8 (the vision encoder may be quantized too); single run per setting.

### Prompted extraction baseline (the bar M4 must beat)

Model: `Qwen/Qwen3-1.7B` (Apache-2.0, BF16), the base of M4's student. It gets the silver labeler's rules (`docs/labeling/silver_prompt.md`), each paragraph with its heading and earlier-year candidates, and the same JSON schema (`src/filinglens/extract/fields.py`). Answers that break the rules, and `not_a_risk` answers, count as wrong. Labels are silver, so every score is **agreement with the silver labeler (Granite 4.1 3B)**, not correctness.

Variant choice on dev (71 paragraphs), rule fixed in advance: the higher category macro-F1 runs on test once (`bench/fields_{zero,fewshot}_dev.json`).

| Variant (dev) | Category macro-F1 (95% CI) | Category accuracy | Severity macro-F1 | First-seen exact / within 1 year |
|---|---|---|---|---|
| **zero-shot (chosen)** | **0.5792** (0.401–0.679) | 0.6761 | 0.3618 | 0.4366 / 0.5493 |
| few-shot (one train example per category) | 0.5075 (0.340–0.600) | 0.5493 | 0.4285 | 0.6338 / 0.7746 |

**Test** (65 paragraphs, zero-shot, run once; `bench/fields_zero_test.json`):

| Rows | Rule-valid | **Category macro-F1** (95% CI) | Category accuracy | Severity macro-F1 | First-seen exact / within 1 year |
|---|---|---|---|---|---|
| All test rows | 64 / 65 | **0.6295** (0.474–0.712) | 0.6615 | 0.2832 | 0.4308 / 0.5846 |
| Rows where both silver labelers agree on the category | 47 / 48 | 0.6001 (0.464–0.666) | 0.7292 | 0.2667 | 0.3542 / 0.5417 |

Runs (`bench/fields_*_run.json`): 45.7 s for the 65 test paragraphs including model load; peak GPU memory 7,799 MiB (display included).

Findings:
- The few-shot examples improved severity and first-seen year on dev but lowered category F1, so the pre-set rule chose zero-shot; M4 can revisit examples per field.
- Severity is weak for every model here (labeler kappa 0.303, baseline macro-F1 0.28): the labels themselves carry little signal.
- On the agreed subset, accuracy rises (0.729) while macro-F1 falls (0.600): with 48 rows and 11 classes, a few rare classes move macro-F1 a lot; the wide CIs say the same.
- These are the numbers M4's fine-tuned student must beat, on the same test split and the same silver labels.

### Entity resolution: company names to CIKs (D4)

Deviation from the spec: the spec asks for precision on 100 *hand-checked* pairs. Here the check is automatic, against SEC's own identifier. Each query is a (CIK, name) pair a filer used in FSDS `sub.txt` (2018q1–2026q3); the resolver sees only the name, matches it against current names in `company_tickers.json` (fetched 19 Nov 2026: 7,993 CIKs), and is right only if it returns the filer's own CIK. Filers that are not listed today stay in the queries: for them any match is a false merge.

Resolver (`src/filinglens/extract/entities.py`): normalized name (punctuation, state tags such as `/DE/` and legal forms removed) → compact key (spaces removed) → fuzzy IDF-weighted token Jaccard over candidates that share a rare token (in ≤ 200 reference names). A key shared by two CIKs returns no match.

Setting: companies split in half by `sha256('20261119:<cik>')`; threshold (0.50–0.95) and margin (0–0.2) chosen on the tune half by F0.5, then the eval half scored once (`bench/entity_eval.json`).

| Half | Queries | True company listed | Matched | Correct | Precision | Recall | F0.5 |
|---|---|---|---|---|---|---|---|
| Tune (chosen: threshold 0.95, margin 0.2) | 7,376 | 4,111 | 3,262 | 3,225 | 0.9887 | 0.7845 | 0.9397 |
| **Eval** | 7,524 | 4,240 | 3,390 | 3,343 | **0.9861** | 0.7884 | 0.939 |

**D4** (`bench/entity_precision.json`, pairs in `bench/entity_pairs_100.csv`): 100 matches drawn at random from the eval half: **97 / 100 correct, precision 0.97 (Wilson 95% CI 0.916–0.990)**. All 100 are exact-name matches. A second sample that leaves out exact-name matches has only 2 pairs (both correct), because the chosen setting makes almost no non-exact matches.

Eval matches by method: exact 3,341 / 3,388 correct (0.9861, CI 0.9816–0.9896); compact 1 / 1; fuzzy 1 / 1.

Findings:
- **The tune half limited fuzzy matching to near-identical names.** F0.5 picked the strictest corner of the grid (threshold 0.95, margin 0.2). Below a score of 0.9 the best candidate is usually the wrong company (tune half: 0.8–0.9: 8 of 21 right; 0.7–0.8: 26 of 75; 0.6–0.7: 27 of 94; 0.5–0.6: 33 of 189); at 0.9–1.0 it was right 4 of 4. The resolver is in effect exact-name-or-nothing on this task (1 fuzzy match on the eval half).
- **Misses are mostly renames, not blocking.** Of 880 tune-half listed companies whose name did not match exactly, only 237 share any token with the current name (II-VI → Coherent); 197 of those 237 are reachable through a rare token. The low pair completeness on the eval half (0.2328) is driven by renames.
- **Errors (47 on eval, all exact-name):** 44 are filers not listed today whose normalized name equals a listed company's. Most are the same company under a successor SEC registrant (new CIK after a holding-company formation, merger or redomicile: Walt Disney, Caesars, Spectrum Brands, Noble), which a check by CIK counts as wrong. The rest are true name collisions (Trinity Capital Corp vs Trinity Capital Inc., United Bancshares /PA vs /OH, Regency Centers LP vs Corp). So the precision above is strict for "same registrant" and a lower bound for "same economic company".
- Blocking compared 188,542 pairs instead of 33,003,097 (reduction ratio 0.9943) for the 4,129 eval queries without an exact match.

Master table: `gold/company_master.csv` (300 companies: CIK, current name, tickers, exchanges, SIC, sector, EIN, state of incorporation, fiscal year end, former names, FSDS names; 181 have former names, 48 have several tickers). Dedup report: `bench/entities.json` (ticker file: 1,433 of 7,993 CIKs have several tickers, 8 normalized names belong to several CIKs; FSDS: 14,900 (CIK, name) pairs from 12,078 filers, 2,207 filers used more than one normalized name).
