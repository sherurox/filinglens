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
