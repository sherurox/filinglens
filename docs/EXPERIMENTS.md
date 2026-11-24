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

## FL-M4: fine-tune and distill

Student `Qwen/Qwen3-1.7B` (Apache-2.0, the FL-M2 baseline's base), teacher `Qwen/Qwen3-8B-AWQ` (Apache-2.0) on Kaggle. Labels are still **silver** (Granite 4.1 3B), so every score below is agreement with the silver labeler, not correctness. Settings are chosen on dev (71 rows); test (65 rows) is scored once, in the final comparison.

### Deviations from the spec

- **"Gold only" arm → "silver only":** there is no gold data (FL-M2 decision). The silver arm is the 191 train rows. Test labels are also Granite's, so the silver arm is trained toward the labeler it is scored against.
- **Sequence length 4,096, not ≤ 2,048:** the system prompt alone is 1,098 tokens; examples have median 1,758, p95 2,321, max 4,039 tokens. At 2,048 a large share of examples would be dropped (the trainer drops over-long examples instead of cutting the answer). Liger's fused linear cross-entropy makes 4,096 fit in 8 GB (below).
- **transformers `Trainer` + PEFT, not TRL's `SFTTrainer`:** TRL truncates over-long examples from the end (that would cut the answer, the only part with loss) and its token-accuracy metric needs full logits, which the fused loss never builds. Completion-only masking is ~20 tested lines in `train/sft.py`.
- **Deterministic evaluation** (`--max-num-seqs 1`) for every FL-M4 comparison, base model included; see "Batch-dependent nondeterminism" below. FL-M2's numbers were batched.

### Teacher labels on Kaggle

Kaggle notebook, 2× T4, one vLLM 0.34.0 process per GPU (data parallel; no tensor parallelism: the model fits one T4 and the T4s have no NVLink), AWQ kernel in fp16 (Turing has no bf16 or Marlin), greedy decoding, thinking off, the silver labeler's prompt, schema and rules, top-20 logprobs at the category step (`bench/teacher_run/`, `bench/teacher_smoke.json`).

| | Rows | Rule-valid | Notes |
|---|---|---|---|
| Smoke (one T4) | 200 pool rows | 193 | 0.57 rows/s after a 143 s engine start; KV cache 38,912 tokens |
| Pool (training inputs) | 7,200 | 7,037 | 166 `not_a_risk` → 6,871 valid risks; first token consistent 7,189; 3,600 rows per GPU in 6,396.4 s and 6,697.6 s |
| Dev | 71 | 67 | 4 risks without `first_seen_year` |
| Test | 65 | 64 | scored only in the final comparison |

Wall time 7,098.4 s including installs. Pool inputs (`train/teacher_inputs.py`, `bench/teacher_inputs.json`): 7,200 rows = 240 per sector × year cell, written round-robin so every prefix is balanced; no text shared with a dev or test company and no silver-sample rows (leakage 0 / 0 / 0).

**Teacher vs silver on dev** (Kaggle run, batched decoding): category macro-F1 0.663 (0.46–0.737), accuracy 0.7042.

### Filtering the teacher labels (`bench/teacher_filter.json`)

Rule fixed before any filtered result was seen, applied in order: rule-valid → not `not_a_risk` → first token consistent → chosen-category confidence ≥ τ = 0.9 (fixed, not tuned on 71 dev rows) → keyword heuristic for the guide's precedence rules (pandemic, climate, tariffs/sanctions/war, cyber) doesn't disagree (it abstains when 0 or ≥ 2 patterns fire).

| Kept | rule_invalid | not_a_risk | inconsistent_first_token | low_confidence | heuristic_disagrees |
|---|---|---|---|---|---|
| **6,259 / 7,200** | 163 | 166 | 11 | 515 | 86 |

Confidence of the 6,871 valid risks: < 0.5: 26; 0.5–0.9: 499; 0.9–0.99: 444; 0.99–0.999: 483; 0.999–0.9999: 425; ≥ 0.9999: 4,994. So 525 are below 0.9 (515 of them reach the confidence step; the rule is applied in order), but most labels sit near 1: greedy decoding with a long rule prompt makes the teacher very peaked, which is not calibration.

On dev, agreement with silver among rows kept by τ: 0.746 (τ 0) → 0.790 (τ 0.9, 62 rows) → 0.793 (τ 0.99, 58 rows). The heuristic fired on 11 dev rows and agreed with silver on 9.

SFT arms (`bench/sft_data.json`, data version `c22a1742…`): silver 191, teacher 6,259, both 6,450.

### Training setup and memory (D2)

QLoRA on Qwen3-1.7B: NF4 with double quantization, bf16 compute, LoRA on all linear projections (alpha = 2 × rank, dropout 0.05), batch 1 × gradient accumulation 8, cosine schedule with 5% warm-up, AdamW, gradient checkpointing, max length 4,096, seed 0. VRAM capped at free-at-start − 300 MiB (`torch.cuda.set_per_process_memory_fraction`), because on WSL the driver's sysmem fallback otherwise spills into system RAM silently (the NVIDIA "Prefer No Sysmem Fallback" setting did not stop it).

| Measurement | Result | Source |
|---|---|---|
| Smoke, batch 1 × 1,024 random tokens | 4,691 MiB allocated / 5,216 MiB reserved, 945 tok/s | `bench/train_smoke.json` |
| Smoke, 2,048 tokens, no cap | spilled to system RAM (484 tok/s); kept as `_spill`, not a VRAM figure | `bench/train_smoke_*` |
| Smoke, 2,048 tokens, capped, no Liger | OutOfMemoryError (1.16 GiB under a 6,699 MiB cap): the cap works | `bench/train_smoke_*` |
| Liger fused linear CE, 2,048 / 4,096 tokens | 3,090 / 3,434 MiB at 752 tok/s; 4,065 / 4,330 MiB at 675 tok/s; loss equal to the non-Liger run to 3 decimals | `bench/train_smoke_*liger*.json` |
| Silver arm, 30 steps | loss 2.565 → 0.4775; peak 3,995 / 4,204 MiB; 18.14 s per step | `bench/sft_silver-r16-lr0.0002-s30.json` |
| Silver arm, 2 epochs | 48 steps, mean loss 0.7057, 986 s; peak 3,995 / 4,204 MiB; max 63.6 °C | `bench/sft_silver-r16-lr0.0002-e2.json` |
| Teacher arm, 1,500 rows, 1 epoch | 188 steps, mean loss 0.3605, 3,757.8 s; peak 3,551 / 4,180 MiB; max 64.6 °C | `bench/sft_teacher-n1500-r16-lr0.0002-e1.json` |
| Final retrain, all 6,259 teacher rows, 1 epoch (not shipped, see below) | 783 steps, mean loss 0.2868, 13,245.0 s (16.92 s per step); peak 3,960 / 4,118 MiB; max 68.8 °C | `bench/sft_teacher-r16-lr0.0002-e1.json` |

Peak VRAM for every FL-M4 training run stayed under 4.2 GiB reserved on the 8 GB card with 4,096-token examples. Per-step loss, learning rate, gradient norm, GPU temperature and reserved VRAM are in MLflow (experiment `fl-m4-sft`; the shipped student is run `2b109c9f…`, the full-data run `246addd1…`) and, for runs from 23 Nov on, also in each run's `steps.jsonl`.

### Batch-dependent nondeterminism in evaluation

The base model on the same 71 dev prompts gave category macro-F1 0.5792 (FL-M2, 19 Nov) and 0.6255 (21 Nov), with only 33 of 71 raw outputs identical. Both runs were greedy. vLLM sizes its KV cache from the GPU memory free at start (224 vs 96 blocks in two runs), so requests are batched differently and bf16 rounding differs, which flips some greedy choices. With `--max-num-seqs 1` (one sequence at a time), two base runs gave **71 / 71 identical** outputs (about 158 s each). Every FL-M4 comparison uses `--gpu-memory-utilization 0.84 --max-model-len 4608 --max-num-seqs 1`. On dev, this variation (up to 0.05 macro-F1) is as large as the differences between training settings.

### Dev results (deterministic eval, 71 rows)

| Model | Category macro-F1 (95% CI) | Accuracy | Rule-valid | Severity macro-F1 | First-seen exact / ±1 year |
|---|---|---|---|---|---|
| Base (prompted) | 0.6437 (0.45–0.745) | 0.7324 | 69 / 71 | 0.368 | 0.493 / 0.6056 |
| Silver arm, r16, lr 2e-4, 2 epochs | 0.6404 (0.478–0.729) | 0.7465 | 71 / 71 | 0.3437 | 0.7465 / 0.8028 |
| **Teacher arm, 1,500 rows**, r16, lr 2e-4, 1 epoch | **0.7414** (0.523–0.821) | 0.7606 | 71 / 71 | 0.2969 | 0.7746 / 0.8592 |
| Both arms (191 silver + 1,500 teacher rows), r16, lr 2e-4 | 0.6951 (0.474–0.768) | 0.7042 | 71 / 71 | 0.2969 | 0.7324 / 0.8169 |
| Teacher arm, 1,500 rows, r8, lr 2e-4 | 0.725 (0.509–0.802) | 0.7465 | 71 / 71 | 0.2969 | 0.7324 / 0.8169 |
| Teacher arm, 1,500 rows, r16, lr 1e-4 | 0.7414 (0.523–0.821) | 0.7606 | 71 / 71 | 0.2969 | 0.7606 / 0.831 |
| Teacher arm, all 6,259 rows, r16, lr 2e-4 | 0.6411 (0.434–0.712) | 0.6901 | 71 / 71 | 0.294 | 0.7887 / 0.8732 |

Paired bootstrap (same 71 rows resampled for both models, 2,000 resamples; `bench/compare_dev_*.json`; "–" = in the bench file, not copied here):

| B vs A | Macro-F1 difference (95% CI) | Share of resamples where B is not better | Rows only A / only B gets right |
|---|---|---|---|
| Teacher-n1500 vs base | +0.0976 (−0.0439, 0.1945) | 0.111 | 5 / 7 |
| Silver-e2 vs base | −0.0034 (−0.1097, 0.1247) | 0.4485 | 5 / 6 |
| Teacher-n1500 vs silver-e2 | +0.101 (−0.0561, 0.1767) | 0.182 | 4 / 5 |
| Both-n1500 vs base | +0.0514 (−0.1079, 0.1616) | 0.3385 | – |
| Teacher-n1500 vs both-n1500 | +0.0463 (−0.0626, 0.1577) | 0.2155 | 2 / 6 |
| Teacher-n1500 r8 vs base | +0.0813 (−0.0615, 0.1818) | 0.159 | – |
| Teacher-n1500 lr 1e-4 vs lr 2e-4 | 0.0 | – | 0 / 0 |
| Teacher all rows vs base | −0.0027 (−0.1434, 0.1148) | 0.6115 | – |
| Teacher all rows vs teacher-n1500 | −0.1003 (CI in the bench file) | – | 8 / 3 |

**Choice rule (fixed before the ablation results):** a difference inside the CI is a tie, and the simpler or default setting wins. No setting differs significantly from another or from the base model on dev. Teacher vs both is a tie, so the teacher arm wins as the simpler one (one label source, and not trained on the labeler it is scored against). Rank 8, rank 16 and lr 1e-4 vs 2e-4 tie; the default (r16, lr 2e-4) is kept. lr 1e-4 matching lr 2e-4 to the last digit was checked: the adapters differ (different hashes and first-seen scores), and both get exactly the same rows' categories right.

### More teacher data lowered dev agreement

At r16 / lr 2e-4, dev macro-F1 by teacher rows: 0 (base) 0.6437 → 400 0.7002 → 1,500 0.7414 → 6,259 0.6411. Agreement with the *teacher's* dev answers rises at the same time: base 52 / 71 (0.732), 1,500 rows 61 / 71 (0.859), 6,259 rows 64 / 71 (0.901). The more the student imitates the teacher, the more it inherits the teacher's disagreements with the silver labels (the teacher agrees with silver on about 70% of dev rows). The first-seen year, where the teacher is strong, gets better with more data (0.7746 → 0.7887 exact).

**Decision (24 Nov, before test was scored):** the shipped student is **teacher-n1500-r16-lr0.0002-e1**, the adapter with the highest dev score. That is model selection on dev; picking the best of several on 71 rows flatters its dev score, and test shows its real level. Scoring both on test and keeping the better one was rejected (that would be choosing on test).

### Hyperparameter search (Optuna)

Objective (decided 23 Nov, before any trial): **held-out loss**, the mean completion-token loss on teacher rows 1,500–1,799 (300 rows), which no trial trains on. Dev macro-F1 was rejected as the objective: 71 rows can't rank settings a few points apart, and 8 trials chosen on dev would overfit it, leaving nothing to confirm with. Dev is used only to confirm.

Setup (`train/optuna_search.py`, study `fl-m4-teacher-n400`): teacher arm, first 400 teacher rows per trial, 1 epoch; learning rate log-uniform 5e-5 to 5e-4, rank {8, 16}; TPE (seed 0, 4 random start-up trials; Optuna's default of 10 would make 8 trials pure random search); the ablation winner enqueued as trial 0; one subprocess per trial; study stored in SQLite (resumable).

| Trial | Rank | Learning rate | Held-out loss | Wall s |
|---|---|---|---|---|
| 0 (baseline) | 16 | 2e-4 | 0.33982 | 985.6 |
| 1 | 8 | 1.77e-4 | 0.36024 | 972.2 |
| 2 | 16 | 1.75e-4 | 0.34564 | 970.9 |
| 3 | 16 | 1.37e-4 | 0.35163 | 971.8 |
| **4 (best)** | 16 | 4.86e-4 | **0.32874** | 971.8 |
| 5 | 16 | 5.32e-5 | 0.3837 | 972.6 |
| 6 | 16 | 4.88e-4 | 0.33149 | 972.9 |
| 7 | 8 | 4.82e-4 | 0.3329 | 973.3 |

8 of 8 trials completed (`bench/optuna.csv`, `bench/optuna.json`, MLflow run `ba05b3ef…` in `fl-m4-optuna`). Held-out loss falls as the learning rate rises, up to the edge of the range; rank barely matters near 4.8e-4. **Decision rule (fixed before the search):** the best trial replaces the ablation winner only if the lower end of the 95% CI of its dev macro-F1 difference against the baseline trial (both trained on 400 rows) is above 0. Result: trial 4 vs trial 0 on dev, 0.7111 vs 0.7002, +0.0109 (−0.0476, 0.0763), not better in 36% of resamples; accuracy +0.0141 (−0.0423, 0.0845); 2 / 3 rows → **tie, default kept** (`bench/compare_dev_optuna-teacher-n400-t00-…_vs_…-t04-….json`).

Limits: trials train 50 steps on 400 rows and a short run rewards a large learning rate; the final models train 188 or 783 steps. The best trial sits at the edge of the range, so the search didn't find an optimum (widening it: LATER). Two earlier starts of the study failed after their first trial (held-out loss built full logits in eval mode; a missing bench folder); both bugs are fixed and the failed studies are kept under `data/interim/`.

### Final comparison on test (D1)

Shipped student: **teacher-n1500-r16-lr0.0002-e1** (1,500 filtered teacher rows, r16, lr 2e-4, 1 epoch, 188 steps). Test: 65 rows, scored once (`train/final_comparison.py`; `bench/m4_test_comparison.json`; MLflow run `6ade9c36…` in `fl-m4-compare`; test metrics also logged into the student's training run). Base and student use the deterministic setting; the teacher's answers are its Kaggle test predictions (batched fp16 on a T4), re-parsed with the same rules as the others (0 validity flags changed).

| Model | Category macro-F1 (95% CI) | Accuracy | JSON-valid / rule-valid | Severity macro-F1 | First-seen exact / ±1 year |
|---|---|---|---|---|---|
| Base, prompted (FL-M2, **batched** decoding) | 0.6295 (0.474–0.712) | 0.6615 | – / 64 of 65 | 0.2832 | 0.4308 / 0.5846 |
| Base, prompted (deterministic) | 0.6525 (0.494–0.734) | 0.6769 | 65 / 64 of 65 | 0.2832 | 0.4 / 0.5538 |
| Teacher, Qwen3-8B-AWQ | 0.783 (0.624–0.866) | 0.7692 | 65 / 64 of 65 | 0.337 | 0.8 / 0.8615 |
| **Fine-tuned student, Qwen3-1.7B + LoRA** | **0.7948** (0.632–0.869) | **0.7846** | 65 / **65** of 65 | 0.2931 | 0.7846 / 0.8769 |

On rows where both silver labelers agree: base 0.6663, teacher 0.9192, student 0.9024 macro-F1.

Paired bootstrap on test (2,000 resamples):

| B vs A | Macro-F1 difference (95% CI) | B not better | Accuracy difference (95% CI) | Rows only A / only B right |
|---|---|---|---|---|
| Student vs base | **+0.1423 (0.0276, 0.2621)** | 0.0065 | +0.1077 (0.0, 0.2154) | 3 / 10 |
| Student vs teacher | +0.0118 (−0.0513, 0.0876) | 0.3625 | +0.0154 (−0.0462, 0.0923) | 2 / 3 |
| Teacher vs base | +0.1305 (0.0024, 0.2678) | 0.0245 | +0.0923 (−0.0154, 0.2) | 4 / 10 |

Reading: the 1.7B student beats its prompted base model on category macro-F1 (CI above 0) and **ties its 8B teacher**. It is the only model with every answer rule-valid. JSON-valid is 65 / 65 for all three because decoding is constrained to the JSON schema; rule-valid is the stricter check. The dev result "student above teacher" (0.7414 vs 0.663, the teacher measured batched) does not hold up on test as a win; it is a tie. Decoding alone moves the base model on test from 0.6295 (batched) to 0.6525 (deterministic).

### Model registry (D3)

Registered model **`filinglens-risk-student` version 1** in the MLflow registry: the LoRA adapter logged by training run `2b109c9f…` (`mlflow-artifacts:/1/2b109c9f…/artifacts/adapter`: adapter config and weights, tokenizer files, chat template). Version tags: base model `Qwen/Qwen3-1.7B` (Apache-2.0), training config, data version, git SHA, dev and test macro-F1, the test comparison run ID. The registered copy was downloaded and its weights' SHA-256 matched the local adapter. The adapter was trained on the NF4 base and is served on the bf16 base in vLLM (standard QLoRA practice; a small mismatch).

### Reliability: an MLflow outage must not kill training (23 Nov)

The first run of the ablation batch died at step 8 of 212: the MLflow container shut down (graceful stop at 03:23:31 UTC on 22 Nov; the cause outside the project is not confirmed), the metrics callback spent 8 min 14 s in HTTP retries and then raised, and the Trainer stopped. Fix (`f5aa353`): every step's metrics go to a local `steps.jsonl` first; the first failed MLflow call switches the run to offline mode, and the missed rows are replayed at the end; the bench JSON records `mlflow_synced`. The batch runner checks MLflow's `/health` before each step. Live test (12 steps, MLflow stopped for 26 s): offline at step 4, synced at the end, run FINISHED with loss at all 12 steps in MLflow, 18.02 s per step (17.75 s without an outage).

### Limits

- Silver labels everywhere: scores are agreement with Granite 4.1 3B, and the teacher agrees with silver on only about 70% of dev rows.
- 71 dev and 65 test rows: CIs of about ±0.12 macro-F1; most differences between settings are ties.
- One seed per setting.
- The student never sees `not_a_risk` (all eval rows are risks); it can't reject non-risk text.
- Teacher confidence is very peaked (4,994 of 6,871 valid risks at ≥ 0.9999), so the τ = 0.9 filter removes a tail (525 rows below 0.9), not a large share; it is not calibrated.
- Optional spec item not done: DPO (LATER).
- The teacher's dev and test predictions came from batched decoding on a T4; base and student were scored deterministically.
- The shipped student was chosen on dev from several adapters; one Optuna search, ending at the edge of its range.

## FL-M3: retrieval, ranking and offline A/B

Retrieval runs over Item 1A risk-factor chunks of the latest 10-K per company, embedded with `BAAI/bge-base-en-v1.5` (MIT, revision `a5beb1e3`) and filtered to the company the question names (CIK filter). Queries are synthetic: one question per paragraph, written by `Qwen/Qwen3-4B-AWQ` with the FL-M2 prompt and checks, graded automatically (grade 2 = the source chunk, grade 1 = a chunk that covers most of the evidence). No human judged relevance. Companies are split by hash into ret-dev (182 queries, 77 companies) and ret-test (1,137 queries, 75 companies).

### Offline A/B: does the reranker ship? (D2)

**Arms.** A: dense top-10 (FAISS Flat, exact inner product, CIK filter). B: dense top-50, rescored by `BAAI/bge-reranker-base` (MIT, revision `2cfc18c9`), keep the top 10.

**Pre-registration.** The decision rule was written by `ab pilot` from ret-dev only and committed in `15b4381` (`bench/ab_rule.json`, fixed 2026-11-24 23:00:19 UTC) before any ret-test query was embedded. `ab test` refuses to run unless that file is committed and unchanged. The rule: use every ret-test query; ship the reranker if and only if the 95% company-cluster bootstrap CI of the per-query NDCG@10 difference (B − A) lies above 0 **and** the reranker's p95 latency on ret-test is at most 500 ms; otherwise keep dense-only.

**Power analysis** (`bench/ab_power.json`, ret-dev pilot, 182 queries). The per-query NDCG@10 difference has sd 0.2752. Detecting a 0.03 gain at two-sided α 0.05 with 80% power needs 663 independent queries (paired t-test, noncentral t). Queries are not independent: questions about the same company move together. Two estimates of the company ICC disagree: 0.1434 (one-way ANOVA) and 0.0297 (implied by the ratio of the cluster to the query bootstrap CI widths). With a design effect of 1 + (m − 1)·ICC, the required n has no solution once ICC ≥ companies / n_independent = 75 / 663; at 0.1434 it has none, so adding questions per company cannot reach 80% power on 75 companies. Instead ret-test was grown from 175 to 1,137 queries (about 15 per company, T10a) and the rule uses all of them. Expected power at that size: 0.5602 at the ANOVA ICC, 0.8694 at the implied ICC (normal approximation); a simulation of the full rule at 75 × 16 gave 0.586 at the ANOVA ICC.

**Result on ret-test** (`bench/ab_test.json`, 1,137 queries, 75 companies, 10,000 company-cluster resamples, seed 0):

| Metric @10 | Dense (A) | Reranked (B) | Mean difference | 95% CI (company cluster) | Effect size d_z | Better / worse / tied |
|---|---|---|---|---|---|---|
| **NDCG (primary)** | 0.4238 | 0.5125 | +0.0888 | 0.0669 – 0.1099 | 0.2854 | 553 / 289 / 295 |
| Recall | 0.6904 | 0.7872 | +0.0967 | 0.0728 – 0.1207 | 0.2491 | 146 / 36 / 955 |
| MRR | 0.4792 | 0.5919 | +0.1128 | 0.0833 – 0.1407 | 0.2639 | 379 / 173 / 585 |

For reference, the query-level bootstrap gives an NDCG@10 CI of 0.0703 – 0.1065; the cluster CI is wider, as it should be. Recall@50 of the dense candidates, the ceiling the reranker can reach, is 0.8725.

Reranker latency on ret-test (`bench/reranker_test.json`; reranking step only, 50 pairs per query, fp16 on the RTX 5070 Laptop GPU, one query at a time, 1,137 samples): p50 229.39 ms, **p95 246.63 ms**, p99 276.21 ms. Peak VRAM allocated 963 MiB.

**Decision: ship the reranker.** Both conditions hold: the CI's lower bound (0.0669) is above 0, and p95 (246.63 ms) is under the 500 ms budget. Why this is a sound call and not a lucky one:

- The CI was computed at the company level, the unit the queries cluster on, so it is not overstated by treating 15 questions about one company as 15 independent pieces of evidence.
- The rule, the metric, the budget and the test size were fixed and committed before the test split was scored, and the test split was scored once.
- The secondary metrics move the same way. They are descriptive only; the rule does not use them.
- Latency leaves about half the budget unused.

**Reading the result with care.**

- The test gain (+0.0888) is larger than the dev pilot's (+0.0512, cluster CI 0.0114 – 0.0927), though inside that CI. 962 of the 1,137 test queries come from the T10a batch, drawn from paragraphs the first batch had not used. A breakdown by batch was not pre-registered, so it is not reported here (`docs/LATER.md`).
- The test ICC is 0.0323, close to the implied dev estimate, so the ANOVA estimate overstated the clustering. No post-hoc power is reported: given the observed effect it is a restatement of the CI.
- Relevance is automatic. Each question is written from its source paragraph, so the labels measure finding that paragraph, not usefulness to an analyst; a human-judged set is needed to rule out a bias toward models that reward word overlap.
- 1,241 of 56,850 query–chunk pairs exceed 512 tokens and are truncated. fp16 vs fp32 logits differ by at most 0.00792 on the checked pairs.
- Latency is the reranking step alone on one laptop GPU with no concurrent load; it excludes the dense search and model loading.

MLflow (experiment `fl-m3-retrieval`): query embedding `284008b6c39b4e5789d113c22b0f0995`, reranking `203a3e4667f0473296640d29c0c2219a`, A/B test `c891ce0b874c47bd9cbbad239fd22375`.

Reproduce (the test steps need `--allow-test`; the last one also needs the committed rule):

```bash
uv run python -m filinglens.retrieval.ab pilot --mde 0.03 --latency-budget-ms 500
~/venvs/fl-train/bin/python -m filinglens.retrieval.embed queries --split test --allow-test --note "..."
uv run python -m filinglens.retrieval.rerank candidates --split test --allow-test
~/venvs/fl-train/bin/python -m filinglens.retrieval.rerank score --split test --allow-test --note "..."
uv run python -m filinglens.retrieval.ab test --allow-test --note "..."
```
