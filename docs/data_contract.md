# FlowTrace — Data Contract

**Status:** Frozen 2026-10-16, after EDA Sections 1-6
**Dataset:** HI-Small (IBM AMLworld)
**Owner:** Priyam Srivastava
**Purpose:** Single source of truth for every decision made in EDA. Week 2+ modeling builds against this document.

---

## 1. Dataset

- **Source:** Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
- **Subset used:** HI-Small (`HI-Small_Trans.csv` + `HI-Small_Patterns.txt`)
- **License:** CDLA (Community Data License Agreement)
- **Raw size:** 5,078,345 transactions, 11 columns, 428 MB
- **Date range:** 2022-09-01 to 2022-09-18
- **Nulls:** 0
- **Full-row duplicates:** 9 (retained; 0.0002%)
- **Reference paper:** Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" (NeurIPS 2023)

Other subsets available but not used in Week 1-4: LI-Small, HI-Medium, LI-Medium. HI->LI distribution shift test reserved for Week 5.

---

## 2. Schema (canonical columns)

| Column | Dtype | Semantics |
|---|---|---|
| `ts` | Datetime (naive, treated as UTC per ingestion contract) | Transaction timestamp |
| `from_bank` | Int64 | Sender bank ID |
| `from_account` | String | Sender account ID (hex-encoded hash) |
| `to_bank` | Int64 | Receiver bank ID |
| `to_account` | String | Receiver account ID |
| `amount_paid` | Float64 | Amount sent (in `currency_paid`) |
| `currency_paid` | String | Sender currency |
| `amount_received` | Float64 | Amount received (in `currency_received`) |
| `currency_received` | String | Receiver currency |
| `payment_format` | String | ACH, Cheque, Credit Card, Cash, Bitcoin, Wire, Reinvestment |
| `is_laundering` | Int (0/1) | Ground truth label |

Added during tagging (Section 4):

| Column | Dtype | Semantics |
|---|---|---|
| `attempt_id` | Int64 | Pattern-file attempt ID (null for unmatched positives and negatives) |
| `typology` | String | Pattern typology name (null for unmatched positives and negatives) |
| `label_source` | String | `pattern` / `unmatched` / `none` |

**Account identity:** Composite `(bank, account)` pair. Account IDs can collide across banks.

**Timestamp assumption:** Naive dtype, treated as UTC per ingestion contract. Standard AMLworld usage; not independently verified from source.

---

## 3. Primary evaluation window

**Window:** `[2022-09-03, 2022-09-11)` — half-open, 8 days, 3,207,867 transactions, 3,792 positives.

**Why these dates:**
- Sept 11-18 excluded: 8 consecutive days with daily volume below 5% of peak volume. Interpreted as simulator shutdown.
- Sept 1-2 excluded: daily volume 2-5x the operating band. Interpreted as simulator warmup.
- Rule applied **without reference to the laundering label** (label-free criterion).
- Laundering-rate differences on excluded days (warmup 0.04%, shutdown 59%) observed post-hoc; not used in the exclusion decision.

**Sensitivity windows (for future experiments):**
- Narrow: `[2022-09-05, 2022-09-10)` — 5 days, 2,584,730 txns, steady-state only
- Permissive: `[2022-09-01, 2022-09-11)` — 10 days, 5,077,237 txns, includes warmup

---

## 4. Splits (frozen in `data/processed/split_manifest.yaml`)

| Split | Date range (half-open) | Rows | Positives | Positive rate |
|---|---|---|---|---|
| train | `[2022-09-03, 2022-09-08)` | 1,862,302 | 2,297 | 0.1233% |
| val | `[2022-09-08, 2022-09-09)` | 482,773 | 539 | 0.1116% |
| test | `[2022-09-09, 2022-09-11)` | 862,792 | 956 | 0.1108% |
| **Total** | `[2022-09-03, 2022-09-11)` | **3,207,867** | **3,792** | **0.1182%** |

**Integrity (verified in Section 3):**
- Half-open boundaries, zero timestamp overlap
- Train end timestamp belongs to val, not train
- Positive count sums exactly match the primary window

**Cold-start exposure (test set):**
- 34.4% of test accounts never appear in training data
- 89% of test positive accounts were never launderers in training

---

## 5. Label semantics

- `is_laundering` in {0, 1} — binary target
- `label_source`:
  - `pattern` (3,209 rows in full source dataset) — in `HI-Small_Patterns.txt`, has `typology` and `attempt_id`
  - `unmatched` (1,968 rows in full source dataset) — `is_laundering=1` but no pattern-file match. Investigation (Section 4) showed these differ substantially from pattern-tagged positives in payment-format distribution and have only 2.1% endpoint overlap with pattern attacks; underlying label-generation mechanism unknown. **No upgrade to "transitive"** without documented AMLworld support.
  - `none` (5,073,168 rows in full source dataset) — `is_laundering=0`

Counts above are for the full source dataset (5,078,345 rows). The frozen primary window contains 3,792 positives in total; the pattern/unmatched split within the primary window is a subset of these full-dataset counts.

**Evaluation metadata only. `label_source`, `typology`, `attempt_id` are NEVER model features** (direct target leakage).

---

## 6. Prediction task

**Rank observed transactions by laundering risk using information available when each transaction is scored. Evaluate temporal generalization using the frozen chronological splits. This is not forecasting future transactions, and performance on synthetic data does not establish real-world deployment performance.**

---

## 7. Evaluation protocol

**Primary unit:** transaction-level ranking. K is a transaction count.

**Primary metrics:**
- **Recall@K** at K = 50, 100, 500 (investigator alert budgets)
- **Precision@K** at the same K values
- **PR-AUC** (overall ranking quality under class imbalance)

**Secondary metrics:**
- Per-typology Recall@K (uses `label_source=="pattern"` + `typology` for stratification)
- Separate metrics on `label_source=="pattern"` vs `label_source=="unmatched"` subsets

**Test-set reference numbers:**
- 956 positive transactions in test
- At K=50: max possible Recall@50 = 50/956 ≈ 5.2%
- At K=100: max possible Recall@100 = 100/956 ≈ 10.5%
- At K=500: max possible Recall@500 = 500/956 ≈ 52.3%

K=500 is the most meaningful metric for this test set.

**Class imbalance handling:**
- Rebalancing is **not** applied in EDA or evaluation (preserve natural distribution).
- Model training may use `scale_pos_weight` (LightGBM) computed from the train split only.
- SMOTE and similar resampling techniques are not used.

**Account-level aggregation (future extension, not primary):** Score an account as max of its outgoing transaction scores within a 24h window; one alert per account per day. Not implemented in the first baseline.

---

## 8. Candidate features (all pending Week 2 ablation)

Each feature is tagged by leakage safety:
- **safe** — computed from the current transaction's own fields only
- **historical** — must be computed using only transactions preceding the one being scored
- **forbidden** — would leak the target; evaluation metadata only

| Feature | Safety | Source | Notes |
|---|---|---|---|
| `amount_paid` | safe | Section 5 | Positives have ~7x median vs negatives |
| `log_amount_paid` | safe | Section 5 | Log transform for LightGBM |
| `currency_paid` (one-hot) | safe | Section 5 | Saudi Riyal 4x baseline |
| `payment_format` (one-hot) | safe | Section 5 | ACH 6.8x baseline; carries 87% of positives |
| `hour_of_day` | safe | Section 3 | Business-hours concentration |
| `is_self_transfer` | safe | Section 6 | Self-transfers 11.7x less likely to be laundering |
| `distance_from_10k_usd` | safe | Section 5 | Exploratory candidate; 17 positives in the $9.5k-$10k USD interval. Threshold clustering is not established. |
| `is_weekend` | safe (ablation) | Section 3 | Based on 1 Saturday only — fragile |
| `day_of_week` | safe (ablation) | Section 3 | Same fragility caveat |
| `account_activity_count_historical` | historical | Section 6 | Count of txns before current ts; NOT full-window counts |
| `account_in_degree_historical`, `account_out_degree_historical` | historical | Week 3 | Graph features, Week 3 |
| `bank_degree`, `bank_pagerank`, etc. | historical | Week 3 | Graph features |
| `typology` | **forbidden** | Section 4 | Evaluation metadata only |
| `label_source` | **forbidden** | Section 4 | Evaluation metadata only |
| `attempt_id` | **forbidden** | Section 4 | Evaluation metadata only |

---

## 9. Known limitations

- Test window is 2 days (Sept 9-10); weekend effects rest on a single Saturday
- Warmup hypothesis for Sept 1-2 is circumstantial (volume anomaly only)
- UTC assumption is per ingestion contract; raw timestamps are naive
- Findings verified on HI-Small only; LI/Medium subsets have schema-level verification only
- The 1,968 unmatched positives have unknown label-generation mechanism
- Bank 70 asymmetry (99.6% outgoing) is observed; its role in the simulator is not independently documented
- All findings are associations in synthetic data; validation on test set required before any claims
- Zero observed rates (Wire, Reinvestment absence) are observations, not proofs of absence

---

## 10. Experiment registry

Planned experiments for Week 2+. Numbered for reference in commits and reports.

| ID | Experiment | Hypothesis | Metric |
|---|---|---|---|
| E01 | Rules baseline (ACH + high amount relative to its currency, with thresholds determined from training data + off-hours) | Simple rules beat random | Recall@500 |
| E02 | Logistic regression with safe features only | Linear model baseline | Recall@500, PR-AUC |
| E03 | LightGBM with safe features (no historical) | Non-linear gains over LogReg | Recall@500, PR-AUC |
| E04 | LightGBM + `account_activity_count_historical` | Historical features add value | Delta vs E03 |
| E05 | LightGBM ablation: without `payment_format` | ACH shortcut risk | Delta vs E04 |
| E06 | LightGBM ablation: without `is_weekend`, `day_of_week` | Weekend finding is fragile | Delta vs E04 |
| E07 | LightGBM + graph features (bank/account degree) | Graph structure adds signal | Delta vs E04 |
| E08 | Per-typology Recall@K breakdown | Model performance varies by typology | Table |
| E09 | Pattern vs unmatched Recall@K breakdown | Model behavior differs on the two label sources | Table |
| E10 | Train on HI-Small, test on LI-Small subset | Cross-subset generalization | Delta vs E04 (Week 5+) |

---

## 11. Files

| Path | Purpose |
|---|---|
| `configs/split.yaml` | Primary window + split boundaries |
| `data/processed/split_manifest.yaml` | Frozen row/positive counts per split |
| `data/processed/hi_small_tagged.parquet` | Main dataset + `attempt_id`, `typology`, `label_source` |
| `data/processed/hi_small_attempts.parquet` | 370 pattern-file attempts, one row each |
| `data/processed/hi_small_pattern_txns.parquet` | 3,209 pattern transactions, one row each |
| `src/flowtrace/data/schema.py` | Pandera schema + loader |
| `src/flowtrace/data/split.py` | `filter_split()` and bounds helpers |
| `notebooks/01_eda.ipynb` | Sections 1-6 EDA |
| `reports/figures/*.png` | All EDA plots |
| `docs/data_contract.md` | This file |

---

**End of contract.** Any change to this document is a decision event. Modeling work in `src/flowtrace/models/` must reference this contract as the source of truth.