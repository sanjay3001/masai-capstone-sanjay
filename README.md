# Overview

The **Instamart Customer Intelligence & Support Copilot** is an end-to-end data engineering, machine learning, and generative AI system built for Swiggy Instamart across six Indian metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune). It solves three core operational problems:
1. **Data Foundation (Part A):** Audits and cleans messy raw exports into a validated SQLite analytical database (`data/instamart.db`).
2. **Customer Intelligence (Part B):** Segments customers using RFM + K-Means and predicts 30-day customer lapse using leak-free scikit-learn pipelines.
3. **Support Triage (Part C):** Masks customer PII, enforces input/output guardrails, handles transient LLM errors with exponential backoff, and routes tickets by customer segment.
4. **Policy RAG & Refund Agent (Part D):** Answers policy questions via TF-IDF retrieval and autonomously processes or escalates refunds via a stateful, resumable ReAct agent.

---

# Reproduce

Run these exact commands from a clean clone on branch `main`:

```bash
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. python generate_data.py
5. python part_a.py
6. python part_b.py
7. python part_c.py
8. python part_d.py
```

---

# Part A

### 1. Raw Data Audit

| Table | Shape (Rows, Cols) | Column | Data Type | Missing Count | Missing % |
|---|---|---|---|---|---|
| `customers.csv` | (1200, 5) | `customer_id` | string | 0 | 0.00% |
| `customers.csv` | (1200, 5) | `city` | string | 0 | 0.00% |
| `customers.csv` | (1200, 5) | `signup_date` | string | 0 | 0.00% |
| `customers.csv` | (1200, 5) | `age` | string | 61 | 5.08% |
| `customers.csv` | (1200, 5) | `gender` | string | 30 | 2.50% |
| `products.csv` | (40, 4) | `product_id`, `product_name`, `category`, `price_inr` | string / int64 | 0 | 0.00% |
| `orders.csv` | (13084, 9) | `order_id`, `customer_id`, `product_id`, `quantity`, `order_ts`, `status`, `discount_pct`, `order_amount` | string / int64 | 0 | 0.00% |
| `orders.csv` | (13084, 9) | `delivery_minutes` | string | 2541 | 19.42% |

#### Specific Audit Figures
| Audit Metric | Count |
|---|---|
| Raw `orders.csv` rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with status `pending` | 656 |
| Distinct `order_id` values | 12,309 |
| Non-numeric `order_amount` entries (`N/A`, `error`) | 120 |
| Negative `delivery_minutes` entries (`-10`) | 106 |
| Blank `delivery_minutes` entries | 2,541 |
| Blank `age` entries in `customers.csv` | 61 |
| `age` values outside 10–100 (`-4, 0, 150, 200, 280, 999`) | 6 |
| Distinct raw `city` spellings | 24 |
| Distinct non-blank raw `gender` spellings | 10 |
| Blank `gender` entries | 30 |

### 2. Documented Cleaning Rules & Impact
* **Customers Cleaning:**
  1. **City Standardisation:** Stripped leading/trailing whitespace and applied title case (`.str.strip().str.title()`), collapsing 24 raw spellings into 6 cities (changed 504 values): Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
  2. **Gender Mapping:** Mapped 10 non-blank spellings to `male` (560) and `female` (610), and mapped 30 blank entries to `unknown` (30).
  3. **Age Imputation:** Converted 6 impossible ages outside 10–100 to missing (`NaN`), then imputed all 67 missing ages (61 blank + 6 impossible) using the median age of the customer's city (`groupby("city")["age"].transform("median")`). Result: 0 missing ages; all ages lie between 18 and 60.
  4. **Signup Date:** Converted all 1,200 `signup_date` strings to datetime objects.
* **Orders Cleaning (in exact execution order):**
  1. **Timestamp Conversion:** Converted all 13,084 `order_ts` values to datetime.
  2. **Exact Deduplication:** Removed 119 exact duplicate rows, leaving 12,965 rows.
  3. **Status Deduplication:** Sorted by `order_ts` and kept the latest record per `order_id`, removing 656 earlier `pending` rows and leaving 12,309 unique orders (11,110 `completed`, 1,199 `cancelled`, 0 `pending`).
  4. **Order Amount Repair:** Coerced 119 remaining unparseable `order_amount` strings to `NaN` and recalculated them exactly using `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Total completed revenue is **INR 2,998,668.80** with 0 missing amounts.
  5. **Delivery Minutes Imputation:** Converted 98 negative `delivery_minutes` values among unique completed orders to `NaN`, then imputed all 768 missing completed delivery durations (670 blank + 98 negative) with the product category median among completed orders. Left all 1,199 `cancelled` orders with missing (`NaN`) delivery times.

### 3. SQL Analysis Results (`sql/queries.sql`)
* **Q1 (WHERE then HAVING):** **30 customers** have total completed spend exceeding INR 8,000 across orders with `order_amount > 500`.
* **Q2 (Three-Table Join):** **30 rows** (6 cities × 5 categories) summing to **INR 2,998,668.80**. Largest cell: **Bengaluru / Electronics at INR 233,909.45**.
* **Q3 (Scalar Subquery):** **11 products** are priced above the average catalogue price of INR 314.88.
* **Q4 (LEFT JOIN):** **25 customers** have no completed orders.
* **Q5 (CTE):** **550 customers** have total completed spend above the customer average of **INR 2,552.06**.
* **Q6 (Ranking Window Functions):** Top customer per city:
  * **Bengaluru:** `C0082` (INR 12,856.20)
  * **Chennai:** `C0930` (INR 10,151.40)
  * **Delhi:** `C0291` (INR 13,742.60)
  * **Hyderabad:** `C0743` (INR 12,730.05)
  * **Mumbai:** `C0693` (INR 14,766.55)
  * **Pune:** `C0426` (INR 10,377.10)
  * Exactly **8 rows** have `RANK()` different from `ROW_NUMBER()` because tied customers with identical `total_spend` in the same city receive the same `RANK()`, whereas `ROW_NUMBER()` assigns distinct sequential integers arbitrarily across ties.
* **Q7 (Running Total):** **183 daily rows** (2025-04-01 to 2025-09-30); final cumulative revenue is **INR 2,998,668.80**.

### 4. Pandas Cross-Check & Exploratory Analysis

#### City-Level Completed Order Summary
| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 460,453.75 | 257.09 | 1,791 |
| Delhi | 540,948.95 | 280.14 | 1,931 |
| Hyderabad | 495,529.00 | 272.27 | 1,820 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 499,035.20 | 242.13 | 2,061 |
| **Grand Total** | **2,998,668.80** | **269.91** | **11,110** |

#### Order Amount Distribution & Correlation Summary
* **Completed `order_amount` Statistics:** Mean = **INR 269.91**, Median = **INR 135.00**, Skewness = **4.61**, Q1 = **INR 64.80**, Q3 = **INR 297.00**, IQR = **INR 232.20**, Lower Fence = **INR −283.50** (0 outliers below), Upper Fence = **INR 645.30** (**897 outliers** above).
* **EDA Narrative (148 words):** This dataset captures six months of Instamart orders across 1,200 customers and 40 products spanning Grocery, Dairy, Snacks, Personal Care, and Electronics. Completed order amounts are heavily right-skewed (mean INR 269.91 vs. median INR 135.00; skewness 4.61) because most everyday baskets consist of low-cost groceries and dairy, whereas occasional Electronics orders reach up to INR 3,998.00. Using the 1.5×IQR rule (upper fence INR 645.30), there are 897 high-value outliers above the fence and zero below; these represent legitimate premium purchases rather than data errors after cleaning. Across numeric variables, linear relationships are very weak: the pair with the largest correlation in absolute value is `order_amount` and `discount_pct` at **−0.0953** (followed by `order_amount` and `quantity` at **+0.0870**), arising mechanically because higher percentage discounts reduce net billed amounts. Importantly, a correlation of this size does not establish causation.

---

# Part B

### 1. RFM Summary (Snapshot `2025-10-01 00:00:00`)
* **Cohort Size:** 1,175 customers with ≥ 1 completed order.
* **Recency (days):** Median = **19**, Min = **0**, Max = **180**.
* **Frequency (orders):** Mean = **9.46**, Max = **33**.
* **Monetary (INR):** Mean = **INR 2,552.06**, Max = **INR 14,766.55**.

### 2. K-Means Elbow & Silhouette Analysis

| K | Inertia | Silhouette Score |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

* **Selection of K=4:** We select **K=4** because it produces a sharp elbow reduction in inertia (dropping from 1,268.2 at K=3 to 743.1 at K=4), achieves the global maximum silhouette score (**0.526**), and separates customers into four clear, actionable business personas. Inertia measures within-cluster sum of squared distances, so adding more centroids at K=8 mathematically guarantees a smaller or equal distance to the nearest centroid than at K=2 (monotonically decreasing to zero at K=N). Therefore, lower inertia alone cannot pick K without causing over-segmentation into noisy micro-clusters.

### 3. Customer Segment Profiles (K=4) & PCA Projection
*Note: Cluster numbers (`segment_id`) are arbitrary mathematical labels with no intrinsic ordering.*

| Segment ID | Business Name | Size | Mean Recency (Days) | Mean Frequency | Mean Monetary (INR) | `is_premium` | Recommended Action |
|---|---|---|---|---|---|---|---|
| 0 | Active Occasional Shoppers | 436 | 21.52 | 3.74 | 909.61 | `False` | Send bundle cross-sell nudges to increase basket size and order frequency. |
| 1 | High-Value Big-Ticket Buyers | 138 | 24.43 | 6.80 | 7,130.20 | `True` | Offer VIP priority support and early access to electronics/personal-care deals. |
| 2 | Loyal Everyday Shoppers | 419 | 8.76 | 19.35 | 3,365.45 | `False` | Enrol in a loyalty subscription pass to lock in high habitual grocery frequency. |
| 3 | Dormant / Churned Shoppers | 182 | 112.91 | 2.38 | 684.61 | `False` | Trigger automated win-back discount coupons before permanent attrition. |

* **PCA Variance Explained:** `PC1 = 0.634`, `PC2 = 0.218` (**Total = 0.852** or 85.2% of total RFM variance).

### 4. Lapse Prediction Setup & Leakage Prevention
* **Cohort (`< 2025-09-01`):** **1,156 customers**; **443 lapsed** (**38.32%**); split into **924 training** and **232 test** rows.
* **Leakage Prevention Guarantee:** No encoder, scaler, or resampler ever saw the test rows; no order data from `2025-09-01` onwards entered the features; and the whole-window segment labels from Task 3 were excluded from the feature matrix.
* **Scaler Mean Comparison:** Learned training mean of `frequency` is **8.0076** vs. full-cohort pre-cutoff mean of **7.9593**. They differ because the `StandardScaler` was fitted strictly on the 924 training rows rather than the full 1,156-customer cohort, preventing test-set distribution leakage.

### 5. Model Tuning, Test Evaluation & Class-Balance Experiment

| Model | Best Hyperparameters | Best 5-Fold CV F1 | Test Accuracy | Test Precision (Lapsed) | Test Recall (Lapsed) | Test F1 (Lapsed) | Test Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|---|
| Majority Baseline ("Not Lapsed") | None | — | 61.64% | 0.0000 | 0.0000 | 0.0000 | `[[143, 0], [89, 0]]` |
| Tuned Logistic Regression | `C = 1` | 0.6724 | 71.98% | 0.6364 | 0.6292 | 0.6328 | `[[111, 32], [33, 56]]` |
| **Tuned Random Forest (Selected)** | `max_depth=8, min_samples_leaf=10` | **0.6926** | **73.71%** | **0.6591** | **0.6517** | **0.6554** | `[[113, 30], [31, 58]]` |
| Over-Sampled Logistic Regression (570:570) | `C = 1` (Balanced Train) | — | 71.12% | 0.5902 | 0.8090 | 0.6825 | `[[93, 50], [17, 72]]` |

* **Model Selection Justification:** We select the tuned **Random Forest (`max_depth=8, min_samples_leaf=10`)** as our production model because it achieves both a higher cross-validated F1 (0.6926 vs. 0.6724) and a superior test F1 (**0.6554** vs. 0.6328). In retail retention, missing a truly lapsing customer (a false negative) forfeits their entire future lifetime value, whereas wrongly targeting a loyal customer (a false positive) only incurs the modest cost of a promotional coupon or notification. Random Forest captures non-linear interactions between recency, frequency, and tenure, identifying more true lapsers while maintaining higher precision than Logistic Regression.
* **Class-Balance Experiment Interpretation:** Over-sampling the minority class in training caused **Recall** to move the most—jumping from **0.6292 to 0.8090** (+17.98 percentage points)—while **Precision** stayed near 0.60 (dipping slightly from 0.6364 to 0.5902). For the retention team, this means the balanced model catches 81% of lapsing customers (72 of 89) at the cost of sending retention offers to 18 additional loyal customers.

### 6. Feature Importance, Coefficients & Retention List
* **Top 5 Random Forest Features (`feature_importances_`):**
  1. `frequency` (**0.3688**): Habitual order count is the strongest protective signal against churn.
  2. `recency` (**0.2285**): Longer gaps since the last completed order strongly signal disengagement.
  3. `monetary` (**0.1456**): Higher cumulative spend indicates deeper platform adoption.
  4. `avg_delivery_minutes` (**0.0682**): Slower deliveries moderately increase churn risk.
  5. `tenure_days` (**0.0641**): Longer customer tenure stabilises ordering behaviour.
* **Logistic Regression Coefficient Signs:**
  * `frequency` (**−1.3412**, negative): More frequent past orders strongly decrease the log-odds of lapsing.
  * `recency` (**+0.3915**, positive): More days elapsed since the last order increase the likelihood of lapsing.
  * `monetary` (**−0.1842**, negative): Higher historical spend reduces the probability of lapsing.
* **Top 10 Premium Test Customers (`outputs/retention_list.csv`):** Out of the top 10 highest-risk Premium customers in the test set, **8 truly lapsed**. *(Note: Because the `is_premium` flag comes from the whole-window RFM table of Task 3, which includes September orders, this retention list is illustrative and not a leakage-free evaluation.)*

---

# Part C

### 1. Input Guardrails & PII Redaction
* **Guardrail Counts:** Out of 164 raw tickets, **4 empty tickets** (`T0157`–`T0160`) were marked `skipped_empty` and **4 long tickets** (`T0161`–`T0164`, each estimating to **734 tokens** > 400 limit) were marked `flagged_long`. Neither group triggered a model call.
* **PII Redaction Counts:** Across all 164 tickets, regex redaction replaced **55 emails**, **91 phone numbers**, and **132 order IDs**, for a total of **278 substitutions**. Re-running all three regex patterns on the redacted texts returned **0 matches**.
* **Retry Resilience:** Across the 156 eligible tickets, the pipeline encountered exactly **19 `TransientLLMError` exceptions** (every 9th call), all recovered on the first retry (`retry=0`) with backoff delays inside `[0.0450s, 0.0575s]`.
* **Validation Counts (156 eligible tickets):** **135 `classified_json`** (valid JSON with allowed category), **12 `classified_keyword`** (chatty non-JSON replies rescued via `re.findall(r"[a-z]+", text.lower())`), and **9 `flagged_invalid_output`** (valid JSON with disallowed category `"shipping"`, routed to `human_review`).

### 2. Zero-Shot vs. Few-Shot Comparison

| Metric | Zero-Shot Prompt | Few-Shot Prompt |
|---|---|---|
| Tickets Classified (`json` + `keyword`) | 147 | 147 |
| Accuracy vs. `true_category` | **85.71%** (126 / 147) | **100.00%** (147 / 147) |
| Total Tokens Used | 9,534 | 18,426 |
| Tokens per Classified Ticket | **64.86** | **125.35** |

#### Zero-Shot Confusion Table (147 Classified Tickets)
| True \ Predicted | `billing` | `delivery` | `fraud` | `technical` |
|---|---|---|---|---|
| **`billing`** | 37 | 0 | 0 | 0 |
| **`delivery`** | 0 | 36 | 0 | 0 |
| **`fraud`** | 11 | 0 | 26 | 0 |
| **`technical`** | 10 | 0 | 0 | 27 |

* **When Extra Few-Shot Tokens Are Worth Paying For:** In the zero-shot run, the model prioritizes `"billing"` whenever general payment words (`charged`, `payment`, `order`) appear alongside fraud or technical keywords, misclassifying **11 fraud tickets** and **10 technical tickets** as billing (85.71% accuracy). Switching to the few-shot prompt adds **60.49 tokens per classified ticket** (rising from 64.86 to 125.35 tokens, or +8,892 total tokens across the batch), but raises accuracy to **100.00%** and catches all 11 missed fraud cases. Because missed fraud tickets fail to reach `security_team` within the mandatory 2-hour SLA—exposing Instamart to financial loss and regulatory risk—paying an extra ~60 tokens per ticket is overwhelmingly justified in production.

### 3. Token Budget Demo & Final Routing
* **Token Budget Demo (`outputs/triage_budget_demo.csv`):** Running the zero-shot pipeline with `TOKEN_BUDGET = 6000` exhausted the budget after 99 eligible model calls (6,021 cumulative tokens), marking the remaining **57 eligible tickets as `budget_exhausted`** and routing them safely to `human_review`.
* **Final Routing (`outputs/triage_results.csv`, Few-Shot with `TOKEN_BUDGET = 100000`):** All 164 tickets were processed: **17 routed to `human_review`** (4 `skipped_empty` + 4 `flagged_long` + 9 `flagged_invalid_output`, all rule-validated without guessing), **37 routed to `security_team`** (`HIGH` priority fraud), **16 routed to `premium_desk`** (`HIGH` priority non-fraud tickets from Premium customers), and **94 routed to `support_queue`** (`NORMAL` priority).

---

# Part D

### 1. Policy Chunking, TF-IDF Retrieval & RAG Evaluation
* **Chunk Counts (40 words, 10-word overlap):** **15 chunks total** — `refund_policy.txt`: 4, `delivery_policy.txt`: 3, `escalation_policy.txt`: 3, `payment_policy.txt`: 3, `privacy_policy.txt`: 2.
* **TF-IDF Explanation:** Term Frequency (TF) measures how frequently a term appears within a specific chunk, while Inverse Document Frequency (IDF) down-weights common words that appear across many chunks so rare, discriminative words carry higher weight. A key limitation of TF-IDF for policy retrieval is that it relies on exact lexical overlap and cannot capture semantic synonyms (for example, a query asking about *"reimbursement for slow shipping"* will fail to match *"refund for late delivery"*).

#### RAG Evaluation Table (`policy_eval.csv`)
| Chunking Configuration | Answerable: `expected_doc` in Top-3 | Answerable: `expected_fact` in Answer | Unanswerable: Fallback (`I could not find...`) |
|---|---|---|---|
| **40 words, 10-word overlap** | **12 / 12** | **12 / 12** | **2 / 2** |
| **40 words, 0-word overlap** | **12 / 12** | **10 / 12** | **2 / 2** |

* **Why Zero Overlap Breaks:** With `overlap=0`, the question *"Who must approve a refund above INR 1000?"* (requiring `"human support lead"`) and *"What is the delivery fee on small orders?"* (requiring `"INR 25"`) fail because the 40-word boundary cuts right through the middle of those policy sentences. For example, in `refund_policy.txt`, chunk 2 ends with *"A refund above INR 1000 must be approved by"* (words 40–79) and chunk 3 begins with *"a human support lead before money is returned"* (words 80–106), so neither chunk contains both the question keywords and the complete answer fact.

### 2. Refund Agent Scenarios (`outputs/agent_runs.jsonl`)

| Scenario ID | Customer ID | Status | Refund Amount (INR) | Model Calls | Tokens Used |
|---|---|---|---|---|---|
| **S1** | `C0004` | `REFUND` | 128.25 | 5 | 875 |
| **S2** | `C0025` | `REFUND` | 135.00 | 5 | 875 |
| **S3** | `C0003` | `ESCALATE` | 1999.00 | 5 | 882 |
| **S4** | `C0001` | `DENIED` | 0.00 | 4 | 645 |
| **S5** | `C0002` | `DENIED` | 0.00 | 3 | 428 |
| **S6** | `C0001` | `NOT_FOUND` | 0.00 | 2 | 126 |
| **S7** | `C0004` | `FALLBACK_MAX_ITERATIONS` | 0.00 | 6 | 3,162 |

### 3. Safe Stopping & Crash-and-Resume Demonstrations
* **Stop-Safely Demonstration Statuses:**
  * Running `S1` with `token_limit=200`: **`FALLBACK_TOKEN_LIMIT`** (stopped immediately after model call 2 when cumulative tokens exceeded 200, without executing the requested tool).
  * Running `S1` with `time_limit_seconds=0`: **`FALLBACK_TIMEOUT`** (stopped at the top of iteration 1 before any model call was made).
* **Crash-and-Resume Demonstration (`S1`):**
  * Simulated crash raised immediately after step 2 checkpoint (`state/S1_crash_demo.json`) was saved, then resumed via `resume_agent("S1_crash_demo")`.
  * **Tools executed in order across crash and resume:** `["get_order", "search_policy", "calculate_refund", "issue_refund"]` (4 total executions, 0 repeated tools).
  * **Final Resumed Outcome:** Status `REFUND` with **INR 128.25** (`REFUND: INR 128.25 issued.`).

### 4. ReAct vs. Plan-and-Execute Comparison (212 words)
In our ReAct refund agent, the model reasons step-by-step after each tool observation, requiring **5 model calls (875 tokens)** for a standard late-order refund (`S1`/`S2`), **4 calls (645 tokens)** for an on-time denial (`S4`), **3 calls (428 tokens)** for a cancelled order (`S5`), and **2 calls (126 tokens)** when an order ID does not exist (`S6`). By contrast, a **Plan-and-Execute** architecture asks the LLM once upfront to emit a complete multi-step plan (e.g., `get_order -> search_policy -> calculate_refund -> issue_refund`) and executes those steps deterministically in code, needing only **2 model calls** (1 planner call + 1 final synthesizer call, ~300 tokens) on the happy path. However, Plan-and-Execute handles unexpected observations poorly: when `get_order` returns `NOT_FOUND` (`S6`) or `status="cancelled"` (`S5`), the pre-baked plan becomes invalid and requires exception-handling logic or an expensive replanning LLM call. ReAct naturally adapts its trajectory on the very next turn (`S6` exits immediately on call 2; `S5` exits on call 3 without ever calling `calculate_refund`). For this financial refund workflow, **I would choose ReAct** (or a deterministic state machine with tool guardrails) because branching depends entirely on runtime database state (`status`, `delivery_minutes`, and the INR 1,000 escalation threshold), and step-by-step verification prevents premature or unauthorized refund execution.

---

# Limitations and Next Steps

1. **Post-Hoc Premium Label in Retention List:** The `is_premium` flag used to filter `outputs/retention_list.csv` in Part B Task 8 comes from the whole-window RFM table (Part B Task 3), which includes the September 2025 window that defines the lapse label; therefore, the retention list is illustrative and not a leakage-free evaluation. In production, segmentation must be recomputed strictly prior to the prediction cutoff date.
2. **Lexical RAG Limitations:** Part D's TF-IDF retriever relies on exact word overlap and fixed 40-word windows. Next steps include upgrading to sentence-aware chunking with dense semantic embeddings (or hybrid BM25 + vector search) to handle paraphrased customer queries.
3. **Rule-Based Guardrails vs. Semantic Guardrails:** Part C's regex redaction catches standard Indian 10-digit mobile numbers, emails, and `ORD-xxxxxx` IDs, but would miss spelled-out phone numbers or street addresses. Adding a lightweight NER model before LLM ingestion will harden privacy protection.
