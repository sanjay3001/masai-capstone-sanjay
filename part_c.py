import json
import random
import re
import time
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = (
    "CLASSIFY the customer support ticket into one of: billing, delivery, technical, fraud.\n"
    'Reply with JSON only: {"category": "...", "reason": "..."}'
)

FEW_SHOT_PROMPT = (
    "CLASSIFY the customer support ticket into one of: billing, delivery, technical, fraud.\n"
    'Reply with JSON only: {"category": "...", "reason": "..."}\n'
    "EXAMPLES:\n"
    'User: Someone hacked my account and used my wallet balance. This is fraud, contact [EMAIL].\n'
    'Assistant: {"category": "fraud", "reason": "unauthorized account access"}\n'
    'User: The app crashes every time I open the cart.\n'
    'Assistant: {"category": "technical", "reason": "app crash"}'
)

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
segments = pd.read_csv("outputs/customer_segments.csv")
seg_map = segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

# ---------------- TASK 2: REDACT PII ----------------
email_cnt = phone_cnt = order_cnt = 0
redacted_texts = []

for txt in tickets["text"]:
    txt, c1 = EMAIL_RE.subn("[EMAIL]", txt)
    txt, c2 = PHONE_RE.subn("[PHONE]", txt)
    txt, c3 = ORDER_RE.subn("[ORDER_ID]", txt)
    email_cnt += c1; phone_cnt += c2; order_cnt += c3
    redacted_texts.append(txt)

tickets["redacted_text"] = redacted_texts
print(f"Redactions -> Email: {email_cnt}, Phone: {phone_cnt}, Order ID: {order_cnt}, Total: {email_cnt+phone_cnt+order_cnt}")

for r_txt in tickets["redacted_text"]:
    assert not EMAIL_RE.search(r_txt) and not PHONE_RE.search(r_txt) and not ORDER_RE.search(r_txt)
print("Verified: 0 PII matches left in redacted texts.")

# ---------------- TASK 4 & 5: RESILIENT CALLS & VALIDATION ----------------
def call_with_retry(ticket_id, messages, rng):
    for n in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if n == 3:
                raise
            wait = BASE_DELAY * (2 ** n) * rng.uniform(0.9, 1.15)
            print(f"[RETRY] ticket_id={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

def validate_reply(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in ALLOWED_CATS:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except Exception:
        words = re.findall(r"[a-z]+", (content or "").lower())
        present = [c for c in ALLOWED_CATS if c in words]
        if len( set(present) ) == 1:
            return "classified_keyword", present[0]
        return "flagged_invalid_output", ""

def run_pipeline(system_prompt, token_budget=None):
    mock_llm.reset()
    rng = random.Random(42)
    running_tokens = 0
    rows = []

    for _, row in tickets.iterrows():
        tid, cid, raw_txt, red_txt, true_cat = row["ticket_id"], row["customer_id"], row["text"], row["redacted_text"], row["true_category"]
        s_info = seg_map.get(cid, {"segment_name": "", "is_premium": False})
        seg_name = s_info["segment_name"]
        is_prem = bool(s_info["is_premium"])

        if not raw_txt.strip():
            status, cat, tok = "skipped_empty", "", 0
        elif mock_llm.est_tokens(raw_txt) > MAX_INPUT_TOKENS:
            status, cat, tok = "flagged_long", "", 0
        elif token_budget is not None and running_tokens >= token_budget:
            status, cat, tok = "budget_exhausted", "", 0
        else:
            assert not EMAIL_RE.search(red_txt) and not PHONE_RE.search(red_txt) and not ORDER_RE.search(red_txt)
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red_txt}]
            resp = call_with_retry(tid, msgs, rng)
            tok = resp.usage.total_tokens
            running_tokens += tok
            status, cat = validate_reply(resp.choices[0].message.content)

        if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            route, prio = "human_review", "NORMAL"
        elif cat == "fraud":
            route, prio = "security_team", "HIGH"
        elif is_prem:
            route, prio = "premium_desk", "HIGH"
        else:
            route, prio = "support_queue", "NORMAL"

        rows.append({
            "ticket_id": tid, "customer_id": cid, "status": status, "category": cat,
            "segment_name": seg_name, "priority": prio, "route": route, "tokens_used": tok,
            "true_category": true_cat
        })
    return pd.DataFrame(rows)

# ---------------- TASK 6: COMPARE ZERO-SHOT AND FEW-SHOT ----------------
print("\n--- Zero-Shot Run ---")
df_zs = run_pipeline(ZERO_SHOT_PROMPT, token_budget=None)
zs_clf = df_zs[df_zs["status"].isin(["classified_json", "classified_keyword"])]
zs_acc = (zs_clf["category"] == zs_clf["true_category"]).mean() * 100
print("Status counts:\n", df_zs["status"].value_counts().to_dict())
print(f"Classified: {len(zs_clf)} | Accuracy: {zs_acc:.2f}% | Total Tokens: {df_zs['tokens_used'].sum()} | Tokens/Classified: {df_zs['tokens_used'].sum()/len(zs_clf):.2f}")
print("Zero-Shot Confusion Table:\n", pd.crosstab(zs_clf["true_category"], zs_clf["category"]))

print("\n--- Few-Shot Run (Final Triage Results) ---")
df_fs = run_pipeline(FEW_SHOT_PROMPT, token_budget=100000)
fs_clf = df_fs[df_fs["status"].isin(["classified_json", "classified_keyword"])]
fs_acc = (fs_clf["category"] == fs_clf["true_category"]).mean() * 100
print(f"Classified: {len(fs_clf)} | Accuracy: {fs_acc:.2f}% | Total Tokens: {df_fs['tokens_used'].sum()} | Tokens/Classified: {df_fs['tokens_used'].sum()/len(fs_clf):.2f}")

out_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
df_fs[out_cols].to_csv("outputs/triage_results.csv", index=False)

# ---------------- TASK 7: BUDGET DEMO ----------------
df_budget = run_pipeline(ZERO_SHOT_PROMPT, token_budget=6000)
df_budget[out_cols].to_csv("outputs/triage_budget_demo.csv", index=False)
print("\n--- Budget Demo (Budget=6000) Status Counts ---\n", df_budget["status"].value_counts().to_dict())
