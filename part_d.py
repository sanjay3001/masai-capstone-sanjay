import glob
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# ---------------- TASK 1 & 2: CHUNK & VECTORISE ----------------
def build_chunks(chunk_size=40, overlap=10):
    step = chunk_size - overlap
    chunks = []
    for path in sorted(glob.glob("data/policies/*.txt")):
        src = os.path.basename(path)
        words = open(path, "r", encoding="utf-8").read().split()
        i = 0
        while True:
            end = i + chunk_size
            chunks.append({"source": src, "text": " ".join(words[i:end])})
            if end >= len(words):
                break
            i += step
    return chunks

CHUNKS = build_chunks(40, 10)
print(f"Total chunks (40w, 10 overlap): {len(CHUNKS)}")
print("Chunks per file:", pd.Series([c["source"] for c in CHUNKS]).value_counts().to_dict())

VEC = TfidfVectorizer().fit([c["text"] for c in CHUNKS])
CHUNK_MAT = VEC.transform([c["text"] for c in CHUNKS])

def retrieve(query, k=3, chunks=CHUNKS, vec=VEC, mat=CHUNK_MAT):
    q_vec = vec.transform([query])
    scores = cosine_similarity(q_vec, mat)[0]
    top_idx = scores.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "score": round(float(scores[i]), 4), "text": chunks[i]["text"]} for i in top_idx]

print("Top 3 for 'refund policy late delivery':", [r["source"] for r in retrieve("refund policy late delivery")])

# ---------------- TASK 3 & 4: AUGMENT, ANSWER & EVALUATE ----------------
def answer(question, chunks=CHUNKS, vec=VEC, mat=CHUNK_MAT):
    top = retrieve(question, k=3, chunks=chunks, vec=vec, mat=mat)
    ctx_lines = "\n".join(f"[{r['source']}]\n{r['text']}" for r in top)
    sys_msg = "You are a POLICY ASSISTANT. Answer only from the supplied context. If the answer is not in the context, say I could not find that information in the policy documents."
    usr_msg = f"CONTEXT:\n{ctx_lines}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}])
    return resp.choices[0].message.content, [r["source"] for r in top]

def evaluate_rag(chunk_size=40, overlap=10):
    ch = build_chunks(chunk_size, overlap)
    vc = TfidfVectorizer().fit([c["text"] for c in ch])
    mt = vc.transform([c["text"] for c in ch])
    eval_df = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hits = fact_hits = unans_hits = 0
    failures = []

    for _, r in eval_df.iterrows():
        q, exp_doc, exp_fact = r["question"], r["expected_doc"], r["expected_fact"]
        ans, srcs = answer(q, ch, vc, mt)
        if exp_doc != "NONE":
            d_ok = exp_doc in srcs
            f_ok = exp_fact in ans
            doc_hits += int(d_ok)
            fact_hits += int(f_ok)
            if not (d_ok and f_ok):
                failures.append((q, exp_fact, ans))
        else:
            unans_hits += int(ans.startswith("I could not find"))
    return doc_hits, fact_hits, unans_hits, failures

print("Eval (overlap=10):", evaluate_rag(40, 10)[:3])
d0, f0, u0, fails0 = evaluate_rag(40, 0)
print("Eval (overlap=0):", (d0, f0, u0), "| Failed questions:", fails0)

# ---------------- TASK 5: TOOLS & TOOL MAP ----------------
LEDGER_PATH = "outputs/refund_ledger.jsonl"
if os.path.exists(LEDGER_PATH):
    os.remove(LEDGER_PATH)

def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]), "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}

def _log_ledger(tool_name, order_id, refund_inr):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "tool": tool_name, "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}

def issue_refund(order_id, refund_inr):
    return _log_ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _log_ledger("escalate_to_human", order_id, refund_inr)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the SQLite database by order_id.", "parameters": {"order_id": "string"}},
    {"name": "search_policy", "description": "Retrieve top policy chunks matching the search query.", "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Compute eligible refund in INR from order_amount and delivery_minutes.", "parameters": {"order_amount": "number", "delivery_minutes": "number"}},
    {"name": "issue_refund", "description": "Approve and log an automatic refund up to INR 1000.", "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human", "description": "Escalate a refund above INR 1000 to a human support lead.", "parameters": {"order_id": "string", "refund_inr": "number"}}
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

# ---------------- TASK 6, 7 & 8: REACT LOOP, SAFE STOPS & RESUME ----------------
def _save_state(run_id, state_obj):
    with open(os.path.join("state", f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(state_obj, f, indent=2)

def _continue_loop(state_obj, max_iterations, token_limit, time_limit_seconds, start_mono, crash_after_step=None):
    messages = state_obj["messages"]
    steps = state_obj["steps"]
    tools_executed = state_obj["tools_executed"]
    llm_calls = state_obj["llm_calls"]
    tokens_used = state_obj["tokens_used"]
    run_id = state_obj["run_id"]

    while llm_calls < max_iterations:
        if (time.monotonic() - start_mono) >= time_limit_seconds:
            return {"scenario_id": state_obj["scenario_id"], "status": "FALLBACK_TIMEOUT",
                    "final_answer": "FALLBACK_TIMEOUT: Handing over to human agent due to timeout.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        state_obj["llm_calls"] = llm_calls
        state_obj["tokens_used"] = tokens_used

        if tokens_used > token_limit:
            return {"scenario_id": state_obj["scenario_id"], "status": "FALLBACK_TOKEN_LIMIT",
                    "final_answer": "FALLBACK_TOKEN_LIMIT: Handing over to human agent due to token limit.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}

        msg = resp.choices[0].message
        if not msg.tool_calls:
            ans = msg.content or ""
            status = ans.split(":")[0].strip() if ":" in ans else "UNKNOWN"
            return {"scenario_id": state_obj["scenario_id"], "status": status, "final_answer": ans,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}

        tc = msg.tool_calls[0]
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})

        step_num = len(steps) + 1
        steps.append({"step": step_num, "tool": tc.name, "arguments": tc.arguments, "observation": obs})
        tools_executed.append(tc.name)
        state_obj["step_number"] = step_num
        _save_state(run_id, state_obj)

        if crash_after_step is not None and step_num == crash_after_step:
            raise RuntimeError(f"Simulated crash after step {step_num}")

    return {"scenario_id": state_obj["scenario_id"], "status": "FALLBACK_MAX_ITERATIONS",
            "final_answer": "FALLBACK_MAX_ITERATIONS: Handing over to human agent after reaching 6 steps.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_step=None):
    start_mono = time.monotonic()
    run_id = run_id or scenario_id
    state_obj = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [{"role": "system", "content": "AGENT: Process customer refund requests using tools."}, {"role": "user", "content": user_message}],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": []
    }
    return _continue_loop(state_obj, max_iterations, token_limit, time_limit_seconds, start_mono, crash_after_step)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    start_mono = time.monotonic()
    with open(os.path.join("state", f"{run_id}.json"), "r", encoding="utf-8") as f:
        state_obj = json.load(f)
    return _continue_loop(state_obj, max_iterations, token_limit, time_limit_seconds, start_mono)

# Run 7 scenarios
scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
runs_path = "outputs/agent_runs.jsonl"
if os.path.exists(runs_path):
    os.remove(runs_path)

print("\n--- 7 Agent Scenarios ---")
for _, row in scenarios.iterrows():
    sid, msg = row["scenario_id"], row["message"]
    res = run_agent(msg, sid)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: res[k] for k in ["scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"]}) + "\n")
    print(f"{sid}: status={res['status']} | calls={res['llm_calls']} | tokens={res['tokens_used']} | ans={res['final_answer']}")

# Safety demos
s1_msg = scenarios.loc[0, "message"]
demo_tok = run_agent(s1_msg, "S1_token_demo", token_limit=200, run_id="S1_token_demo")
demo_time = run_agent(s1_msg, "S1_timeout_demo", time_limit_seconds=0, run_id="S1_timeout_demo")
print(f"\nSafety Demos -> token_limit=200: {demo_tok['status']} | time_limit_seconds=0: {demo_time['status']}")

# Crash and resume demo on S1
try:
    run_agent(s1_msg, "S1", run_id="S1_crash_demo", crash_after_step=2)
except RuntimeError as e:
    print(f"\nCaught expected crash: {e}")

resumed = resume_agent("S1_crash_demo")
print(f"Resumed S1 -> status={resumed['status']} | tools={resumed['tools_executed']} | ans={resumed['final_answer']}")
