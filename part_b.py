import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.makedirs("outputs", exist_ok=True)

orders = pd.read_csv("data/clean/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
cust = pd.read_csv("data/clean/customers.csv")
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
prod = pd.read_csv("data/clean/products.csv")

# ---------------- TASK 1: BUILD RFM TABLE ----------------
comp = orders[orders["status"] == "completed"].copy()
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = comp.groupby("customer_id", as_index=False).agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
)
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]].sort_values("customer_id").reset_index(drop=True)

print(f"RFM customers: {len(rfm)} | Recency med={rfm['recency'].median()}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
print(f"Frequency mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()} | Monetary mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")

# ---------------- TASK 2: CLUSTER WITH K-MEANS ----------------
scaler_rfm = StandardScaler()
X_rfm = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

print("\n--- K-Means Clustering ---")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_rfm)
    sil = silhouette_score(X_rfm, km.labels_)
    print(f"K={k}: Inertia={km.inertia_:.1f}, Silhouette={sil:.3f}")

# Choose K=4
km4 = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_rfm)
rfm["segment_id"] = km4.labels_

# ---------------- TASK 3: PROFILE, NAME & PROJECT SEGMENTS ----------------
prof = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
)
print("\n--- Segment Profiles (K=4) ---\n", prof.round(2))

prem_seg = prof["mean_monetary"].idxmax()
freq_seg = prof["mean_frequency"].idxmax()
at_risk_seg = prof["mean_recency"].idxmax()
occ_seg = (set(range(4)) - {prem_seg, freq_seg, at_risk_seg}).pop()

name_map = {
    prem_seg: "High-Value Big-Ticket Buyers",
    freq_seg: "Loyal Everyday Shoppers",
    at_risk_seg: "Dormant / Churned Shoppers",
    occ_seg: "Active Occasional Shoppers"
}
rfm["segment_name"] = rfm["segment_id"].map(name_map)
rfm["is_premium"] = rfm["segment_id"] == prem_seg

pca = PCA(n_components=2, random_state=42).fit(X_rfm)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_.round(3)} | Total: {pca.explained_variance_ratio_.sum():.3f}")

rfm.to_csv("outputs/customer_segments.csv", index=False)

# ---------------- TASK 4: DEFINE LAPSE PROBLEM ----------------
cutoff = pd.Timestamp("2025-09-01 00:00:00")
ord_pre = orders[orders["order_ts"] < cutoff].merge(prod[["product_id", "category"]], on="product_id", how="left")
comp_pre = ord_pre[ord_pre["status"] == "completed"]
comp_sep = orders[(orders["status"] == "completed") & (orders["order_ts"] >= cutoff) & (orders["order_ts"] <= pd.Timestamp("2025-09-30 23:59:59"))]

cohort_ids = sorted(comp_pre["customer_id"].unique())
sep_active_ids = set(comp_sep["customer_id"].unique())

feat_comp = comp_pre.groupby("customer_id", as_index=False).agg(
    last_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean"),
    fav_category=("category", lambda s: s.value_counts().index[0])
)
feat_comp["recency"] = (cutoff - feat_comp["last_ts"]).dt.days

feat_all = ord_pre.groupby("customer_id", as_index=False).agg(
    total_orders=("order_id", "count"),
    canc_orders=("status", lambda s: (s == "cancelled").sum())
)
feat_all["cancellation_rate"] = feat_all["canc_orders"] / feat_all["total_orders"]

cohort = feat_comp.merge(feat_all[["customer_id", "cancellation_rate"]], on="customer_id").merge(cust, on="customer_id")
cohort["tenure_days"] = (cutoff - cohort["signup_date"]).dt.days
cohort["lapsed"] = (~cohort["customer_id"].isin(sep_active_ids)).astype(int)

num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
cat_cols = ["fav_category", "city", "gender"]
feature_cols = num_cols + cat_cols

assert cohort[feature_cols].isna().sum().sum() == 0, "Missing values found in features!"

X = cohort[feature_cols]
y = cohort["lapsed"]
X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
    X, y, cohort["customer_id"], test_size=0.2, stratify=y, random_state=42
)

y_base = np.zeros_like(y_test)
print(f"\nCohort: {len(cohort)} | Lapsed: {y.sum()} ({y.mean()*100:.2f}%) | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Baseline Accuracy: {accuracy_score(y_test, y_base)*100:.1f}%, Lapsed F1: {f1_score(y_test, y_base, zero_division=0):.2f}")

# ---------------- TASK 5: PREPROCESS WITHOUT LEAKAGE ----------------
preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
])
preproc.fit(X_train)
train_freq_mean = preproc.named_transformers_["num"].mean_[num_cols.index("frequency")]
print(f"Frequency Mean -> Train Scaler: {train_freq_mean:.4f} vs Full Cohort: {X['frequency'].mean():.4f}")

# ---------------- TASK 6: TRAIN, TUNE AND EVALUATE ----------------
pipe_lr = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
gs_lr = GridSearchCV(pipe_lr, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)

pipe_rf = Pipeline([("prep", preproc), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
gs_rf = GridSearchCV(pipe_rf, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_train, y_train)

print(f"\nLR Best Params: {gs_lr.best_params_}, Best CV F1: {gs_lr.best_score_:.4f}")
print(f"RF Best Params: {gs_rf.best_params_}, Best CV F1: {gs_rf.best_score_:.4f}")

for name, model in [("Logistic Regression", gs_lr.best_estimator_), ("Random Forest", gs_rf.best_estimator_)]:
    preds = model.predict(X_test)
    print(f"\n--- Test Evaluation: {name} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f} | Recall: {recall_score(y_test, preds, zero_division=0):.4f} | F1: {f1_score(y_test, preds, zero_division=0):.4f}")

# ---------------- TASK 7: CLASS-BALANCE EXPERIMENT ----------------
pos_idx = y_train[y_train == 1].index
neg_count = (y_train == 0).sum()
n_to_add = neg_count - len(pos_idx)
rng_os = np.random.RandomState(42)
dup_idx = rng_os.choice(pos_idx, size=n_to_add, replace=True)

X_train_bal = pd.concat([X_train, X_train.loc[dup_idx]], ignore_index=True)
y_train_bal = pd.concat([y_train, y_train.loc[dup_idx]], ignore_index=True)

lr_bal = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1000, C=gs_lr.best_params_["clf__C"], random_state=42))])
lr_bal.fit(X_train_bal, y_train_bal)
preds_bal = lr_bal.predict(X_test)
print(f"\n--- Balanced LR Test Evaluation (Class counts: {y_train_bal.value_counts().to_dict()}) ---")
print(f"Precision: {precision_score(y_test, preds_bal):.4f} | Recall: {recall_score(y_test, preds_bal):.4f} | F1: {f1_score(y_test, preds_bal):.4f}")

# ---------------- TASK 8: INTERPRET, PERSIST AND USE ----------------
best_rf = gs_rf.best_estimator_
feat_names = num_cols + list(best_rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
rf_imp = pd.Series(best_rf.named_steps["clf"].feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTop 5 RF Features:\n", rf_imp.head(5).round(4))

best_lr = gs_lr.best_estimator_
lr_coefs = pd.Series(best_lr.named_steps["clf"].coef_[0], index=feat_names)
print("\nLR Coefficients (recency, frequency, monetary):\n", lr_coefs[["recency", "frequency", "monetary"]].round(4))

joblib.dump(best_rf, "outputs/lapse_model.joblib")
reloaded_rf = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(best_rf.predict(X_test), reloaded_rf.predict(X_test)), "Reloaded model predictions differ!"

test_df = pd.DataFrame({
    "customer_id": cid_test.values,
    "lapse_probability": reloaded_rf.predict_proba(X_test)[:, 1].round(4),
    "actually_lapsed": y_test.values
}).merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")

ret_list = test_df[test_df["is_premium"] == True].sort_values("lapse_probability", ascending=False).head(10)
ret_list[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
print(f"\nRetention list saved ({len(ret_list)} rows). Truly lapsed in top 10: {ret_list['actually_lapsed'].sum()}")
