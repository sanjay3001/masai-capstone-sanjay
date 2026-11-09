import os
import sqlite3
import numpy as np
import pandas as pd

os.makedirs(os.path.join("data", "clean"), exist_ok=True)
os.makedirs("sql", exist_ok=True)

# ---------------- TASK 2: AUDIT RAW DATA ----------------
cust_raw = pd.read_csv("data/raw/customers.csv", keep_default_na=False)
prod_raw = pd.read_csv("data/raw/products.csv", keep_default_na=False)
ord_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False)

def audit_table(df, name):
    print(f"=== Audit: {name} | Shape: {df.shape} ===")
    for col in df.columns:
        missing = (df[col] == "").sum() + df[col].isna().sum()
        pct = (missing / len(df)) * 100
        print(f"  {col} ({df[col].dtype}): missing={missing} ({pct:.2f}%)")

audit_table(cust_raw, "customers.csv")
audit_table(prod_raw, "products.csv")
audit_table(ord_raw, "orders.csv")

exact_dups = ord_raw.duplicated().sum()
pending_rows = (ord_raw["status"] == "pending").sum()
distinct_oids = ord_raw["order_id"].nunique()
non_num_amt = pd.to_numeric(ord_raw["order_amount"], errors="coerce").isna().sum()
deliv_num = pd.to_numeric(ord_raw["delivery_minutes"], errors="coerce")
neg_deliv = (deliv_num < 0).sum()
blank_deliv = (ord_raw["delivery_minutes"] == "").sum()

age_num = pd.to_numeric(cust_raw["age"], errors="coerce")
blank_age = (cust_raw["age"] == "").sum()
impossible_age = ((age_num < 10) | (age_num > 100)).sum()
raw_cities = cust_raw["city"].nunique()
raw_genders = cust_raw.loc[cust_raw["gender"] != "", "gender"].nunique()
blank_genders = (cust_raw["gender"] == "").sum()

print("\n--- Specific Audit Figures ---")
print(f"Orders rows: {len(ord_raw)} | Exact dups: {exact_dups} | Pending: {pending_rows} | Distinct order_id: {distinct_oids}")
print(f"Non-numeric amounts: {non_num_amt} | Negative delivery: {neg_deliv} | Blank delivery: {blank_deliv}")
print(f"Blank age: {blank_age} | Age outside 10-100: {impossible_age} | Raw cities: {raw_cities} | Raw genders: {raw_genders} | Blank genders: {blank_genders}")

# ---------------- TASK 3: CLEAN CUSTOMERS ----------------
cust = cust_raw.copy()
cust["city"] = cust["city"].str.strip().str.title()

gender_map = {
    "male": "male", "m": "male",
    "female": "female", "f": "female",
    "": "unknown"
}
cust["gender"] = cust["gender"].str.strip().str.lower().map(lambda x: gender_map.get(x, "unknown"))

cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

# ---------------- TASK 4: CLEAN ORDERS ----------------
orders = ord_raw.copy()
# 1. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
# 2. Remove exact duplicate rows
orders = orders.drop_duplicates().copy()
# 3. Keep latest order_ts per order_id
orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
# 4. Convert order_amount to numeric & fill from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders["quantity"] = pd.to_numeric(orders["quantity"])
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])

prod = prod_raw.copy()
prod["price_inr"] = pd.to_numeric(prod["price_inr"])
orders = orders.merge(prod[["product_id", "price_inr", "category"]], on="product_id", how="left")

calc_amt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100.0)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc_amt)

# 5. Delivery minutes cleaning
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

comp_mask = orders["status"] == "completed"
orders.loc[comp_mask, "delivery_minutes"] = orders.loc[comp_mask, "delivery_minutes"].fillna(
    orders.loc[comp_mask].groupby("category")["delivery_minutes"].transform("median")
)
orders.loc[orders["status"] == "cancelled", "delivery_minutes"] = np.nan
orders = orders.drop(columns=["price_inr", "category"])

# ---------------- TASK 5: STORE AND VALIDATE ----------------
cust_out = cust.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

cust_out.to_csv("data/clean/customers.csv", index=False)
prod.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
cust_out.to_sql("customers", conn, if_exists="replace", index=False)
prod.to_sql("products", conn, if_exists="replace", index=False)
orders_out.to_sql("orders", conn, if_exists="replace", index=False)

# Validation Checkpoints
assert cust["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert cust["age"].isna().sum() == 0 and cust["age"].between(18, 60).all()
assert len(orders) == 12309 and orders["order_id"].nunique() == 12309
assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[orders["status"] == "completed", "delivery_minutes"].isna().sum() == 0
assert orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199
assert round(orders.loc[orders["status"] == "completed", "order_amount"].sum(), 2) == 2998668.80
print("ALL CHECKPOINTS PASSED")

# ---------------- TASK 6: SQL ANALYSIS ----------------
with open("sql/queries.sql", "r", encoding="utf-8") as f:
    sql_queries = [q.strip() for q in f.read().split(";") if q.strip()]

for idx, q in enumerate(sql_queries, 1):
    res = pd.read_sql_query(q, conn)
    print(f"\n--- SQL Q{idx} (rows={len(res)}) ---")
    if idx == 6:
        top_per_city = res[res["rnk"] == 1]
        diff_rnk = (res["rnk"] != res["row_num"]).sum()
        print(top_per_city[["city", "customer_id", "total_spend"]])
        print(f"Rows where RANK() != ROW_NUMBER(): {diff_rnk}")
    else:
        print(res.head(3))

conn.close()

# ---------------- TASK 7: PANDAS CROSS-CHECK & EDA ----------------
m_df = orders[orders["status"] == "completed"].merge(cust, on="customer_id").merge(prod, on="product_id")
piv = pd.pivot_table(m_df, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True).round(2)
print("\n--- Pivot Table Grand Total ---", piv.loc["All", "All"])

city_agg = m_df.groupby("city").agg(
    sum_amt=("order_amount", "sum"),
    mean_amt=("order_amount", "mean"),
    count_amt=("order_amount", "count")
).round(2)
print("\n--- City Named Aggregation ---\n", city_agg)

comp_amt = orders.loc[orders["status"] == "completed", "order_amount"]
q1, q2, q3 = comp_amt.quantile([0.25, 0.50, 0.75])
iqr = q3 - q1
low_fence, up_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nStats: Mean={comp_amt.mean():.2f}, Median={q2:.2f}, Skew={comp_amt.skew():.2f}")
print(f"Q1={q1:.2f}, Q3={q3:.2f}, Lower Fence={low_fence:.2f}, Upper Fence={up_fence:.2f}")
print(f"Outliers above={ (comp_amt > up_fence).sum() }, below={ (comp_amt < low_fence).sum() }")

corr = orders.loc[orders["status"] == "completed", ["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\n--- Correlation Matrix ---\n", corr.round(4))
