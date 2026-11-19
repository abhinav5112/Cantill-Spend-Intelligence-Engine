"""Independent grouped holdout check for two Company A classifier candidates.

This uses a new group split seed (123), distinct from the tuning split (42).
Material number is used only for grouping, never as a text feature.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

FIELDS = [
    "material_short_text", "service_short_text",
    "material_type", "spend_type", "uom", "po_uom", "opex_capex",
    "cost_centre",
]

def norm(v):
    if pd.isna(v):
        return ""
    v = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", v).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Cantill_DataML_Task_Kit")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--max-iter", type=int, default=20000)
    args = ap.parse_args()

    folder = Path(args.root) / "company_A"
    erp = pd.read_csv(folder / "erp_goods_receipts.csv", dtype=str, low_memory=False).fillna("")
    labels = pd.read_csv(folder / "labels_A.csv", dtype=str, low_memory=False).fillna("")
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    df = erp.merge(labels, on="record_id", how="inner", validate="one_to_one")

    required = FIELDS + ["category_l1", "category_l2", "category_l3", "material_number", "line_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    text = df[FIELDS].apply(lambda r: " ".join(norm(v) for v in r if norm(v)), axis=1)
    cats = df[["category_l1", "category_l2", "category_l3"]].fillna("").astype(str)
    cats = cats.apply(lambda c: c.str.strip())
    keep = cats["category_l1"].ne("")
    df, text, cats = df.loc[keep].reset_index(drop=True), text.loc[keep].reset_index(drop=True), cats.loc[keep].reset_index(drop=True)
    groups = df["material_number"].where(df["material_number"].str.strip().ne(""), df["line_id"])

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed).split(text, cats["category_l1"], groups))
    overlap = set(groups.iloc[tr]) & set(groups.iloc[te])
    if overlap:
        raise SystemExit(f"Material group leakage detected: {len(overlap)}")

    y_train = cats.iloc[tr].apply(lambda r: " > ".join(x for x in r if x), axis=1)
    y_test = cats.iloc[te].reset_index(drop=True)
    train_text, test_text = text.iloc[tr], text.iloc[te]

    configs = [
        ("previous_baseline", (1, 2), 1.0, "balanced"),
        ("tuned_candidate", (1, 3), 0.25, "balanced"),
    ]
    rows = []
    print("=== Independent Company A grouped holdout ===")
    print(f"Seed: {args.seed} | Rows: {len(df):,} | Train: {len(tr):,} | Test: {len(te):,}")
    print(f"Material-number overlap: {len(overlap)}")

    for name, ngram, c_value, weights in configs:
        vec = TfidfVectorizer(
            analyzer="word", ngram_range=ngram, min_df=2,
            max_features=40000, sublinear_tf=True, strip_accents="unicode"
        )
        Xtr = vec.fit_transform(train_text)
        Xte = vec.transform(test_text)
        model = LinearSVC(C=c_value, class_weight=weights, max_iter=args.max_iter, tol=1e-4)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        parts = [p.split(" > ") for p in pred]
        p1 = [p[0] if len(p) > 0 else "" for p in parts]
        p2 = [p[1] if len(p) > 1 else "" for p in parts]
        p3 = [p[2] if len(p) > 2 else "" for p in parts]

        l1 = accuracy_score(y_test["category_l1"], p1)
        m2 = y_test["category_l2"].ne("")
        l12 = sum(
            y_test["category_l1"].iloc[i] == p1[i] and y_test["category_l2"].iloc[i] == p2[i]
            for i in range(len(p1)) if m2.iloc[i]
        ) / max(1, int(m2.sum()))
        m3 = y_test["category_l3"].ne("")
        full = sum(
            y_test["category_l1"].iloc[i] == p1[i]
            and y_test["category_l2"].iloc[i] == p2[i]
            and y_test["category_l3"].iloc[i] == p3[i]
            for i in range(len(p1)) if m3.iloc[i]
        ) / max(1, int(m3.sum()))

        rows.append({"model": name, "ngram": str(ngram), "C": c_value,
                     "L1": l1, "L1_L2": l12, "full_path": full})
        print(f"{name:18s} | L1 {l1:.4%} | L1+L2 {l12:.4%} | Full {full:.4%}")

    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/independent_holdout_comparison.csv", index=False)
    print("Saved: outputs/independent_holdout_comparison.csv")

if __name__ == "__main__":
    main()
