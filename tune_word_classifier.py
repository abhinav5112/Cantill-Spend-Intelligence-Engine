"""Small controlled hyperparameter search for Company A flat word TF-IDF + LinearSVC.

Uses the same grouped material-number split (seed 42 by default) as the earlier
benchmarks. No material number is used as a model feature.
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

def norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Cantill_DataML_Task_Kit")
    ap.add_argument("--seed", type=int, default=42)
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

    X_text = df[FIELDS].apply(lambda r: " ".join(norm(v) for v in r if norm(v)), axis=1)
    y = df[["category_l1", "category_l2", "category_l3"]].fillna("").astype(str)
    y = y.apply(lambda c: c.str.strip())
    valid = y["category_l1"].ne("")
    df, X_text, y = df.loc[valid].reset_index(drop=True), X_text.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    groups = df["material_number"].where(df["material_number"].str.strip().ne(""), df["line_id"])

    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    ).split(X_text, y["category_l1"], groups))
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise SystemExit(f"Leakage detected: {len(overlap)} material groups overlap")

    train_text, test_text = X_text.iloc[train_idx], X_text.iloc[test_idx]
    train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
    train_path = train_y.apply(lambda r: " > ".join(v for v in r if v), axis=1)
    test_path = test_y.apply(lambda r: " > ".join(v for v in r if v), axis=1)

    configs = []
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        for c_value in [0.25, 1.0, 4.0]:
            for weights in [None, "balanced"]:
                configs.append((ngram, c_value, weights))

    results = []
    print("=== Company A: flat word TF-IDF tuning ===")
    print(f"Rows: {len(df):,} | Train: {len(train_idx):,} | Test: {len(test_idx):,}")
    print(f"Material-number overlap: {len(overlap)}")
    print(f"Configurations: {len(configs)} | max_iter: {args.max_iter}")

    for ngram, c_value, weights in configs:
        vec = TfidfVectorizer(
            analyzer="word", ngram_range=ngram, min_df=2,
            max_features=40000, sublinear_tf=True, strip_accents="unicode"
        )
        Xtr = vec.fit_transform(train_text)
        Xte = vec.transform(test_text)
        clf = LinearSVC(C=c_value, class_weight=weights, max_iter=args.max_iter, tol=1e-4)
        clf.fit(Xtr, train_path)
        pred = clf.predict(Xte)

        pred_parts = [p.split(" > ") for p in pred]
        p1 = [p[0] if len(p) > 0 else "" for p in pred_parts]
        p2 = [p[1] if len(p) > 1 else "" for p in pred_parts]
        p3 = [p[2] if len(p) > 2 else "" for p in pred_parts]

        l1 = accuracy_score(test_y["category_l1"], p1)
        mask2 = test_y["category_l2"].ne("")
        l12 = sum(
            test_y["category_l1"].iloc[i] == p1[i]
            and test_y["category_l2"].iloc[i] == p2[i]
            for i in range(len(p1)) if mask2.iloc[i]
        ) / max(1, int(mask2.sum()))
        mask3 = test_y["category_l3"].ne("")
        full = sum(
            test_y["category_l1"].iloc[i] == p1[i]
            and test_y["category_l2"].iloc[i] == p2[i]
            and test_y["category_l3"].iloc[i] == p3[i]
            for i in range(len(p1)) if mask3.iloc[i]
        ) / max(1, int(mask3.sum()))

        row = {
            "ngram_range": str(ngram), "C": c_value,
            "class_weight": str(weights), "L1": l1, "L1_L2": l12,
            "full_path": full,
        }
        results.append(row)
        print(
            f"ngram={ngram} C={c_value:<4} weight={str(weights):<8} | "
            f"L1={l1:.4%} L1+L2={l12:.4%} Full={full:.4%}"
        )

    result = pd.DataFrame(results).sort_values(
        ["full_path", "L1_L2", "L1"], ascending=False
    )
    Path("outputs").mkdir(exist_ok=True)
    result.to_csv("outputs/word_tuning_results.csv", index=False)
    print("\nTop 5 configurations by full-path accuracy:")
    print(result.head(5).to_string(index=False))
    print("\nSaved: outputs/word_tuning_results.csv")

if __name__ == "__main__":
    main()
