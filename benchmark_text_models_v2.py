"""Compare word, character, and combined TF-IDF classifiers on a leakage-resistant split.

All rows sharing a material_number stay in one partition. Labels are used only
for fitting and offline evaluation; category labels are never included as text.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
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

def make_word():
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=30000, sublinear_tf=True, strip_accents="unicode",
    )

def make_char():
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, sublinear_tf=True, strip_accents="unicode",
    )

def make_combined():
    return FeatureUnion([
        ("word", make_word()),
        ("char", make_char()),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Cantill_DataML_Task_Kit")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    company = Path(args.root) / "company_A"
    erp = pd.read_csv(company / "erp_goods_receipts.csv", dtype=str, low_memory=False).fillna("")
    labels = pd.read_csv(company / "labels_A.csv", dtype=str, low_memory=False).fillna("")
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    data = erp.merge(labels, on="record_id", how="inner", validate="one_to_one")

    required = FIELDS + ["category_l1", "category_l2", "category_l3", "material_number", "line_id"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")

    text = data[FIELDS].apply(lambda row: " ".join(norm(x) for x in row if norm(x)), axis=1)
    y = data[["category_l1", "category_l2", "category_l3"]].astype(str).agg(" > ".join, axis=1).str.strip(" > ")
    keep = y.ne("")
    data, text, y = data.loc[keep].reset_index(drop=True), text.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    groups = data["material_number"].where(data["material_number"].str.strip().ne(""), data["line_id"])
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    ).split(text, y, groups))

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise SystemExit(f"Leakage detected: {len(overlap)} material groups overlap")

    experiments = [
        ("word", make_word()),
        ("char", make_char()),
        ("word_plus_char", make_combined()),
    ]

    print("=== Company A grouped model comparison ===")
    print(f"Rows: {len(data):,} | Train: {len(train_idx):,} | Test: {len(test_idx):,}")
    print(f"Material-number overlap: {len(overlap)}")
    print(f"Random seed: {args.seed} | LinearSVC C: {args.c} | max_iter: 10000")
    results = []

    for name, vectorizer in experiments:
        X_train = vectorizer.fit_transform(text.iloc[train_idx])
        X_test = vectorizer.transform(text.iloc[test_idx])
        model = LinearSVC(C=args.c, class_weight="balanced", max_iter=10000, tol=1e-4)
        model.fit(X_train, y.iloc[train_idx])
        pred = pd.Series(model.predict(X_test), index=test_idx)

        scores = {}
        for level, label in [(1, "L1"), (2, "L1_L2"), (3, "FULL_PATH")]:
            project = lambda value: " > ".join(str(value).split(" > ")[:level])
            scores[label] = accuracy_score(y.iloc[test_idx].map(project), pred.map(project))
        results.append({"model": name, **scores})
        print(
            f"{name:15s} | L1 {scores['L1']:.4%} | "
            f"L1+L2 {scores['L1_L2']:.4%} | Full {scores['FULL_PATH']:.4%}"
        )

    result_df = pd.DataFrame(results).sort_values("FULL_PATH", ascending=False)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    result_df.to_csv(out / "classification_model_comparison.csv", index=False)
    print(f"\nSaved comparison: {out / 'classification_model_comparison.csv'}")
    print("Select a model only after reviewing all three metrics and error patterns.")

if __name__ == "__main__":
    main()
