"""Leakage-resistant Company A classification benchmark.

Keeps all rows sharing a material_number in one split. Labels are used only
for supervised training and offline scoring; they are never included in text.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

FIELDS = [
    "material_short_text", "service_short_text", "material_number",
    "material_type", "spend_type", "uom", "po_uom", "opex_capex",
    "cost_centre",
]

def norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Cantill_DataML_Task_Kit", help="Path to extracted task-kit folder")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    company = root / "company_A"
    erp = pd.read_csv(company / "erp_goods_receipts.csv", dtype=str, low_memory=False).fillna("")
    labels = pd.read_csv(company / "labels_A.csv", dtype=str, low_memory=False).fillna("")
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    data = erp.merge(labels, on="record_id", how="inner", validate="one_to_one")
    if data.empty:
        raise SystemExit("No ERP/label rows matched. Check the extracted kit path and record IDs.")
    missing = [c for c in FIELDS + ["category_l1", "category_l2", "category_l3", "material_number", "line_id"] if c not in data.columns]
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")

    text = data[FIELDS].apply(lambda row: " ".join(norm(x) for x in row if norm(x)), axis=1)
    y = data[["category_l1", "category_l2", "category_l3"]].fillna("").astype(str).agg(" > ".join, axis=1).str.strip(" > ")
    keep = y.ne("")
    data, text, y = data.loc[keep].reset_index(drop=True), text.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    groups = data["material_number"].where(data["material_number"].str.strip().ne(""), data["line_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(text, y, groups))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30000,
        sublinear_tf=True, strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(text.iloc[train_idx])
    X_test = vectorizer.transform(text.iloc[test_idx])
    model = LinearSVC(C=1.0, class_weight="balanced")
    model.fit(X_train, y.iloc[train_idx])
    predicted = pd.Series(model.predict(X_test), index=test_idx)

    print("\n=== Company A grouped validation ===")
    print(f"Matched labelled rows: {len(data):,}")
    print(f"Train rows: {len(train_idx):,} | Test rows: {len(test_idx):,}")
    print(f"Unique material groups: {groups.nunique():,}")
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    print(f"Material-number overlap across split: {len(overlap)} (must be 0)")
    print(f"Train classes: {y.iloc[train_idx].nunique()} | Test classes: {y.iloc[test_idx].nunique()}")
    for level, name in [(1, "L1"), (2, "L1 + L2"), (3, "Full path")]:
        project = lambda value: " > ".join(str(value).split(" > ")[:level])
        score = accuracy_score(y.iloc[test_idx].map(project), predicted.map(project))
        print(f"{name} accuracy: {score:.4%}")

    errors = pd.DataFrame({
        "description": text.iloc[test_idx].values,
        "actual": y.iloc[test_idx].values,
        "predicted": predicted.values,
    })
    errors = errors[errors.actual != errors.predicted].head(20)
    if not errors.empty:
        print("\n=== First 20 full-path errors (for diagnosis) ===")
        print(errors.to_string(index=False, max_colwidth=100))

if __name__ == "__main__":
    main()
