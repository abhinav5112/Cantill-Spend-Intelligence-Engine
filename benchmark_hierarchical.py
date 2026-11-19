"""Compare flat vs cascaded hierarchical classification on a grouped holdout.

Rows sharing material_number remain in one partition. The hierarchy is trained
only on the training partition. At inference, predicted parents (not true
parents) select the next-level classifier, preventing oracle-parent leakage.
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

def fit_svc(X, y, C, max_iter):
    labels = pd.Series(y).astype(str)
    if labels.nunique() == 1:
        return ("constant", labels.iloc[0])
    model = LinearSVC(C=C, class_weight="balanced", max_iter=max_iter, tol=1e-4)
    model.fit(X, labels)
    return ("model", model)

def predict_svc(fitted, X):
    kind, obj = fitted
    if kind == "constant":
        return [obj] * X.shape[0]
    return obj.predict(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Cantill_DataML_Task_Kit")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=20000)
    args = ap.parse_args()

    company = Path(args.root) / "company_A"
    erp = pd.read_csv(company / "erp_goods_receipts.csv", dtype=str, low_memory=False).fillna("")
    labels = pd.read_csv(company / "labels_A.csv", dtype=str, low_memory=False).fillna("")
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    data = erp.merge(labels, on="record_id", how="inner", validate="one_to_one")

    needed = FIELDS + ["category_l1", "category_l2", "category_l3", "material_number", "line_id"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    text = data[FIELDS].apply(lambda r: " ".join(norm(v) for v in r if norm(v)), axis=1)
    cats = data[["category_l1", "category_l2", "category_l3"]].fillna("").astype(str)
    cats = cats.apply(lambda col: col.str.strip())
    keep = cats["category_l1"].ne("")
    data, text, cats = data.loc[keep].reset_index(drop=True), text.loc[keep].reset_index(drop=True), cats.loc[keep].reset_index(drop=True)

    groups = data["material_number"].where(data["material_number"].str.strip().ne(""), data["line_id"])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed).split(text, cats["category_l1"], groups))
    overlap = set(groups.iloc[tr]) & set(groups.iloc[te])
    if overlap:
        raise SystemExit(f"Material group leakage: {len(overlap)}")

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=30000, sublinear_tf=True, strip_accents="unicode"
    )
    Xtr = vectorizer.fit_transform(text.iloc[tr])
    Xte = vectorizer.transform(text.iloc[te])
    train = cats.iloc[tr].reset_index(drop=True)
    test = cats.iloc[te].reset_index(drop=True)

    # Flat baseline: one model per full path.
    train_paths = train.apply(lambda r: " > ".join(x for x in r if x), axis=1)
    flat = fit_svc(Xtr, train_paths, args.c, args.max_iter)
    flat_pred = predict_svc(flat, Xte)
    flat_parts = [p.split(" > ") for p in flat_pred]
    flat_l1 = [p[0] if len(p) > 0 else "" for p in flat_parts]
    flat_l2 = [p[1] if len(p) > 1 else "" for p in flat_parts]
    flat_l3 = [p[2] if len(p) > 2 else "" for p in flat_parts]

    # Cascaded hierarchy: L1 -> L2 conditioned on predicted L1 -> L3
    l1_model = fit_svc(Xtr, train["category_l1"], args.c, args.max_iter)
    pred_l1 = predict_svc(l1_model, Xte)
    pred_l2, pred_l3 = [], []

    l2_models = {}
    for parent in train["category_l1"].unique():
        mask = train["category_l1"].eq(parent).to_numpy()
        child = train.loc[mask, "category_l2"]
        valid = child.ne("").to_numpy()
        if valid.any():
            l2_models[parent] = fit_svc(Xtr[mask][valid], child[valid], args.c, args.max_iter)

    l3_models = {}
    for (parent, child), group in train.groupby(["category_l1", "category_l2"], dropna=False):
        if not child:
            continue
        valid = group["category_l3"].ne("")
        if valid.any():
            positions = group.index[valid].to_numpy()
            l3_models[(parent, child)] = fit_svc(Xtr[positions], group.loc[valid, "category_l3"], args.c, args.max_iter)

    for i, parent in enumerate(pred_l1):
        l2_fit = l2_models.get(parent)
        child = predict_svc(l2_fit, Xte[i]) [0] if l2_fit else ""
        pred_l2.append(child)
        l3_fit = l3_models.get((parent, child)) if child else None
        grandchild = predict_svc(l3_fit, Xte[i])[0] if l3_fit else ""
        pred_l3.append(grandchild)

    def score(pred1, pred2, pred3):
        l1 = accuracy_score(test["category_l1"], pred1)
        mask2 = test["category_l2"].ne("")
        exact2 = [a == b and c == d for a, b, c, d in zip(test["category_l1"], pred1, test["category_l2"], pred2)]
        l1l2 = sum(exact2[i] for i in range(len(exact2)) if mask2.iloc[i]) / max(1, int(mask2.sum()))
        mask3 = test["category_l3"].ne("")
        exact3 = [
            a == b and c == d and e == f
            for a, b, c, d, e, f in zip(test["category_l1"], pred1, test["category_l2"], pred2, test["category_l3"], pred3)
        ]
        full = sum(exact3[i] for i in range(len(exact3)) if mask3.iloc[i]) / max(1, int(mask3.sum()))
        return l1, l1l2, full

    flat_scores = score(flat_l1, flat_l2, flat_l3)
    hier_scores = score(pred_l1, pred_l2, pred_l3)

    print("=== Company A: flat vs cascaded hierarchical ===")
    print(f"Rows: {len(data):,} | Train: {len(tr):,} | Test: {len(te):,}")
    print(f"Material-number overlap: {len(overlap)}")
    print("Metrics are exact prefix-path accuracy on rows with that level present.")
    print(f"Flat word TF-IDF       | L1 {flat_scores[0]:.4%} | L1+L2 {flat_scores[1]:.4%} | Full {flat_scores[2]:.4%}")
    print(f"Hierarchical cascade   | L1 {hier_scores[0]:.4%} | L1+L2 {hier_scores[1]:.4%} | Full {hier_scores[2]:.4%}")

    output = pd.DataFrame({
        "description": text.iloc[te].values,
        "actual_l1": test["category_l1"].values,
        "predicted_l1": pred_l1,
        "actual_l2": test["category_l2"].values,
        "predicted_l2": pred_l2,
        "actual_l3": test["category_l3"].values,
        "predicted_l3": pred_l3,
    })
    output["full_path_correct"] = (
        output.actual_l1.eq(output.predicted_l1)
        & output.actual_l2.eq(output.predicted_l2)
        & output.actual_l3.eq(output.predicted_l3)
    )
    Path("outputs").mkdir(exist_ok=True)
    output.to_csv("outputs/hierarchical_validation_predictions.csv", index=False)
    pd.DataFrame([
        {"model": "flat_word", "L1": flat_scores[0], "L1_L2": flat_scores[1], "full_path": flat_scores[2]},
        {"model": "hierarchical_cascade", "L1": hier_scores[0], "L1_L2": hier_scores[1], "full_path": hier_scores[2]},
    ]).to_csv("outputs/hierarchical_model_comparison.csv", index=False)
    print("Saved outputs/hierarchical_model_comparison.csv")
    print("Saved outputs/hierarchical_validation_predictions.csv")

if __name__ == "__main__":
    main()
