
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

FIELDS = [
    "material_short_text",
    "service_short_text",
    "material_type",
    "spend_type",
    "uom",
    "po_uom",
    "opex_capex",
    "cost_centre",
]


def norm(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize(
        "NFKD", str(value)
    ).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def main():
    folder = Path("Cantill_DataML_Task_Kit/company_A")

    erp = pd.read_csv(
        folder / "erp_goods_receipts.csv",
        dtype=str,
        low_memory=False
    ).fillna("")

    labels = pd.read_csv(
        folder / "labels_A.csv",
        dtype=str,
        low_memory=False
    ).fillna("")

    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)

    df = erp.merge(
        labels,
        on="record_id",
        how="inner",
        validate="one_to_one"
    )

    text = df[FIELDS].apply(
        lambda r: " ".join(
            norm(v) for v in r if norm(v)
        ),
        axis=1
    )

    cats = df[
        ["category_l1", "category_l2", "category_l3"]
    ].fillna("").astype(str)

    cats = cats.apply(lambda c: c.str.strip())

    keep = cats["category_l1"].ne("")

    df = df.loc[keep].reset_index(drop=True)
    text = text.loc[keep].reset_index(drop=True)
    cats = cats.loc[keep].reset_index(drop=True)

    groups = df["material_number"].where(
        df["material_number"].str.strip().ne(""),
        df["line_id"]
    )

    split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=123
    )

    train_idx, test_idx = next(
        split.split(text, cats["category_l1"], groups)
    )

    train_text = text.iloc[train_idx]
    test_text = text.iloc[test_idx]

    y_train = cats.iloc[train_idx].apply(
        lambda r: " > ".join(x for x in r if x),
        axis=1
    )

    y_test = cats.iloc[test_idx].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=20000,
        tol=1e-4
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    parts = [p.split(" > ") for p in predictions]

    predicted = pd.DataFrame({
        "category_l1": [p[0] for p in parts],
        "category_l2": [p[1] if len(p) > 1 else "" for p in parts],
        "category_l3": [p[2] if len(p) > 2 else "" for p in parts],
    })

    actual = y_test.reset_index(drop=True)

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    # Category-wise performance
    report = classification_report(
        actual["category_l1"],
        predicted["category_l1"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        output / "category_performance_A.csv"
    )

    # Confusion matrix
    labels_sorted = sorted(
        set(actual["category_l1"]) |
        set(predicted["category_l1"])
    )

    cm = confusion_matrix(
        actual["category_l1"],
        predicted["category_l1"],
        labels=labels_sorted
    )

    pd.DataFrame(
        cm,
        index=labels_sorted,
        columns=labels_sorted
    ).to_csv(output / "confusion_matrix_A.csv")

    # Misclassified records
    errors = pd.DataFrame({
        "record_id": df.iloc[test_idx]["record_id"].values,
        "actual_l1": actual["category_l1"],
        "predicted_l1": predicted["category_l1"],
        "actual_l2": actual["category_l2"],
        "predicted_l2": predicted["category_l2"],
        "actual_l3": actual["category_l3"],
        "predicted_l3": predicted["category_l3"],
    })

    errors = errors[
        (errors.actual_l1 != errors.predicted_l1) |
        (errors.actual_l2 != errors.predicted_l2) |
        (errors.actual_l3 != errors.predicted_l3)
    ]

    errors.to_csv(
        output / "misclassified_records_A.csv",
        index=False
    )

    # Summary
    print("\n=== BASELINE ERROR ANALYSIS ===")
    print("Test records:", len(test_idx))
    print(
        "L1 Accuracy:",
        round(accuracy_score(
            actual["category_l1"],
            predicted["category_l1"]
        ), 4)
    )
    print("Misclassified records:", len(errors))

    print("\n=== CATEGORY PERFORMANCE ===")
    print(
        pd.DataFrame(report).transpose()
        .sort_values("f1-score")
        .head(15).to_string()
    )

    print("\n=== TOP 20 CONFUSION PAIRS ===")

    pairs = pd.DataFrame({
        "actual": actual["category_l1"],
        "predicted": predicted["category_l1"]
    })

    pairs = pairs[pairs.actual != pairs.predicted]

    print(
        pairs.groupby(["actual", "predicted"])
        .size()
        .sort_values(ascending=False)
        .head(20).to_string()
    )

    print("\nSaved analysis files:")
    print("outputs/category_performance_A.csv")
    print("outputs/confusion_matrix_A.csv")
    print("outputs/misclassified_records_A.csv")


if __name__ == "__main__":
    main()
