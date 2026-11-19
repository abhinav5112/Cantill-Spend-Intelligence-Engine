# Procurement Spend Intelligence Engine

A config-driven pipeline for consolidating raw procurement records into a normalized spend cube. It supports the supplied Company A and B layouts and a generic Company C layout with an unseen category tree.

## Capabilities

- Multi-source ingestion for Company A's ERP goods receipts and AP export.
- Cross-system duplicate matching using normalized PO references, net amounts, supplier similarity and line-text similarity.
- Duplicate invoice-export detection.
- Gross-to-net VAT normalization for A; monthly FX conversion to USD for B/C.
- Exclusion reason codes: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`.
- Supplier and item key normalization.
- Company-specific supervised taxonomy classification for A/B, with a tree-only fallback for an unseen company.
- One output row per input record, including excluded rows.
- Persisted compact model artifacts so A2/B2 can be processed without re-training on their fresh rows.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Unzip the private task kit beside this repository so that `Cantill_DataML_Task_Kit/` is present. The data folder is intentionally excluded from Git. Do not commit the source data or the ZIP.

## Build run

From the repository root:

```bash
python run_build.py --root Cantill_DataML_Task_Kit --output-dir outputs
```

The first run trains and saves `models/company_A.joblib` and `models/company_B.joblib` from the supplied labels. It also saves `models/company_B_rules.json` for account-level non-procurement rules. The output files are `outputs/abhinav__A.csv` and `outputs/abhinav__B.csv`.

## Blind-test run

Keep the model artifacts and code frozen after the build deadline. Replace the input files with the fresh A2/B2 files, preserving the paths in the configs. Use the provided templates (or copy them and update only file paths):

```bash
python spend_engine.py --config config_A2_template.json --root blind_test_kit --output-dir submission
python spend_engine.py --config config_B2_template.json --root blind_test_kit --output-dir submission
```

For C, copy `config_C_template.json` to `config_C.json`, update the input file names and column mappings to match C's file, and run:

```bash
python spend_engine.py --config config_C.json --root blind_test_kit --output-dir submission
```

C does not require labelled examples or a saved classifier. Its taxonomy is read at runtime and used as the candidate set. The generic C config supports up to four category levels (`l1`–`l4`).

## Output schema

`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence`

- `include` is `1` or `0`; excluded spend is zero.
- `spend` is net spend in the reporting currency (ZAR for A; USD for B/C).
- Category names are taken from the provided tree; levels below the leaf remain blank.
- `confidence` is a similarity score in `[0,1]`; treat it as a ranking signal rather than a calibrated probability.

## Design notes

See [APPROACH.md](APPROACH.md) for the pipeline and measured label-hidden baseline, and [DECISIONS.md](DECISIONS.md) for data-specific assumptions. No paid API or LLM is used; all processing is local.
