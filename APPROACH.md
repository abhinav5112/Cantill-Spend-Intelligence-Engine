# Approach

## Pipeline

1. **Config-driven ingestion.** Company configs identify source files, taxonomy, optional labels, model artifacts and output names. A/B have dedicated adapters for their supplied schemas; C uses column mappings in its config. Category trees are read at runtime; category names are not embedded in the code.
2. **Normalization.** Numeric strings are parsed defensively. A ERP receipt values are already net; A AP gross amounts are divided by `1 + VAT rate` (15% default). B/C convert to USD by dividing local-currency amounts by the monthly `units_per_usd` rate; USD is unchanged.
3. **Spend decision.** Explicit cancelled / non-approved statuses and non-procurement account/text rules receive the brief's reason codes. B's analyst seed labels are used to learn repeated high-purity non-procurement GL account and account-description rules. Negative values are retained as credits/reversals unless independently excluded.
4. **Cross-system reconciliation.** A tool invoice lines are compared to ERP receipts using digit-normalized PO references. Candidate pairs are accepted when net amount agrees within a small tolerance and supplier or line-text evidence supports the match. Tool invoice exports repeated with the same invoice, normalized line text and net amount are counted once. ERP records are retained as the counted side of a cross-system match.
5. **Entity resolution.** Supplier keys use normalized supplier names (with vendor ID as fallback), with conservative fuzzy merging of near-identical names. Item keys prefer material/item codes; otherwise they use normalized descriptions. A matched AP duplicate inherits the ERP item key.
6. **Classification.** A/B use TF-IDF text representations and class-centroid similarity trained on A's full ERP labels and B's analyst seed labels. C and label-free fallback use taxonomy-only TF-IDF similarity against the tree paths. The predicted path is emitted down to the tree's lowest available level. Confidence is a similarity score, not a calibrated probability.
7. **Output.** Every source record receives exactly one output row, including excluded rows with zero spend and a reason code.

## Label-hidden A test

A was classified using only its raw text/metadata and the taxonomy tree, with `labels_A.csv` withheld from the classifier. Evaluation was then performed against the withheld labels:

- L1 exact accuracy: **45.6%**
- L1+L2 exact path accuracy: **9.9%**
- Full L1–L3 leaf-path exact accuracy: **6.1%**

This exposes the limit of taxonomy-only lexical matching on short mining descriptions and abbreviations. The supervised A/B models improve on this by learning from the supplied labels. C remains a true zero-label case; the classifier uses the unseen tree and the C line descriptions only.

## Runtime and cost

The supplied A/B runs completed locally in a few minutes each on the development environment, well below the 60-minute limit for 25,000 lines. No hosted LLM or paid API is called; inference cost is **$0 per 1,000 lines** beyond local compute.

## Blind-test readiness

The fitted A/B vectorizers and taxonomy centroids are persisted under `models/`, so A2/B2 inference does not depend on labelled examples being present in the fresh batch. B's account exclusion rules are persisted separately. C's config supplies column mappings and a new taxonomy; no C model training or labels are required.
