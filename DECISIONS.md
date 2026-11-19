# Decisions and data observations

## Company A — mining / chemicals group

- `erp_goods_receipts.csv` contains 37,029 records and `invoice_tool_export.csv` contains 28,520. The ERP value is treated as net ZAR. AP tool amounts are gross; VAT values such as `15`, `0.15` and `15%` are normalized, with 15% used when blank.
- A tool-to-ERP reconciliation uses digit-only PO references (removing prefixes, separators and item suffixes), then checks net amount plus supplier-name and line-text similarity. A pair is marked `DUPLICATE` on the AP tool side; the ERP receipt remains the counted record. This found 12,758 cross-system matches. Repeated AP exports are additionally deduplicated by invoice number, normalized line text and net amount; total tool rows excluded as duplicates: 13,382.
- The AP export has explicit statuses. `Paid` and `Approved` count; `Cancelled` receives `CANCELLED`; `On Hold` and `Rejected` receive `NOT_APPROVED`.
- ERP receipts do not have an approval-status field. They are assumed to be valid procurement records unless a non-procurement signal is present. Credits/negative amounts are retained as spend adjustments.
- Supplier names are more complete than vendor IDs (many ERP vendor IDs are blank), so normalized supplier name is the primary key basis; vendor ID is a fallback. Conservative fuzzy matching is used for AP supplier names.
- Material number is the preferred item identity. When missing, normalized material/service description is used. A tool record matched to an ERP receipt inherits its ERP item key.

## Company B — metal packaging manufacturer

- 41,028 invoice lines span 8 ERP instances and 25 currencies. FX rates are monthly and expressed as local-currency units per USD; therefore local amounts are divided by the rate. USD amounts are unchanged.
- Posting date determines the FX month. The supplied file's dates and monthly FX table cover all currencies in the build data. For a future record with an impossible date and a non-USD currency, no reliable month-specific FX rate can be selected; such a record currently receives zero converted spend and should be reviewed.
- Document type `CA` is treated as cancelled. The data has no general approval-status column. Negative lines are retained as credit/reversal values unless excluded for another reason.
- Analyst seed labels are imperfect but useful. Repeated GL accounts / GL descriptions with at least two labelled examples and at least 90% excluded examples are treated as non-procurement rules. Explicit intercompany, royalty, tax-payable, financing and interest patterns are also excluded. Ambiguous account groups are not automatically excluded.
- Vendor IDs are used as supplier keys because supplier names are not supplied. Material number is the preferred item key; line description is the fallback.

## General / unseen Company C

- The C adapter is driven by config column mappings. It assumes the reporting currency is USD, and uses a configured monthly FX file or a per-line rate for non-USD amounts. If C has no FX file, configure `currency_rate_column` when rates are supplied per line.
- The generic exclusion rules are deliberately conservative. Company-specific account or document semantics should be added as explicit config-driven rules when C's schema makes them observable; category names are never hardcoded.
- Taxonomy-only classification uses lexical TF-IDF similarity. It is fully label-free but will be less reliable for cryptic abbreviations or category names that do not share vocabulary with invoice descriptions. Low-confidence lines should be reviewed first.
