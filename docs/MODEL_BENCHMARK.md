# Model comparison benchmark

SourceLens compares pair scorers under one leakage-aware protocol instead of
reporting unrelated metrics from separate notebooks.

## Models

The benchmark supports:

- `baseline`: the transparent pairwise analyzer used by `/v1/analyze`.
- `hashing`: deterministic character n-gram embeddings with cosine similarity.
- `semantic`: a sentence-transformer bi-encoder.
- `cross-encoder`: a pairwise transformer reranker.

The default run uses only `baseline,hashing` and never downloads a model.
Transformer loading requires both the `semantic` package extra and the explicit
`--enable-transformers` flag.

## Smoke comparison

```bash
python scripts/benchmark_models.py dataset_new.csv \
  --models baseline,hashing \
  --output reports/model_comparison_smoke.json
```

The bundled CSV is suitable only for pipeline regression. Its short synthetic
sentences and ambiguous labels cannot support a production-quality claim.

## Representative evaluation

Convert a licensed target-domain corpus to the same CSV columns:

```text
source_text,plagiarized_text,label
```

Keep every source document under one normalized source group. The runner splits
by group, calibrates a threshold only on the calibration partition, and reports
held-out test performance. Duplicate sources therefore cannot leak into both
partitions.

Run transformer candidates explicitly:

```bash
pip install -e ".[semantic]"

python scripts/benchmark_models.py /data/licensed_pairs.csv \
  --models baseline,hashing,semantic,cross-encoder \
  --enable-transformers \
  --sentence-model sentence-transformers/all-MiniLM-L6-v2 \
  --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output reports/licensed_model_comparison.json
```

Model downloads should happen during a controlled benchmark or image build,
not during an API request.

## Report contract

Every JSON report contains:

- a content-derived dataset fingerprint;
- pair, label, and unique-source-group counts;
- calibration and held-out partition sizes;
- an explicit source-group leakage count;
- the calibrated threshold for each model;
- precision, recall, F1, accuracy, and confusion counts;
- held-out average precision;
- per-category classification metrics;
- mean, p50, and p95 pair-scoring latency;
- observed score ranges; and
- declared dataset limitations.

The dataset fingerprint makes results auditable without publishing restricted
documents. Model identity is recorded in each report; do not rename a local
checkpoint without updating its benchmark identifier.

## Promotion gate

A transformer should replace the lightweight default only when it improves the
frozen held-out set, stays within the deployment latency budget, and is reviewed
for licensing, privacy, multilingual behavior, citation handling, and subgroup
failure modes.
