# SourceLens

SourceLens is an evidence-first, pairwise text-similarity service for plagiarism
review. It compares a **known source** with a **candidate**, reports transparent
similarity signals, and returns the exact matching passages and character offsets
that produced the result.

It deliberately does **not** claim to determine misconduct. Similarity is not
probability, and plagiarism requires authorship, citation, and policy context.

## Why this rebuild exists

The original research notebook concatenated the source and candidate during
training but supplied only one document during inference. It also applied a
sigmoid to an uncalibrated SVM margin and presented that value as a plagiarism
probability. That was a train/serve mismatch and an invalid interpretation.

Version 2 removes the opaque pickle-based classifier from the production path:

- Pairwise features are computed from both documents at inference.
- Word and character n-gram similarity are symmetric and inspectable.
- Sentence-level matches include source and candidate offsets.
- Verdict thresholds prioritize human review; scores are never called probabilities.
- Evaluation splits by normalized source hash to prevent group leakage.
- A typed API, CLI, UI, tests, CI, and container make the system reproducible.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn plagiarism_detection.api:app --reload
```

Open `http://127.0.0.1:8000`. Interactive API documentation is at `/docs`.

### CLI

```bash
sourcelens path/to/source.txt path/to/candidate.txt
```

### Docker

```bash
docker build -t sourcelens .
docker run --rm -p 8000:8000 sourcelens
```

## API contract

`POST /v1/analyze`

```json
{
  "source": "A production model needs monitoring and evaluation evidence.",
  "candidate": "A production model needs monitoring and evaluation evidence."
}
```

The response includes:

- `similarity_score`: bounded review signal, not probability.
- `verdict`: `low_overlap`, `review_recommended`, or `high_overlap`.
- `lexical_similarity` and `character_similarity`: inspectable components.
- `candidate_coverage`: fraction of meaningful candidate vocabulary in evidence.
- `evidence`: matched text, offsets, local score, and match type.
- `method`: versioned scoring method for traceability.

Inputs are capped at 100,000 characters each. The service processes input in memory
and does not persist document content.

## Evaluate and test

```bash
pytest
ruff check .
python scripts/evaluate.py dataset_new.csv --output reports/baseline_metrics.json
```

The evaluation performs a deterministic source-group split and asserts zero source
group leakage. The bundled 370-row synthetic dataset is retained only as a smoke
benchmark; its short sentences and ambiguous labels are not evidence of
production-level quality. See [MODEL_CARD.md](MODEL_CARD.md).

## Architecture

```text
Browser / API client
        |
        v
FastAPI validation (size limits, strict schema)
        |
        v
PairwiseAnalyzer
  |-- word unigram + bigram cosine
  |-- character 3-5 gram cosine
  |-- sentence/window evidence matching
  `-- versioned review policy
        |
        v
Typed JSON result with evidence offsets
```

The core engine uses the Python standard library and has no model artifact to
deserialize. Delivery dependencies are pinned to compatible version ranges.

## Repository map

```text
src/plagiarism_detection/  production engine, API, and CLI
web/                        accessible review interface
tests/                      unit, API-contract, and leakage tests
scripts/evaluate.py         reproducible grouped evaluation
.github/workflows/ci.yml    lint, test, evaluation, and image build
MODEL_CARD.md               intended use, metrics, and limitations
docs/                       architecture and interview notes
*.ipynb / *.pkl             legacy research artifacts, not production code
```

## Roadmap

- Benchmark sentence-embedding and cross-encoder models on a representative,
  licensed document-pair dataset.
- Add citation-aware exclusions and reviewer annotations.
- Add document ingestion behind malware scanning and isolated text extraction.
- Monitor score distributions and reviewer outcomes after a privacy review.

## Responsible use

SourceLens is a decision-support tool. Do not automatically penalize a writer,
infer intent, or treat a high score as proof. Always show the passages to a
qualified reviewer and consider quotes, citations, templates, and common phrases.

## Author

Karthik Guvvala
