# Evaluation protocol

SourceLens separates pipeline smoke tests from claims about real-world quality.

## Dataset contract

The passage benchmark reads JSON Lines. Each record contains paths relative to an
explicit dataset root:

```json
{
  "example_id": "candidate-1:source-8",
  "group_id": "source-8",
  "source_path": "src/source-8.txt",
  "candidate_path": "susp/candidate-1.txt",
  "category": "paraphrase",
  "annotations": [
    {
      "candidate_start": 120,
      "candidate_end": 340,
      "source_start": 44,
      "source_end": 262
    }
  ]
}
```

Manifest paths are resolved beneath `--dataset-root`; traversal outside that
directory is rejected.

## PAN-compatible import

PAN text-alignment corpora provide paired source/suspicious documents and XML
passage offsets. Because the documents have research-use restrictions, download
them through the official PAN/TIRA process and keep them outside this repository.

```bash
python scripts/import_pan.py \
  --pairs /data/pairs \
  --source-dir src \
  --candidate-dir susp \
  --truth-dir /data/truth \
  --output /data/manifest.jsonl

python scripts/evaluate_manifest.py /data/manifest.jsonl \
  --dataset-root /data \
  --output reports/pan_metrics.json
```

## Reported metrics

- Pair-level precision, recall, F1, and confusion counts.
- Candidate passage micro-precision, micro-recall, and micro-F1 over annotated
  character intervals.
- Per-category binary metrics.
- Unique source-group count.
- Exact scoring-method versions observed in results.

The current synthetic CSV report remains a smoke benchmark only. Do not compare
its labels or metrics directly with passage-annotated PAN results.

## Robustness suite

`generate_adversarial_suite` provides deterministic cases for casing,
punctuation, whitespace, sentence order, and filler insertion. It is a regression
suite, not a replacement for human-written or LLM-generated paraphrase data.
