# Model card: SourceLens pairwise baseline v2

## Summary

SourceLens v2 is a deterministic retrieval-style baseline that measures lexical
overlap between a known source and a candidate. It combines word unigram/bigram
cosine similarity, character 3-5 gram cosine similarity, containment, and local
passage evidence. It does not use the legacy serialized SVM.

## Intended use

- Prioritize document pairs for human review.
- Locate exact, near-verbatim, and lexical overlap.
- Provide an auditable baseline for comparison with semantic models.

## Out of scope

- Automatic academic, hiring, legal, or disciplinary decisions.
- Authorship attribution or intent detection.
- Internet-wide source discovery.
- Reliable semantic paraphrase detection across languages.
- Code plagiarism. The earlier notebook mixed Python AST processing with raw
  Java/C++ and is intentionally not exposed as a supported capability.

## Outputs

`similarity_score` is a bounded similarity signal. It is **not** a calibrated
probability. `verdict` is a review-priority band:

| Band | Default threshold | Meaning |
| --- | ---: | --- |
| `low_overlap` | below 0.36 | No strong lexical signal from this source |
| `review_recommended` | 0.36-0.6399 | Inspect passages and context |
| `high_overlap` | 0.64 or above | Strong overlap; human decision required |

Every evidence match includes both texts, character offsets, a local similarity,
and an interpretable match type.

## Evaluation protocol

`scripts/evaluate.py`:

1. Normalizes and hashes each source document.
2. Assigns entire source groups to calibration or held-out test.
3. Tunes a binary-review threshold on calibration F1.
4. Reports precision, recall, F1, confusion counts, and group leakage.
5. Keeps the production policy visible next to benchmark results.

The supplied dataset contains only 370 rows, mostly short synthetic sentences.
Some negative labels are lexically or semantically close. Its metrics should be
treated as a pipeline smoke test, not as evidence of generalization.

## Known limitations

- Lexical features under-detect meaning-preserving paraphrases with different words.
- Boilerplate, definitions, and common phrases can create false positives.
- Passage matching is sentence/window based and can miss cross-boundary copies.
- English-style tokenization has not been validated for every script or language.
- Thresholds need recalibration on a representative, licensed target-domain dataset.
- No score can establish whether attribution was present or reuse was permitted.

## Risk controls

- The UI and API say "similarity," never "probability."
- Result language expresses review priority, not guilt.
- Strict input schema and 100,000-character limits bound API work.
- Inputs are processed in memory and are not logged or stored by the application.
- CI checks unit behavior, API contracts, source-group isolation, lint, and container build.

## Future validation gate

A semantic model should ship only after it beats this baseline on a frozen
source-grouped test set covering exact copy, light edits, paraphrases, cited
quotations, boilerplate, unrelated hard negatives, adversarial edits, and long
documents. Report subgroup metrics and calibrate any probability with a separate
held-out calibration split.
