# Interview notes

## What was wrong with version 1?

It was trained on one TF-IDF vector made by concatenating the source and candidate,
then received only the uploaded text during inference. The features therefore had
different meaning at training and serving. A sigmoid over a LinearSVC margin was
also presented as probability without calibration.

## Why not immediately use BERT?

A transparent baseline provides a reproducible floor, creates an evaluation
contract, and surfaces data problems. A transformer adds latency and operational
cost but does not fix leakage, weak labels, or missing evidence. I would ship one
only after it wins on a frozen, source-grouped test set.

## How is leakage prevented?

Evaluation normalizes and hashes each source, then assigns the whole source group
to one partition. The report calculates the intersection and requires it to be
zero. In a larger dataset, I would also cluster near-duplicate sources before
splitting rather than relying on exact normalized hashes.

## Why is the score not probability?

It is a weighted similarity signal. Probability would require a clearly defined
event, representative labels, and held-out calibration with reliability metrics.
Even then, "probability the pair matches a dataset label" is not "probability of
misconduct."

## What would production monitoring include?

- Request latency, error rate, and input-size distribution without raw text.
- Similarity and verdict distributions by approved domain.
- Reviewer agreement, false-positive themes, and subgroup performance.
- Policy/model version on every result.
- Drift alerts and a rollback-ready release artifact.

## What is the next technical milestone?

Create a licensed, representative evaluation set with long documents, citations,
hard negatives, adversarial edits, and paraphrases. Benchmark sentence embeddings
for retrieval and a cross-encoder for reranking, then add citation-aware evidence
suppression. Keep this baseline as a regression oracle.
