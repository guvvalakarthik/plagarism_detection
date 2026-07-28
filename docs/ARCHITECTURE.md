# Architecture decisions

## ADR-001: Pairwise inference

**Decision:** every request contains a known source and candidate. All features are
computed on the pair at inference.

**Why:** the legacy classifier trained on concatenated pairs but inferred on one
text, so production input did not match training input.

## ADR-002: Transparent baseline before embeddings

**Decision:** ship a deterministic word/character n-gram baseline with passage
evidence before introducing a transformer.

**Why:** it is reproducible, inexpensive, debuggable, and establishes the metric a
more complex model must beat. It also avoids downloading a large model at startup.

## ADR-003: Similarity is not probability

**Decision:** expose a score and review band. Never label it probability.

**Why:** cosine-like similarity is not calibrated, and even calibrated pair
classification cannot determine plagiarism without contextual evidence.

## ADR-004: Human-reviewable evidence

**Decision:** return matched passages and character offsets in both documents.

**Why:** a reviewer can verify a match, find it in the original input, and account
for citations or boilerplate. A bare label is not actionable.

## ADR-005: Source-grouped evaluation

**Decision:** normalized source hashes define split groups.

**Why:** duplicate source content crossing a row-level random split inflates
metrics. Grouping makes that leakage measurable and preventable.

## Request flow

1. Pydantic rejects missing, extra, too-short, or oversized input.
2. The analyzer normalizes Unicode and extracts word tokens.
3. Global word and character similarities are calculated symmetrically.
4. Candidate segments are matched to their best source segments.
5. Non-overlapping evidence above the policy threshold is retained.
6. A versioned decision policy maps the aggregate signal to a review band.
7. The API serializes the typed result; document text is not persisted.

## Scaling direction

For source discovery across a corpus, keep this pairwise analyzer as the reranking
and evidence layer. Add content-addressed storage, asynchronous chunk embedding,
approximate-nearest-neighbor retrieval, pairwise reranking, and privacy-aware
reviewer feedback.
