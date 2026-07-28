# Hybrid retrieval and reranking

Pairwise analysis assumes a source is already known. Corpus retrieval discovers
which source passages deserve pairwise evidence analysis.

## Pipeline

```text
candidate passage
      |
      +-- BM25 lexical ranking
      |
      +-- sentence-transformer dense ranking
                |
                v
       reciprocal-rank fusion
                |
                v
        top candidate passages
                |
                v
       cross-encoder reranking
                |
                v
       pairwise evidence analyzer
```

`HybridRetriever` accepts embedder and reranker protocols, allowing deterministic
test doubles, local models, or managed inference without coupling core logic to a
specific vendor.

## Lightweight default

The local corpus API uses BM25 plus a deterministic character n-gram hashing
embedder. This tests the complete dense-retrieval path without downloading a
model. It is explicitly not described as semantic understanding.

## Transformer configuration

Install the optional dependency:

```bash
pip install -e ".[semantic]"
```

Then construct:

```python
from plagiarism_detection.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    SentenceTransformerEmbedder,
)

retriever = HybridRetriever(
    embedder=SentenceTransformerEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    reranker=CrossEncoderReranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ),
)
```

Model names and versions must be recorded with every benchmark report. Model
downloads should happen during image construction or deployment provisioning,
not unexpectedly during request handling.

## API

- `POST /v1/corpus/index` chunks and indexes local documents in memory.
- `POST /v1/corpus/search` returns ranked passages with lexical rank, dense rank,
  reranker score, document offsets, and method identity.

The in-memory endpoint is for local evaluation. The next storage layer replaces
it with PostgreSQL and pgvector without changing the retrieval result contract.
