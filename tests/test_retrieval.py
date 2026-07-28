import pytest

from plagiarism_detection.retrieval import (
    HashingNgramEmbedder,
    HybridRetriever,
    SourcePassage,
    chunk_document,
)


class TopicEmbedder:
    name = "topic-test"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float(any(word in lowered for word in ("car", "automobile", "vehicle"))),
                    float(any(word in lowered for word in ("garden", "flower", "soil"))),
                ]
            )
        return vectors


class ReverseReranker:
    name = "reverse-test"

    def score(self, query, passages):
        del query
        return [float(index) for index, _ in enumerate(passages)]


def passages():
    return [
        SourcePassage("one", "cars", "An automobile requires regular maintenance.", 0, 43),
        SourcePassage("two", "garden", "Garden soil supports flowers and vegetables.", 0, 43),
        SourcePassage("three", "ml", "Models require representative evaluation data.", 0, 46),
    ]


def test_chunking_preserves_document_offsets() -> None:
    text = " ".join(f"token{index}" for index in range(30))

    chunks = chunk_document("document", text, max_tokens=10, overlap_tokens=2)

    assert len(chunks) == 4
    assert chunks[0].text == text[chunks[0].start : chunks[0].end]
    assert chunks[1].text.split()[0] == "token8"
    assert chunks[-1].end == len(text)


def test_bm25_retrieves_lexical_match() -> None:
    retriever = HybridRetriever()
    retriever.index(passages())

    hits = retriever.search("representative model evaluation", top_k=1)

    assert hits[0].passage.document_id == "ml"
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_rank is None


def test_dense_channel_retrieves_semantic_synonym() -> None:
    retriever = HybridRetriever(embedder=TopicEmbedder(), lexical_weight=0.0)
    retriever.index(passages())

    hits = retriever.search("vehicle servicing", top_k=1)

    assert hits[0].passage.document_id == "cars"
    assert hits[0].dense_rank == 1
    assert hits[0].method == "bm25+topic-test"


def test_reranker_changes_candidate_order() -> None:
    retriever = HybridRetriever(reranker=ReverseReranker())
    retriever.index(passages())

    hits = retriever.search("evaluation", top_k=1, candidate_k=3)

    assert hits[0].reranker_score == 2.0
    assert hits[0].method.endswith("reverse-test")


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingNgramEmbedder(32)

    first, second = embedder.encode(["same text", "same text"])

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_retrieval_validates_configuration() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(lexical_weight=0, dense_weight=0)
    with pytest.raises(ValueError):
        chunk_document("document", "some words", max_tokens=8, overlap_tokens=8)
