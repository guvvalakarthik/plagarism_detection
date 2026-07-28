# Reviewer dashboard

The browser interface exposes two deliberately separate workflows:

1. **Quick comparison** calls `POST /v1/analyze` and keeps both texts in memory.
2. **Workspace review** uses the authenticated ingestion, search, and feedback APIs.

## Workspace flow

1. Enter the workspace UUID and matching API key.
2. Upload a TXT, PDF, or DOCX document.
3. The dashboard polls the asynchronous job until it is `ready` or `failed`.
4. Search the indexed corpus with a candidate passage or idea.
5. Review ranked passages and record one of the supported feedback decisions.

The workspace UUID is retained only in `sessionStorage`. The API key remains in
the current page and is sent in the `x-api-key` header; it is never placed in a
URL or persisted by the dashboard.

## Search contract

`POST /v1/workspaces/{workspace_id}/search` accepts:

```json
{
  "query": "representative evaluation evidence",
  "limit": 10
}
```

Results include the document and passage identifiers, text offsets, embedding
method, and similarity score. The endpoint uses the same embedder as ingestion,
so stored and query vector dimensions cannot drift within one runtime.

The UI creates all result content with DOM `textContent`; uploaded document text
is never interpreted as HTML.
