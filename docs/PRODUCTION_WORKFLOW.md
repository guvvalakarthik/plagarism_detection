# Production review workflow

## Security boundary

Production endpoints are fail-closed. `DATABASE_URL` and
`SOURCELENS_API_KEYS_JSON` must both exist; otherwise they return HTTP 503.

The key configuration maps raw deployment secrets to workspace UUIDs:

```json
{
  "replace-with-a-secret-at-least-20-characters": "c6222bd5-2554-4df7-a33a-f407792335a8"
}
```

Keys are SHA-256 digested at configuration load and compared in constant time.
They are never returned or written to job records. A key bound to one workspace
receives HTTP 403 for another workspace.

Use a managed secret store in production. The JSON environment format is a
portable deployment adapter, not a recommendation to commit secrets.

## Upload lifecycle

1. `POST /v1/workspaces/{workspace}/documents` accepts TXT, PDF, or DOCX.
2. The API reads at most 10 MB and returns HTTP 202 with a job ID.
3. A bounded worker pool extracts at most 500,000 characters.
4. SHA-256 ingestion deduplicates content within the workspace.
5. Chunk embeddings and offsets are committed to PostgreSQL/pgvector.
6. The persisted job becomes `ready` or `failed`.
7. `GET /v1/workspaces/{workspace}/jobs/{job}` returns safe status information.

Raw uploads are held only for the lifetime of the background task. They are not
stored in job tables, metrics, or logs.

For multi-instance deployment, replace `ThreadPoolExecutor` with a durable queue
such as a managed task service while retaining the workflow/repository contracts.

## Reviewer feedback

`POST /v1/workspaces/{workspace}/feedback` persists one of:

- `accepted_match`
- `dismissed`
- `properly_cited`
- `common_phrase`

Feedback remains workspace-scoped under PostgreSQL row-level security. It is
review evidence, not automatically trusted model training data.

## Operational controls

- Request IDs are accepted only from a bounded safe character set or regenerated.
- Per-workspace sliding-window rate limits default to 60 requests/minute.
- Prometheus text metrics contain outcome/decision counts but no document text.
- Extraction errors are reduced to stable error codes.
- PDF and DOCX content signatures must match their extensions.

An external malware scanner should be placed before this API in production.
