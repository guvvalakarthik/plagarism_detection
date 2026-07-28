CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    workspace_id uuid NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    document_id text NOT NULL,
    content_sha256 char(64) NOT NULL,
    original_filename text NOT NULL,
    character_count integer NOT NULL CHECK (character_count >= 0),
    status text NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, document_id),
    UNIQUE (workspace_id, content_sha256)
);

CREATE TABLE IF NOT EXISTS passages (
    workspace_id uuid NOT NULL,
    passage_id text NOT NULL,
    document_id text NOT NULL,
    content text NOT NULL,
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    embedding vector(256) NOT NULL,
    embedding_method text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, passage_id),
    FOREIGN KEY (workspace_id, document_id)
        REFERENCES documents(workspace_id, document_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS passages_workspace_document_idx
    ON passages (workspace_id, document_id);

CREATE INDEX IF NOT EXISTS passages_embedding_hnsw_idx
    ON passages USING hnsw (embedding vector_cosine_ops);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE passages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS documents_workspace_policy ON documents;
CREATE POLICY documents_workspace_policy ON documents
    USING (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    )
    WITH CHECK (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    );

DROP POLICY IF EXISTS passages_workspace_policy ON passages;
CREATE POLICY passages_workspace_policy ON passages
    USING (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    )
    WITH CHECK (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    );
