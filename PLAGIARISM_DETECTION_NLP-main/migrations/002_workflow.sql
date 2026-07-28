CREATE TABLE IF NOT EXISTS ingestion_jobs (
    workspace_id uuid NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
    job_id text NOT NULL,
    filename text NOT NULL,
    status text NOT NULL CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    document_id text,
    error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, job_id),
    FOREIGN KEY (workspace_id, document_id)
        REFERENCES documents(workspace_id, document_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS review_feedback (
    workspace_id uuid NOT NULL,
    feedback_id text NOT NULL,
    document_id text NOT NULL,
    evidence_id text NOT NULL,
    decision text NOT NULL CHECK (
        decision IN ('accepted_match', 'dismissed', 'properly_cited', 'common_phrase')
    ),
    note text CHECK (char_length(note) <= 2000),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, feedback_id),
    FOREIGN KEY (workspace_id, document_id)
        REFERENCES documents(workspace_id, document_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ingestion_jobs_workspace_status_idx
    ON ingestion_jobs (workspace_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS review_feedback_workspace_document_idx
    ON review_feedback (workspace_id, document_id, created_at DESC);

ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS ingestion_jobs_workspace_policy ON ingestion_jobs;
CREATE POLICY ingestion_jobs_workspace_policy ON ingestion_jobs
    USING (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    )
    WITH CHECK (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    );

DROP POLICY IF EXISTS review_feedback_workspace_policy ON review_feedback;
CREATE POLICY review_feedback_workspace_policy ON review_feedback
    USING (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    )
    WITH CHECK (
        workspace_id = nullif(current_setting('app.workspace_id', true), '')::uuid
    );
