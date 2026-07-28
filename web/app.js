const byId = (id) => document.getElementById(id);
const percent = (value) => `${Math.round(value * 100)}%`;
const delay = (milliseconds) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const verdictLabels = {
  low_overlap: "Low overlap",
  review_recommended: "Review recommended",
  high_overlap: "High overlap",
};

const analysisForm = byId("analysis-form");
const analyzeButton = byId("analyze");
const results = byId("results");
const evidenceList = byId("evidence");
const source = byId("source");
const candidate = byId("candidate");
const workspaceIdInput = byId("workspace-id");
const apiKeyInput = byId("api-key");
const uploadForm = byId("upload-form");
const fileInput = byId("document-file");
const uploadButton = byId("upload-document");
const searchForm = byId("search-form");
const searchButton = byId("search-workspace");
const workspaceResults = byId("workspace-results");
const uuidPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

let toastTimer;

function showToast(message, tone = "neutral") {
  const toast = byId("toast");
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.dataset.tone = tone;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 5000);
}

function errorMessage(payload, fallback) {
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg).join("; ");
  }
  return payload?.detail || fallback;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(errorMessage(payload, `Request failed with ${response.status}`));
  }
  return payload;
}

function workspaceCredentials() {
  const workspaceId = workspaceIdInput.value.trim();
  const apiKey = apiKeyInput.value.trim();
  if (!uuidPattern.test(workspaceId)) {
    throw new Error("Enter a valid workspace UUID.");
  }
  if (apiKey.length < 16) {
    throw new Error("Enter the API key configured for this workspace.");
  }
  return { workspaceId, apiKey };
}

function workspaceRequest(path, options = {}) {
  const { workspaceId, apiKey } = workspaceCredentials();
  const headers = new Headers(options.headers || {});
  headers.set("x-api-key", apiKey);
  return requestJson(
    `/v1/workspaces/${encodeURIComponent(workspaceId)}${path}`,
    { ...options, headers },
  );
}

for (const button of document.querySelectorAll(".mode-button")) {
  button.addEventListener("click", () => {
    const mode = button.dataset.mode;
    for (const candidateButton of document.querySelectorAll(".mode-button")) {
      const active = candidateButton === button;
      candidateButton.classList.toggle("is-active", active);
      candidateButton.setAttribute("aria-pressed", String(active));
    }
    byId("compare-panel").hidden = mode !== "compare";
    byId("workspace-panel").hidden = mode !== "workspace";
  });
}

for (const field of [source, candidate]) {
  const count = byId(`${field.id}-count`);
  field.addEventListener("input", () => {
    count.textContent = field.value.length.toLocaleString();
  });
}

function evidenceCard(match) {
  const article = document.createElement("article");
  article.className = "evidence-item";

  const score = document.createElement("div");
  score.className = "evidence-score";
  score.textContent =
    `${percent(match.similarity)} · ${match.match_type.replace("_", " ")}`;

  const sourceColumn = document.createElement("div");
  const sourceHeading = document.createElement("h4");
  sourceHeading.textContent =
    `Source · chars ${match.source_start}–${match.source_end}`;
  const sourceQuote = document.createElement("blockquote");
  sourceQuote.textContent = match.source_text;
  sourceColumn.append(sourceHeading, sourceQuote);

  const candidateColumn = document.createElement("div");
  const candidateHeading = document.createElement("h4");
  candidateHeading.textContent =
    `Candidate · chars ${match.candidate_start}–${match.candidate_end}`;
  const candidateQuote = document.createElement("blockquote");
  candidateQuote.textContent = match.candidate_text;
  candidateColumn.append(candidateHeading, candidateQuote);

  article.append(score, sourceColumn, candidateColumn);
  return article;
}

function renderAnalysis(data) {
  byId("verdict").textContent = verdictLabels[data.verdict] ?? data.verdict;
  byId("interpretation").textContent = data.score_interpretation;
  byId("score").textContent = Math.round(data.similarity_score * 100);
  byId("lexical").textContent = percent(data.lexical_similarity);
  byId("character").textContent = percent(data.character_similarity);
  byId("coverage").textContent = percent(data.candidate_coverage);
  byId("evidence-count").textContent =
    `${data.evidence.length} ${data.evidence.length === 1 ? "match" : "matches"}`;

  evidenceList.replaceChildren();
  if (data.evidence.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No passage exceeded the evidence threshold.";
    evidenceList.append(empty);
  } else {
    evidenceList.append(...data.evidence.map(evidenceCard));
  }

  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

analysisForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  analyzeButton.disabled = true;
  analyzeButton.firstElementChild.textContent = "Analyzing…";

  try {
    const data = await requestJson("/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source.value, candidate: candidate.value }),
    });
    renderAnalysis(data);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.firstElementChild.textContent = "Analyze overlap";
  }
});

const savedWorkspace = window.sessionStorage.getItem("sourcelens.workspace");
if (savedWorkspace) {
  workspaceIdInput.value = savedWorkspace;
}

byId("save-connection").addEventListener("click", () => {
  try {
    const { workspaceId } = workspaceCredentials();
    window.sessionStorage.setItem("sourcelens.workspace", workspaceId);
    byId("connection-status").textContent = "Workspace credentials ready";
    byId("connection-status").dataset.state = "ready";
    showToast("Workspace selected. The API key remains in this tab.", "success");
  } catch (error) {
    byId("connection-status").textContent = "Connection details incomplete";
    byId("connection-status").dataset.state = "error";
    showToast(error.message, "error");
  }
});

fileInput.addEventListener("change", () => {
  byId("file-label").textContent =
    fileInput.files[0]?.name || "Choose a document";
});

function updateJob(job) {
  const panel = byId("job-status");
  const title = byId("job-title");
  const detail = byId("job-detail");
  const progress = byId("job-progress");
  const indicator = byId("job-indicator");
  const progressByStatus = {
    queued: 20,
    processing: 62,
    ready: 100,
    failed: 100,
  };

  panel.hidden = false;
  panel.dataset.state = job.status;
  indicator.dataset.state = job.status;
  title.textContent =
    job.status === "ready"
      ? "Indexed and ready"
      : job.status === "failed"
        ? "Ingestion failed"
        : job.status[0].toUpperCase() + job.status.slice(1);
  detail.textContent =
    job.status === "ready"
      ? `Document ${job.document_id}`
      : job.status === "failed"
        ? job.error_code || "Processing failed"
        : job.filename;
  progress.style.width = `${progressByStatus[job.status] || 10}%`;
}

async function pollJob(jobId) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const job = await workspaceRequest(`/jobs/${encodeURIComponent(jobId)}`);
    updateJob(job);
    if (job.status === "ready" || job.status === "failed") {
      return job;
    }
    await delay(750);
  }
  throw new Error("The ingestion job is still running. Check its status later.");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    showToast("Choose a TXT, PDF, or DOCX file first.", "error");
    return;
  }

  uploadButton.disabled = true;
  uploadButton.textContent = "Uploading…";
  try {
    const body = new FormData();
    body.append("file", file);
    const job = await workspaceRequest("/documents", { method: "POST", body });
    updateJob(job);
    const completed = await pollJob(job.job_id);
    if (completed.status === "failed") {
      throw new Error(`Document processing failed: ${completed.error_code}`);
    }
    showToast("Document indexed. Search the workspace for evidence.", "success");
    byId("search-query").focus();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload & index";
  }
});

function feedbackForm(hit) {
  const form = document.createElement("form");
  form.className = "feedback-form";

  const decisionLabel = document.createElement("label");
  decisionLabel.className = "field";
  const decisionText = document.createElement("span");
  decisionText.textContent = "Reviewer decision";
  const decision = document.createElement("select");
  const decisions = [
    ["accepted_match", "Accepted match"],
    ["dismissed", "Dismissed"],
    ["properly_cited", "Properly cited"],
    ["common_phrase", "Common phrase"],
  ];
  for (const [value, label] of decisions) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    decision.append(option);
  }
  decisionLabel.append(decisionText, decision);

  const noteLabel = document.createElement("label");
  noteLabel.className = "field grow";
  const noteText = document.createElement("span");
  noteText.textContent = "Optional note";
  const note = document.createElement("input");
  note.type = "text";
  note.maxLength = 2000;
  note.placeholder = "Add review context";
  noteLabel.append(noteText, note);

  const submit = document.createElement("button");
  submit.className = "feedback-button";
  submit.type = "submit";
  submit.textContent = "Save feedback";

  form.append(decisionLabel, noteLabel, submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submit.disabled = true;
    submit.textContent = "Saving…";
    try {
      await workspaceRequest("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: hit.document_id,
          evidence_id: hit.passage_id,
          decision: decision.value,
          note: note.value.trim() || null,
        }),
      });
      submit.textContent = "Feedback saved";
      form.dataset.saved = "true";
      decision.disabled = true;
      note.disabled = true;
      showToast("Reviewer feedback recorded.", "success");
    } catch (error) {
      submit.disabled = false;
      submit.textContent = "Save feedback";
      showToast(error.message, "error");
    }
  });
  return form;
}

function workspaceHit(hit, index) {
  const article = document.createElement("article");
  article.className = "workspace-hit";

  const heading = document.createElement("div");
  heading.className = "hit-heading";
  const rank = document.createElement("span");
  rank.className = "hit-rank";
  rank.textContent = String(index + 1).padStart(2, "0");
  const identity = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = `Document ${hit.document_id.slice(0, 8)}`;
  const metadata = document.createElement("p");
  metadata.textContent =
    `Chars ${hit.start_offset}–${hit.end_offset} · ${hit.embedding_method}`;
  identity.append(title, metadata);
  const score = document.createElement("strong");
  score.className = "hit-score";
  score.textContent = percent(hit.score);
  heading.append(rank, identity, score);

  const passage = document.createElement("blockquote");
  passage.textContent = hit.content;
  article.append(heading, passage, feedbackForm(hit));
  return article;
}

function renderWorkspaceResults(data) {
  const hits = data.hits || [];
  byId("search-summary").hidden = false;
  byId("search-count").textContent =
    `${hits.length} ${hits.length === 1 ? "passage" : "passages"}`;
  byId("search-method").textContent = data.method;
  workspaceResults.replaceChildren();

  if (hits.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty workspace-empty";
    empty.textContent = "No indexed passage matched this query.";
    workspaceResults.append(empty);
    return;
  }
  workspaceResults.append(...hits.map(workspaceHit));
}

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  searchButton.disabled = true;
  searchButton.textContent = "Searching…";
  try {
    const data = await workspaceRequest("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: byId("search-query").value,
        limit: Number(byId("search-limit").value),
      }),
    });
    renderWorkspaceResults(data);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Search evidence";
  }
});
