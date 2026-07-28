const form = document.querySelector("#analysis-form");
const button = document.querySelector("#analyze");
const results = document.querySelector("#results");
const evidenceList = document.querySelector("#evidence");
const source = document.querySelector("#source");
const candidate = document.querySelector("#candidate");

const percent = (value) => `${Math.round(value * 100)}%`;
const verdictLabels = {
  low_overlap: "Low overlap",
  review_recommended: "Review recommended",
  high_overlap: "High overlap",
};

for (const field of [source, candidate]) {
  const count = document.querySelector(`#${field.id}-count`);
  field.addEventListener("input", () => {
    count.textContent = field.value.length.toLocaleString();
  });
}

function evidenceCard(match) {
  const article = document.createElement("article");
  article.className = "evidence-item";

  const score = document.createElement("div");
  score.className = "evidence-score";
  score.textContent = `${percent(match.similarity)} · ${match.match_type.replace("_", " ")}`;

  const sourceColumn = document.createElement("div");
  const sourceHeading = document.createElement("h4");
  sourceHeading.textContent = `Source · chars ${match.source_start}–${match.source_end}`;
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

function render(data) {
  document.querySelector("#verdict").textContent =
    verdictLabels[data.verdict] ?? data.verdict;
  document.querySelector("#interpretation").textContent = data.score_interpretation;
  document.querySelector("#score").textContent = Math.round(data.similarity_score * 100);
  document.querySelector("#lexical").textContent = percent(data.lexical_similarity);
  document.querySelector("#character").textContent = percent(data.character_similarity);
  document.querySelector("#coverage").textContent = percent(data.candidate_coverage);
  document.querySelector("#evidence-count").textContent =
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.firstElementChild.textContent = "Analyzing…";

  try {
    const response = await fetch("/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source.value, candidate: candidate.value }),
    });
    const data = await response.json();
    if (!response.ok) {
      const message = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join("; ")
        : data.detail;
      throw new Error(message || "Analysis failed");
    }
    render(data);
  } catch (error) {
    window.alert(error.message);
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = "Analyze overlap";
  }
});
