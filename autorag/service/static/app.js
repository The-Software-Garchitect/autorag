const questionInput = document.querySelector("#question");
const form = document.querySelector("#ask-form");
const askBtn = document.querySelector("#ask-btn");
const answerEl = document.querySelector("#answer");
const chipsEl = document.querySelector("#chips");
const statusEl = document.querySelector("#status");
const chartEl = document.querySelector("#chart");
const tableEl = document.querySelector("#tune-table");
const tuneCopy = document.querySelector("#tune-copy");

let ready = false;
let lastQuestion = "";

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

async function pollHealth() {
  try {
    const response = await fetch("/health");
    if (response.status === 503) {
      setStatus("Indexing the handbook…");
      setTimeout(pollHealth, 700);
      return;
    }
    const body = await response.json();
    if (!response.ok) {
      setStatus(body.detail || "The handbook failed to index.", "error");
      return;
    }
    ready = true;
    const label = body.config ? `Ready · ${body.config} config · ${body.generator}` : "Ready";
    setStatus(label, "ready");
    askBtn.disabled = false;
  } catch (error) {
    setStatus("Cannot reach the handbook service.", "error");
  }
}

async function loadSuggestions() {
  try {
    const response = await fetch("/suggested-questions");
    if (!response.ok) return;
    const body = await response.json();
    if (!body.questions || !body.questions.length) return;
    chipsEl.innerHTML = "";
    body.questions.slice(0, 6).forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.q = question;
      button.textContent = shortLabel(question);
      button.addEventListener("click", () => submitQuestion(question, button));
      chipsEl.appendChild(button);
    });
  } catch (error) {
    /* chips already in the page */
  }
}

function shortLabel(question) {
  const known = {
    "How many PTO days after 3 years?": "PTO after 3 years",
    "What's the client dinner expense limit?": "Client dinner limit",
    "How do I request parental leave?": "Parental leave",
    "What is the lodging per diem in Chicago?": "Chicago per diem",
    "Which health plan has an individual deductible of $500?": "$500 deductible",
    "What is the VPN gateway hostname?": "VPN gateway",
    "How much does Acme match on the 401k?": "401k match",
  };
  return known[question] || question.replace(/\?$/, "").slice(0, 42);
}

function markChip(question) {
  chipsEl.querySelectorAll("button").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.q === question ? "true" : "false");
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value.trim());
});

chipsEl.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => submitQuestion(button.dataset.q, button));
});

async function submitQuestion(question) {
  if (!question) return;
  if (!ready) {
    answerEl.className = "answer";
    answerEl.innerHTML = '<p class="empty-copy">Still indexing the handbook. Try again in a moment.</p>';
    return;
  }
  lastQuestion = question;
  questionInput.value = question;
  markChip(question);
  askBtn.disabled = true;
  answerEl.className = "answer";
  answerEl.innerHTML = '<p class="empty-copy">Looking through the handbook…</p>';
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, variant: "best" }),
    });
    const body = await response.json();
    if (!response.ok) {
      renderError(body.detail || "The question didn’t go through.");
      return;
    }
    renderAnswer(body);
  } catch (error) {
    renderError("The handbook service didn’t respond.");
  } finally {
    askBtn.disabled = false;
  }
}

function renderError(message) {
  answerEl.className = "answer refused";
  answerEl.innerHTML = `<p class="answer-kicker">Something went wrong</p><p class="answer-text"></p>`;
  answerEl.querySelector(".answer-text").textContent = message;
}

function renderAnswer(body) {
  answerEl.className = body.refused ? "answer refused" : "answer";
  const kicker = body.refused ? "Not in the handbook" : "From the handbook";
  const percent = Math.round((body.confidence || 0) * 100);
  const citations = (body.citations || [])
    .map((citation) => {
      const badge =
        citation.status && citation.status !== "current"
          ? `<span class="badge">${escapeHtml(citation.status)}</span>`
          : "";
      return `<article class="citation">
        <p class="meta">${escapeHtml(citation.title)}${badge}</p>
        <p></p>
      </article>`;
    })
    .join("");
  answerEl.innerHTML = `
    <p class="answer-kicker">${kicker}</p>
    <p class="answer-text"></p>
    <div class="confidence"><span>Support ${percent}%</span><div class="bar"><span style="width:${percent}%"></span></div></div>
    <div class="citations">${citations}</div>
    <div class="actions">
      <div class="vote">
        <button type="button" data-vote="up">Helpful</button>
        <button type="button" data-vote="down">Not helpful</button>
      </div>
      <button type="button" class="ghost" id="compare">Compare the untuned baseline</button>
    </div>
    <p class="note" id="vote-note"></p>
    <div id="compare-slot"></div>
  `;
  answerEl.querySelector(".answer-text").textContent = body.answer;
  answerEl.querySelectorAll(".citation p:last-child").forEach((node, index) => {
    node.textContent = body.citations[index].snippet;
  });
  answerEl.querySelectorAll("[data-vote]").forEach((button) => {
    button.addEventListener("click", () => sendFeedback(body, button.dataset.vote));
  });
  answerEl.querySelector("#compare").addEventListener("click", () => loadBaseline(body.question));
}

async function sendFeedback(body, rating) {
  const note = document.querySelector("#vote-note");
  answerEl.querySelectorAll("[data-vote]").forEach((button) => {
    button.disabled = true;
  });
  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: body.question,
        answer: body.answer,
        rating,
        doc_ids: (body.citations || []).map((citation) => citation.doc_id),
      }),
    });
    if (!response.ok) throw new Error("not stored");
    note.textContent =
      rating === "up"
        ? "Saved. A thumbs-up is added to the eval set the next time tuning runs."
        : "Saved. A thumbs-down is reviewed on the next tuning run.";
  } catch (error) {
    note.textContent = "Could not store that vote.";
    answerEl.querySelectorAll("[data-vote]").forEach((button) => {
      button.disabled = false;
    });
  }
}

async function loadBaseline(question) {
  const slot = document.querySelector("#compare-slot");
  const button = document.querySelector("#compare");
  button.disabled = true;
  slot.innerHTML = '<p class="note">Scoring the untuned baseline…</p>';
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, variant: "baseline" }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "failed");
    slot.innerHTML = `<div class="compare"><h3>Untuned baseline</h3><p></p></div>`;
    slot.querySelector("p").textContent = body.answer;
  } catch (error) {
    slot.innerHTML = '<p class="note">The baseline config isn’t available yet.</p>';
  }
}

async function loadRuns() {
  try {
    const response = await fetch("/runs");
    const body = await response.json();
    drawRuns(body);
  } catch (error) {
    chartEl.innerHTML = '<p class="empty-chart">Tuning history is unavailable.</p>';
  }
}

function drawRuns(payload) {
  const runs = payload.runs || [];
  if (!runs.length) {
    chartEl.innerHTML = '<p class="empty-chart">No tuning run yet. Ask anyway — the default config is already loaded. Run <code>autorag tune --source data/acme</code> to fill this chart.</p>';
    tableEl.innerHTML = "";
    return;
  }
  const scores = runs.map((run) => run.metrics.composite);
  const bestSoFar = [];
  let best = -1;
  scores.forEach((score) => {
    best = Math.max(best, score);
    bestSoFar.push(best);
  });
  const improvement = payload.improvement || 0;
  const direction = improvement > 0 ? "up" : "flat";
  tuneCopy.textContent =
    direction === "up"
      ? `Composite score moved from ${scores[0].toFixed(3)} to ${best.toFixed(3)} on ${payload.question_count} questions. A config is promoted only when it beats the current best.`
      : `Best composite is ${best.toFixed(3)} across ${runs.length} configs on ${payload.question_count} questions.`;

  const width = 520;
  const height = 180;
  const pad = 28;
  const min = Math.max(0, Math.min(...scores) - 0.05);
  const max = Math.min(1, Math.max(...bestSoFar) + 0.04);
  const x = (index) => pad + (index * (width - pad * 2)) / Math.max(1, runs.length - 1);
  const y = (value) => height - pad - ((value - min) / (max - min || 1)) * (height - pad * 2);
  const line = bestSoFar.map((value, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(1)} ${y(value).toFixed(1)}`).join(" ");
  const dots = runs
    .map((run, index) => {
      const promoted = run.promoted ? "#1d6a43" : "#8a8172";
      const radius = run.name === payload.best.name ? 5.5 : 3.5;
      return `<circle cx="${x(index).toFixed(1)}" cy="${y(scores[index]).toFixed(1)}" r="${radius}" fill="${promoted}"><title>${run.name}: ${scores[index].toFixed(3)}</title></circle>`;
    })
    .join("");
  chartEl.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Composite score across tuning runs">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d4cbb8"/>
      <path d="${line}" fill="none" stroke="#1d6a43" stroke-width="2"/>
      ${dots}
      <text x="${pad}" y="16" fill="#5f584e" font-size="11">best so far</text>
    </svg>`;

  const rows = runs
    .map((run) => {
      const metrics = run.metrics;
      const cls = run.name === payload.best.name ? "best" : "";
      return `<tr class="${cls}">
        <td>${escapeHtml(run.name)}</td>
        <td class="num">${metrics.composite.toFixed(3)}</td>
        <td class="num">${metrics.correctness.toFixed(3)}</td>
        <td class="num">${metrics.hit_rate.toFixed(3)}</td>
        <td>${run.promoted ? "yes" : ""}</td>
      </tr>`;
    })
    .join("");
  tableEl.innerHTML = `
    <table>
      <thead><tr><th>Config</th><th class="num">Score</th><th class="num">Correct</th><th class="num">Hit</th><th>Kept</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

askBtn.disabled = true;
pollHealth();
loadSuggestions();
loadRuns();
