/* Batch page: many molecules at once, from a pasted list or a CSV/TXT file. */

let ROWS = [];
const SAMPLE = ["ethanol", "1-octanol", "glucose", "caffeine", "aspirin", "ibuprofen", "naphthalene", "cholesterol",
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2ccccc2c1", "NaCl", "AgCl", "CaCO3"].join("\n");

document.addEventListener("DOMContentLoaded", () => {
  const area = document.getElementById("list");
  document.getElementById("sample").onclick = () => { area.value = SAMPLE; };
  document.getElementById("file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    area.value = parseFile(await f.text()).join("\n");
    e.target.value = "";
  };
  document.getElementById("go").onclick = run;
  document.getElementById("csv").onclick = () => downloadText("soluai_predictions.csv", toCsv());
});
document.addEventListener("modelchange", () => { if (ROWS.length) run(); });

/* One molecule per line. For a CSV with a header, the "smiles" (or "name") column is used, otherwise the first column. */
function parseFile(text) {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim());
  if (!lines.length) return [];
  const split = (l) => (l.match(/("([^"]|"")*"|[^,;\t]*)(?:[,;\t]|$)/g) || [l]).map((c) => c.replace(/[,;\t]$/, "").replace(/^"|"$/g, "").replace(/""/g, '"').trim());
  const head = split(lines[0]).map((h) => h.toLowerCase());
  let col = head.indexOf("smiles");
  if (col < 0) col = head.indexOf("name");
  if (col < 0) col = head.indexOf("molecule");
  const body = col >= 0 ? lines.slice(1) : lines;
  return body.map((l) => split(l)[Math.max(col, 0)]).filter(Boolean);
}

async function run() {
  const queries = document.getElementById("list").value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const out = document.getElementById("out");
  document.getElementById("csv").hidden = true;
  if (!queries.length) { out.innerHTML = errorBox({ message: T("Paste at least one molecule (one per line).", "دست‌کم یک مولکول وارد کنید (هر خط یک مولکول).") }); return; }
  if (queries.length > 200) { out.innerHTML = errorBox({ message: T(`That is ${queries.length} lines; the limit is 200 per batch.`, `${queries.length} خط وارد شده؛ سقف هر بار 200 خط است.`) }); return; }
  out.innerHTML = loadingBox(T(`Predicting ${queries.length} molecules…`, `در حال پیش‌بینی ${queries.length} مولکول…`));
  try {
    const r = await api("/api/batch", { queries });
    ROWS = r.rows;
    out.innerHTML = `<p class="muted">${T(`${r.n_ok} predicted by the AI model · ${r.n_salts} salts answered by the rules · ${r.n_failed} not recognised`,
      `${r.n_ok} مورد با مدل هوش مصنوعی · ${r.n_salts} نمک با قواعد · ${r.n_failed} مورد ناشناخته`)}</p>
      <div class="card table-wrap" style="padding:8px 16px"><table><thead><tr><th>#</th><th>${T("Input", "ورودی")}</th><th>${T("Formula", "فرمول")}</th><th class="num">logS</th><th class="num">${T("90% range", "بازهٔ 90%")}</th>
      <th class="num">${T("Amount", "مقدار")}</th><th>${T("Class", "رده")}</th><th class="num">${T("Measured", "اندازه‌گیری")}</th><th>${T("Trust", "اعتماد")}</th></tr></thead><tbody>${ROWS.map(row).join("")}</tbody></table></div>`;
    document.getElementById("csv").hidden = false;
  } catch (err) { out.innerHTML = errorBox(err); }
}

function row(r, i) {
  const q = `<td dir="auto" class="mono small" style="max-width:260px;overflow-wrap:anywhere">${esc(r.query)}</td>`;
  if (r.kind === "error") return `<tr><td>${i + 1}</td>${q}<td colspan="7" style="color:var(--warn-ink)">${esc(r.error)}</td></tr>`;
  if (r.kind === "ionic") return `<tr><td>${i + 1}</td>${q}<td class="ltr">${esc(r.formula)}</td><td class="num">–</td><td class="num">–</td>
    <td class="num ltr">${r.measured_g_per_100mL === null ? "–" : `${r.measured_g_per_100mL >= 0.01 ? r.measured_g_per_100mL : sci(r.measured_g_per_100mL)} g/100 mL`}</td>
    <td><b class="verdict ${r.status}" style="font-size:1rem">${esc(r.status_label)}</b></td><td class="num">–</td><td><span class="pill">${T("rules", "قواعد")}</span></td></tr>`;
  return `<tr><td>${i + 1}</td><td><a href="/?q=${encodeURIComponent(r.smiles)}" target="_blank" dir="auto">${esc(r.query)}</a></td><td class="ltr">${formula(r.formula)}</td>
    <td class="num ltr"><b>${num(r.logS)}</b></td><td class="num ltr">${num(r.low, 1)} … ${num(r.high, 1)}</td><td class="num ltr">${fmtConc(r.g_per_L)}</td>
    <td>${badge(r.category)}</td><td class="num ltr">${r.measured === null ? "–" : num(r.measured)}</td>
    <td>${r.confidence ? confidencePill(r.confidence) : r.in_domain ? `<span class="pill accent">${T("familiar", "آشنا")}</span>` : `<span class="pill" style="color:var(--warn-ink)">⚠ ${T("unfamiliar", "ناآشنا")}</span>`}</td></tr>`;
}

function toCsv() {
  const cols = ["query", "kind", "name", "smiles", "formula", "mol_weight", "logS", "low", "high", "g_per_L", "category", "usp", "measured", "classic", "confidence", "in_domain", "similarity", "status", "measured_g_per_100mL", "error"];
  const cell = (v) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [cols.join(","), ...ROWS.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
}
