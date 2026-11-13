/* Compare page: up to four molecules side by side plus a trend chart. */

const PRESETS = {
  "Alcohol chain": ["ethanol", "1-butanol", "1-octanol", "hexane"],
  "Benzene family": ["benzene", "phenol", "benzoic acid", "naphthalene"],
  "Sugar vs fat-like": ["glucose", "sucrose", "glycerol", "cholesterol"],
  "Painkillers": ["aspirin", "paracetamol", "ibuprofen", "caffeine"],
};

document.addEventListener("DOMContentLoaded", () => {
  const inputs = document.getElementById("inputs");
  inputs.innerHTML = [1, 2, 3, 4].map((i) =>
    `<label class="field"><span class="kicker">Molecule ${i}${i > 2 ? " (optional)" : ""}</span>
     <input id="m${i}" maxlength="300" spellcheck="false" placeholder="name or SMILES"></label>`).join("");
  const presets = document.getElementById("presets");
  presets.innerHTML = Object.keys(PRESETS).map((p) => `<button class="chip" data-p="${esc(p)}">${esc(p)}</button>`).join("");
  presets.onclick = (e) => { if (e.target.dataset.p) { fill(PRESETS[e.target.dataset.p]); run(); } };
  document.getElementById("go").onclick = run;
  inputs.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

  const fromUrl = new URLSearchParams(location.search).getAll("m");
  fill(fromUrl.length >= 2 ? fromUrl : PRESETS["Alcohol chain"]);
  run();
});

function fill(list) {
  [1, 2, 3, 4].forEach((i) => { document.getElementById(`m${i}`).value = list[i - 1] || ""; });
}

let last = null;
async function run() {
  const queries = [1, 2, 3, 4].map((i) => document.getElementById(`m${i}`).value.trim()).filter(Boolean);
  const out = document.getElementById("out");
  if (queries.length < 2) { out.innerHTML = `<div style="margin-top:20px">${errorBox({ message: "Please enter at least two molecules." })}</div>`; return; }
  out.innerHTML = loadingBox("Comparing…");
  try {
    const { results } = await api("/api/compare", { queries });
    last = results;
    history.replaceState(null, "", "?" + queries.map((q) => `m=${encodeURIComponent(q)}`).join("&"));
    out.innerHTML = `
      <div class="grid grid-${results.length === 3 ? 3 : results.length === 2 ? 2 : 4} fade-in" style="margin-top:20px">${results.map(card).join("")}</div>
      <div class="card fade-in" style="margin-top:20px">
        <div class="card-title"><h3>Trend</h3><span class="small muted">bars show the ${Math.round((results.find((r) => !r.error) || { coverage: 0.9 }).coverage * 100)}% range</span></div>
        <div id="trend" class="plot" style="min-height:340px"></div>
      </div>`;
    drawTrend();
  } catch (err) {
    out.innerHTML = `<div style="margin-top:20px">${errorBox(err)}</div>`;
  }
}

function card(r) {
  if (r.error) return `<div class="card compare-card"><p class="mol-name">${esc(r.query)}</p>${errorBox(r.error)}</div>`;
  const top = r.explanation[0] || "";
  return `<div class="card compare-card">
    <p class="mol-name" style="font-size:1.3rem">${esc(capitalize(r.name))}</p>
    <div class="mol-meta">${formula(r.formula)} · ${r.mol_weight.toFixed(0)} g/mol</div>
    <div class="mol-paper" style="margin:12px 0">${r.svg_explained}</div>
    <div class="big-number">${r.logS.toFixed(2)}<small>logS</small></div>
    <div style="margin:10px 0">${badge(r.category)}</div>
    <div class="small muted">≈ ${fmtConc(r.g_per_L)} · range ${r.interval[0].toFixed(1)} to ${r.interval[1].toFixed(1)}</div>
    ${scaleBar(r)}
    <p class="small" style="margin-top:14px">${esc(top)}</p>
    ${r.domain.in_domain ? "" : `<p class="small" style="color:var(--warn-ink)">⚠ Outside the model's experience</p>`}
  </div>`;
}

function drawTrend() {
  const ok = (last || []).filter((r) => !r.error);
  if (!ok.length || !window.Plotly) return;
  const colors = ok.map((r) => css(`--${CLASS_KEY[r.category]}`));
  Plotly.newPlot("trend", [{
    type: "bar", x: ok.map((r) => capitalize(r.name)), y: ok.map((r) => r.logS), marker: { color: colors },
    error_y: { type: "data", symmetric: false, array: ok.map((r) => r.interval[1] - r.logS),
      arrayminus: ok.map((r) => r.logS - r.interval[0]), color: css("--ink-3"), thickness: 1.5, width: 8 },
    hovertemplate: "%{x}<br>logS %{y:.2f}<extra></extra>",
  }], plotLayout({ yaxis: { title: "Predicted logS (log mol/L)", gridcolor: css("--line"), zerolinecolor: css("--ink-3") }, showlegend: false }), PLOT_CONFIG);
}
document.addEventListener("themechange", drawTrend);
