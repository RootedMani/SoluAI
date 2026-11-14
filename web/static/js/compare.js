/* Compare page: up to four molecules side by side plus a trend chart. */

const PRESETS = [
  [T("Alcohol chain", "زنجیر الکل‌ها"), ["ethanol", "1-butanol", "1-octanol", "hexane"]],
  [T("Benzene family", "خانوادهٔ بنزن"), ["benzene", "phenol", "benzoic acid", "naphthalene"]],
  [T("Sugar vs fat-like", "قند در برابر چربی"), ["glucose", "sucrose", "glycerol", "cholesterol"]],
  [T("Painkillers", "مسکن‌ها"), ["aspirin", "paracetamol", "ibuprofen", "caffeine"]],
  [T("Organic vs salt", "آلی در برابر نمک"), ["sucrose", "NaCl", "naphthalene", "CaCO3"]],
];

document.addEventListener("DOMContentLoaded", () => {
  const inputs = document.getElementById("inputs");
  inputs.innerHTML = [1, 2, 3, 4].map((i) =>
    `<label class="field"><span class="kicker">${T("Molecule", "مولکول")} ${i}${i > 2 ? T(" (optional)", " (اختیاری)") : ""}</span>
     <input id="m${i}" dir="auto" maxlength="300" spellcheck="false" placeholder="${T("name or SMILES", "نام یا SMILES")}"></label>`).join("");
  const presets = document.getElementById("presets");
  presets.innerHTML = PRESETS.map(([p], i) => `<button class="chip" data-p="${i}">${esc(p)}</button>`).join("");
  presets.onclick = (e) => { if (e.target.dataset.p) { fill(PRESETS[e.target.dataset.p][1]); run(); } };
  document.getElementById("go").onclick = run;
  inputs.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

  const fromUrl = new URLSearchParams(location.search).getAll("m");
  fill(fromUrl.length >= 2 ? fromUrl : PRESETS[0][1]);
  run();
});

function fill(list) {
  [1, 2, 3, 4].forEach((i) => { document.getElementById(`m${i}`).value = list[i - 1] || ""; });
}

let last = null;
async function run() {
  const queries = [1, 2, 3, 4].map((i) => document.getElementById(`m${i}`).value.trim()).filter(Boolean);
  const out = document.getElementById("out");
  if (queries.length < 2) { out.innerHTML = `<div style="margin-top:20px">${errorBox({ message: T("Please enter at least two molecules.", "لطفاً دست‌کم دو مولکول وارد کنید.") })}</div>`; return; }
  out.innerHTML = loadingBox(T("Comparing…", "در حال مقایسه…"));
  try {
    const { results } = await api("/api/compare", { queries });
    last = results;
    history.replaceState(null, "", "?" + queries.map((q) => `m=${encodeURIComponent(q)}`).join("&"));
    const cov = Math.round((results.find((r) => r.coverage) || { coverage: 0.9 }).coverage * 100);
    out.innerHTML = `
      <div class="grid grid-${results.length === 3 ? 3 : results.length === 2 ? 2 : 4} fade-in" style="margin-top:20px">${results.map(card).join("")}</div>
      <div class="card fade-in" style="margin-top:20px">
        <div class="card-title"><h3>${T("Trend", "روند")}</h3><span class="small muted">${T(`bars show the ${cov}% range; dots are lab measurements`, `خط‌ها بازهٔ ${cov}% را نشان می‌دهند؛ نقطه‌ها اندازه‌گیری آزمایشگاهی‌اند`)}</span></div>
        <div id="trend" class="plot" style="min-height:340px"></div>
      </div>`;
    drawTrend();
  } catch (err) {
    out.innerHTML = `<div style="margin-top:20px">${errorBox(err)}</div>`;
  }
}

function card(r) {
  if (r.error) return `<div class="card compare-card"><p class="mol-name" dir="auto">${esc(r.query)}</p>${errorBox(r.error)}</div>`;
  if (r.kind === "ionic") {
    return `<div class="card compare-card">
      <p class="mol-name" style="font-size:1.3rem">${esc(capitalize(r.name))}</p>
      <div class="mol-meta">${esc(r.formula)} · ${r.molar_mass.toFixed(0)} g/mol</div>
      <div class="mol-paper" style="margin:12px 0"><div class="verdict ${r.status} ltr" style="font-size:2.2rem">${esc(r.formula)}</div></div>
      <div class="verdict ${r.status}" style="font-size:1.5rem">${esc(r.status_label)}</div>
      ${r.measured ? `<div class="small muted" style="margin-top:8px">${T("measured", "اندازه‌گیری")}: <span class="ltr">${r.measured.g_per_100mL >= 0.01 ? r.measured.g_per_100mL : sci(r.measured.g_per_100mL)} g/100 mL · logS ${num(r.measured.logS)}</span></div>` : ""}
      <p class="small" style="margin-top:14px">${esc(r.rule_text)}</p>
      <p class="small"><span class="pill accent">${T("solubility rules, not the AI", "قواعد انحلال‌پذیری، نه هوش مصنوعی")}</span></p>
    </div>`;
  }
  const top = r.explanation[0] || "";
  return `<div class="card compare-card">
    <p class="mol-name" style="font-size:1.3rem">${esc(capitalize(r.name))}</p>
    <div class="mol-meta">${formula(r.formula)} · ${r.mol_weight.toFixed(0)} g/mol</div>
    <div class="mol-paper" style="margin:12px 0">${r.svg_explained}</div>
    <div class="big-number">${num(r.logS)}<small>logS</small></div>
    <div style="margin:10px 0">${badge(r.category)}</div>
    <div class="small muted">≈ <span class="ltr">${fmtConc(r.g_per_L)}</span> · ${T("range", "بازه")} <span class="ltr">${num(r.interval[0], 1)} … ${num(r.interval[1], 1)}</span></div>
    ${r.measured ? `<div class="small muted">${T("measured in the lab", "اندازه‌گیری آزمایشگاهی")}: <b class="ltr">${num(r.measured.logS)}</b></div>` : ""}
    ${scaleBar(r)}
    <p class="small" style="margin-top:14px">${esc(top)}</p>
    ${r.domain.in_domain ? "" : `<p class="small" style="color:var(--warn-ink)">⚠ ${T("Outside the model's experience", "بیرون از تجربهٔ مدل")}</p>`}
  </div>`;
}

function drawTrend() {
  const all = (last || []).filter((r) => !r.error);
  if (!all.length || !window.Plotly) return;
  const ok = all.filter((r) => r.kind !== "ionic"), salts = all.filter((r) => r.kind === "ionic" && r.measured);
  const colors = ok.map((r) => css(`--${CLASS_KEY[r.category]}`));
  const traces = [{
    type: "bar", name: "SoluAI", x: ok.map((r) => capitalize(r.name)), y: ok.map((r) => r.logS), marker: { color: colors },
    error_y: { type: "data", symmetric: false, array: ok.map((r) => r.interval[1] - r.logS),
      arrayminus: ok.map((r) => r.logS - r.interval[0]), color: css("--ink-3"), thickness: 1.5, width: 8 },
    hovertemplate: "%{x}<br>logS %{y:.2f}<extra>SoluAI</extra>",
  }];
  const meas = ok.filter((r) => r.measured);
  if (meas.length || salts.length) traces.push({
    type: "scatter", mode: "markers", name: T("measured", "اندازه‌گیری‌شده"),
    x: [...meas.map((r) => capitalize(r.name)), ...salts.map((r) => capitalize(r.name))],
    y: [...meas.map((r) => r.measured.logS), ...salts.map((r) => r.measured.logS)],
    marker: { size: 13, symbol: "diamond", color: css("--ink"), line: { color: css("--bg-elev"), width: 2 } },
    hovertemplate: "%{x}<br>logS %{y:.2f}<extra></extra>",
  });
  Plotly.newPlot("trend", traces, plotLayout({ yaxis: { title: T("logS (log mol/L)", "logS (لگاریتم مول بر لیتر)"), gridcolor: css("--line"), zerolinecolor: css("--ink-3") },
    showlegend: traces.length > 1, legend: { orientation: "h", y: 1.12 } }), PLOT_CONFIG);
}
document.addEventListener("themechange", drawTrend);
