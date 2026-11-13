/* "How the model works" page: metrics, parity plots, importance, uncertainty. */

let REP = null;
let ds = "aqsoldb", sp = "random", parityModel = null;

document.addEventListener("DOMContentLoaded", async () => {
  try { REP = await api("/api/report"); } catch (err) { document.getElementById("kpis").innerHTML = errorBox(err); return; }
  // Round metrics half-up once (0.835 -> 0.84) so they match the README; toFixed alone would show 0.83.
  const r2 = (v) => Math.round(v * 100 + 1e-6) / 100;
  REP.metrics.results.forEach((r) => ["cv_rmse", "cv_rmse_sd", "test_rmse", "test_mae", "test_r2"].forEach((k) => { r[k] = r2(r[k]); }));
  kpis();
  bindSeg("ds", (v) => { ds = v; parityModel = null; drawSplit(); });
  bindSeg("sp", (v) => { sp = v; parityModel = null; drawSplit(); });
  document.getElementById("parity-model").onchange = (e) => { parityModel = e.target.value; drawParity(); };
  drawAll();
});
document.addEventListener("themechange", () => REP && drawAll());

function bindSeg(id, fn) {
  document.getElementById(id).onclick = (e) => {
    const v = e.target.dataset.v; if (!v) return;
    document.querySelectorAll(`#${id} button`).forEach((b) => b.classList.toggle("active", b === e.target));
    fn(v);
  };
}
const rows = (d, s) => REP.metrics.results.filter((r) => r.dataset === d && r.split === s);
const bestBy = (list, key) => list.reduce((a, b) => (b[key] < a[key] ? b : a));
const lgbmBest = (d, s) => bestBy(rows(d, s).filter((r) => r.model === "LightGBM"), "cv_rmse");

function kpis() {
  const r = lgbmBest("aqsoldb", "random"), s = lgbmBest("aqsoldb", "scaffold");
  const del = rows("aqsoldb", "random").find((x) => x.model === "Delaney (original)");
  const unc = REP.splits.aqsoldb.scaffold.uncertainty;
  const box = (k, v, sub) => `<div class="card"><div class="kicker">${k}</div><div class="big-number" style="font-size:2.3rem;margin-top:6px">${v}</div><div class="small muted">${sub}</div></div>`;
  document.getElementById("kpis").innerHTML =
    box("Typical error · random", r.test_rmse.toFixed(2), `RMSE in log units (R² ${r.test_r2.toFixed(2)})`) +
    box("Typical error · new skeletons", s.test_rmse.toFixed(2), `scaffold split (R² ${s.test_r2.toFixed(2)})`) +
    box("Delaney's formula", del.test_rmse.toFixed(2), "same test molecules, random split") +
    box("90% range really covers", `${Math.round(unc.coverage_all * 100)}%`, "of unseen scaffold-test molecules");
}

function drawAll() { drawSplit(); drawHuman(); drawImportance(); drawDependence(); drawUncertainty(); }

function drawSplit() {
  const list = rows(ds, sp);
  const best = bestBy(list, "cv_rmse");
  document.getElementById("table").innerHTML = `<table><thead><tr><th>Model</th><th>Features</th><th class="num">CV RMSE</th><th class="num">Test RMSE</th><th class="num">MAE</th><th class="num">R²</th></tr></thead><tbody>${
    list.map((r) => `<tr class="${r === best ? "best" : ""}"><td>${esc(r.model)}</td><td>${esc(r.features)}</td><td class="num">${r.cv_rmse.toFixed(2)} <span class="muted small">±${r.cv_rmse_sd.toFixed(2)}</span></td><td class="num">${r.test_rmse.toFixed(2)}</td><td class="num">${r.test_mae.toFixed(2)}</td><td class="num">${r.test_r2.toFixed(2)}</td></tr>`).join("")
  }</tbody></table><p class="small muted" style="margin-top:8px">${list[0].n_train} training molecules · ${list[0].n_test} test molecules</p>`;
  const preds = REP.splits[ds][sp].predictions;
  const sel = document.getElementById("parity-model");
  const names = Object.keys(preds);
  if (!parityModel) parityModel = names.find((n) => n.startsWith("LightGBM")) || names[0];
  sel.innerHTML = names.map((n) => `<option ${n === parityModel ? "selected" : ""}>${esc(n)}</option>`).join("");
  drawParity();
}

function drawParity() {
  const s = REP.splits[ds][sp], y = s.y_test, p = s.predictions[parityModel];
  const lo = Math.min(...y, ...p) - 0.5, hi = Math.max(...y, ...p) + 0.5;
  const rmse = Math.sqrt(y.reduce((a, v, i) => a + (v - p[i]) ** 2, 0) / y.length);
  Plotly.newPlot("p-parity", [
    { x: [lo, hi, hi, lo], y: [lo - 1, hi - 1, hi + 1, lo + 1], fill: "toself", fillcolor: css("--accent-soft"), line: { width: 0 }, hoverinfo: "skip", name: "±1 log unit" },
    { x: [lo, hi], y: [lo, hi], mode: "lines", line: { color: css("--ink-3"), dash: "dash", width: 1.5 }, hoverinfo: "skip", name: "perfect" },
    { type: "scattergl", mode: "markers", x: y, y: p, marker: { size: 6, color: css("--accent"), opacity: 0.55 },
      hovertemplate: "measured %{x:.2f}<br>predicted %{y:.2f}<extra></extra>", name: "test molecules" },
  ], plotLayout({ showlegend: false,
    xaxis: { title: "Measured logS", range: [lo, hi], gridcolor: css("--line"), zeroline: false },
    yaxis: { title: "Predicted logS", range: [lo, hi], gridcolor: css("--line"), zeroline: false, scaleanchor: "x" },
    annotations: [{ x: lo + 0.3, y: hi - 0.4, xanchor: "left", showarrow: false, text: `RMSE ${rmse.toFixed(2)}`, font: { size: 15, color: css("--ink") } }] }), PLOT_CONFIG);
}

function drawHuman() {
  const order = ["Delaney (original)", "Delaney (refit)", "Linear regression", "Random forest", "LightGBM"];
  const pick = (d, s, m) => { const l = rows(d, s).filter((r) => r.model === m); return l.length ? bestBy(l, "cv_rmse").test_rmse : null; };
  const labels = order.map((o) => o.replace(" (", "<br>(").replace("Linear regression", "Linear<br>regression").replace("Random forest", "Random<br>forest"));
  const trace = (s, color) => ({ type: "bar", name: s === "random" ? "Random split" : "Scaffold split", x: labels,
    y: order.map((m) => pick("aqsoldb", s, m)), marker: { color }, text: order.map((m) => (pick("aqsoldb", s, m) || 0).toFixed(2)), textposition: "outside", cliponaxis: false });
  const gain = (s) => Math.round(100 * (1 - pick("aqsoldb", s, "LightGBM") / pick("aqsoldb", s, "Delaney (original)")));
  document.getElementById("gain").textContent = `${gain("scaffold")}–${gain("random")}%`;
  Plotly.newPlot("p-human", [trace("random", css("--accent")), trace("scaffold", css("--ink-3"))],
    plotLayout({ barmode: "group", yaxis: { title: "Test RMSE on AqSolDB (lower = better)", gridcolor: css("--line"), rangemode: "tozero" },
      xaxis: { tickangle: 0 }, legend: { orientation: "h", y: 1.12 }, margin: { l: 60, r: 10, t: 30, b: 80 } }), PLOT_CONFIG);
}

function drawImportance() {
  const imp = [...REP.importance].reverse();
  Plotly.newPlot("p-importance", [{ type: "bar", orientation: "h", x: imp.map((d) => d.mean_abs_shap), y: imp.map((d) => d.label),
    marker: { color: imp.map((d) => (d.feature.startsWith("FP_") ? css("--ink-3") : css("--accent"))) },
    hovertemplate: "%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>" }],
    plotLayout({ margin: { l: 210, r: 20, t: 10, b: 50 }, xaxis: { title: "mean |SHAP| (log units)", gridcolor: css("--line") }, yaxis: { automargin: true } }), PLOT_CONFIG);
}

function drawDependence() {
  const labels = { MolLogP: "logP", TPSA: "Polar surface (Å²)", MolWt: "Molecular weight", NumHDonors: "H-bond donors", NumHAcceptors: "H-bond acceptors", AromaticProportion: "Aromatic proportion" };
  const keys = ["MolLogP", "TPSA", "MolWt", "NumHDonors"].filter((k) => REP.dependence[k]);
  const el = document.getElementById("dep");
  el.innerHTML = keys.map((k) => `<div><div id="dep-${k}" class="plot" style="min-height:260px"></div><p class="small muted" style="text-align:center">trend: ${REP.shap_trend_correlation[k] < 0 ? "↓ lowers solubility as it grows" : "↑ raises solubility as it grows"}</p></div>`).join("");
  keys.forEach((k) => {
    const d = REP.dependence[k];
    const sorted = [...d.x].sort((a, b) => a - b);
    const q = (p) => sorted[Math.floor(p * (sorted.length - 1))];
    const pad = (q(0.99) - q(0.01)) * 0.05 || 0.5;
    Plotly.newPlot(`dep-${k}`, [{ type: "scattergl", mode: "markers", x: d.x, y: d.shap,
      marker: { size: 4, opacity: 0.5, color: d.shap.map((v) => (v >= 0 ? css("--up") : css("--down"))) }, hoverinfo: "skip" }],
      plotLayout({ margin: { l: 50, r: 10, t: 10, b: 45 }, showlegend: false,
        xaxis: { title: labels[k], gridcolor: css("--line"), range: [q(0.01) - pad, q(0.99) + pad] }, yaxis: { title: "push on logS", gridcolor: css("--line"), zerolinecolor: css("--ink-3") } }), PLOT_CONFIG);
  });
}

function drawUncertainty() {
  const u = REP.splits.aqsoldb.scaffold.uncertainty, ur = REP.splits.aqsoldb.random.uncertainty;
  const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  document.getElementById("unc").innerHTML = `
    <p class="muted">Every prediction comes with a <b>90% range</b>. We looked at the model's mistakes on molecules it had not seen, and made the range wide enough to cover 90% of them. Unfamiliar molecules get a wider range.</p>
    <table><thead><tr><th></th><th class="num">Random</th><th class="num">Scaffold</th></tr></thead><tbody>
      <tr><td>Range (familiar molecule)</td><td class="num">± ${ur.half_width_in.toFixed(2)}</td><td class="num">± ${u.half_width_in.toFixed(2)}</td></tr>
      <tr><td>Range (unfamiliar molecule)</td><td class="num">± ${ur.half_width_out.toFixed(2)}</td><td class="num">± ${u.half_width_out.toFixed(2)}</td></tr>
      <tr><td>Test molecules inside their range</td><td class="num">${pct(ur.coverage_all)}</td><td class="num">${pct(u.coverage_all)}</td></tr>
      <tr><td>Share flagged "unfamiliar"</td><td class="num">${pct(ur.share_out_of_domain)}</td><td class="num">${pct(u.share_out_of_domain)}</td></tr>
      <tr><td>Error (RMSE) familiar / unfamiliar</td><td class="num">${ur.rmse_in?.toFixed(2)} / ${ur.rmse_out?.toFixed(2) ?? "–"}</td><td class="num">${u.rmse_in?.toFixed(2)} / ${u.rmse_out?.toFixed(2) ?? "–"}</td></tr>
    </tbody></table>`;
  const b = u.error_by_similarity;
  Plotly.newPlot("p-sim", [{ type: "bar", x: b.map((d) => d.range), y: b.map((d) => d.rmse), text: b.map((d) => `n=${d.n}`), textposition: "outside", cliponaxis: false,
    marker: { color: b.map((d) => (parseFloat(d.range) < 0.3 ? css("--cls-slight") : css("--accent"))) },
    hovertemplate: "similarity %{x}<br>RMSE %{y:.2f}<br>%{text}<extra></extra>" }],
    plotLayout({ xaxis: { title: "Similarity to closest training molecule" }, yaxis: { title: "Test RMSE", gridcolor: css("--line"), rangemode: "tozero" } }), PLOT_CONFIG);
}
