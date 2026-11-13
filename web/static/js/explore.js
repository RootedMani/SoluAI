/* Explore page: interactive charts of the training data. */

let DATA = [];
let colorBy = "HBD";
const COLOR_LABEL = { HBD: "H-bond donors", aromatic: "Aromatic proportion", TPSA: "Polar surface (Å²)" };

document.addEventListener("DOMContentLoaded", async () => {
  ["p-logp", "p-mw", "p-hist"].forEach((id) => { document.getElementById(id).innerHTML = loadingBox("Loading data…"); });
  try {
    DATA = await api("/api/dataset");
  } catch (err) {
    document.getElementById("stats").innerHTML = errorBox(err);
    return;
  }
  ["p-logp", "p-mw", "p-hist"].forEach((id) => { document.getElementById(id).innerHTML = ""; });
  stats();
  drawAll();
  document.getElementById("color-by").onclick = (e) => {
    if (!e.target.dataset.c) return;
    colorBy = e.target.dataset.c;
    document.querySelectorAll("#color-by button").forEach((b) => b.classList.toggle("active", b === e.target));
    drawScatter();
  };
  let timer;
  document.getElementById("filter").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(drawScatter, 250); });
});
document.addEventListener("themechange", () => DATA.length && drawAll());

function corr(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxy / Math.sqrt(sxx * syy);
}

function stats() {
  const y = DATA.map((d) => d.logS);
  const count = (lo, hi) => y.filter((v) => v >= lo && v < hi).length;
  const box = (label, value, sub) => `<div class="card"><div class="kicker">${label}</div><div class="big-number" style="font-size:2.2rem;margin-top:6px">${value}</div><div class="small muted">${sub}</div></div>`;
  document.getElementById("stats").innerHTML =
    box("Molecules", DATA.length.toLocaleString(), "after cleaning AqSolDB") +
    box("logS range", `${Math.min(...y).toFixed(0)} to ${Math.max(...y).toFixed(0)}`, "a factor of 10¹⁵ in solubility!") +
    box("Insoluble", `${Math.round(100 * count(-99, -4) / y.length)}%`, "logS below −4") +
    box("Highly soluble", `${Math.round(100 * count(0, 99) / y.length)}%`, "logS above 0");
  document.getElementById("r-logp").textContent = corr(DATA.map((d) => d.logP), y).toFixed(2);
  document.getElementById("r-mw").textContent = corr(DATA.map((d) => d.MW), y).toFixed(2);
}

function drawAll() { drawScatter(); drawHist(); }

function scatter(id, xKey, xTitle) {
  const f = document.getElementById("filter").value.trim().toLowerCase();
  const match = (d) => !f || d.name.toLowerCase().includes(f);
  const base = DATA.filter(match);
  const cmax = { HBD: 6, aromatic: 1, TPSA: 160 }[colorBy];
  const trace = {
    type: "scattergl", mode: "markers", x: base.map((d) => d[xKey]), y: base.map((d) => d.logS),
    text: base.map((d) => d.name), customdata: base.map((d) => d.smiles),
    marker: { size: f ? 8 : 5, opacity: f ? 0.95 : 0.6, color: base.map((d) => d[colorBy]), cmin: 0, cmax,
      colorscale: [[0, "#f59e0b"], [0.5, "#10b981"], [1, "#0891b2"]],
      colorbar: { title: { text: COLOR_LABEL[colorBy], side: "right" }, thickness: 12, outlinewidth: 0 } },
    hovertemplate: `<b>%{text}</b><br>${xTitle}: %{x}<br>logS: %{y}<extra></extra>`,
  };
  const traces = [trace];
  if (f) traces.unshift({ type: "scattergl", mode: "markers", x: DATA.map((d) => d[xKey]), y: DATA.map((d) => d.logS),
    marker: { size: 4, color: css("--line") }, hoverinfo: "skip" });
  Plotly.newPlot(id, traces, plotLayout({ showlegend: false,
    xaxis: { title: xTitle, gridcolor: css("--line"), zerolinecolor: css("--line"), range: xKey === "logP" ? [-8, 14] : [0, 1300] },
    yaxis: { title: "Measured logS (log mol/L)", gridcolor: css("--line"), zerolinecolor: css("--line") } }), PLOT_CONFIG);
  const el = document.getElementById(id);
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on("plotly_click", (ev) => { const p = ev.points[0]; if (p && p.customdata) location.href = `/?q=${encodeURIComponent(p.customdata)}`; });
}

function drawScatter() {
  scatter("p-logp", "logP", "logP (oiliness)");
  scatter("p-mw", "MW", "Molecular weight (g/mol)");
}

function drawHist() {
  const bands = [[-14, -4, "--cls-insol"], [-4, -2, "--cls-slight"], [-2, 0, "--cls-sol"], [0, 3, "--cls-high"]];
  const shapes = bands.map(([a, b, c]) => ({ type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 1,
    fillcolor: css(c), opacity: 0.08, line: { width: 0 } }));
  const annotations = [["Insoluble", -7], ["Slightly", -3], ["Soluble", -1], ["Highly", 1.2]].map(([t, x]) =>
    ({ x, y: 1, yref: "paper", text: t, showarrow: false, font: { size: 12, color: css("--ink-3") }, yanchor: "bottom" }));
  Plotly.newPlot("p-hist", [{ type: "histogram", x: DATA.map((d) => d.logS), xbins: { size: 0.25 },
    marker: { color: css("--accent"), line: { width: 0 } }, hovertemplate: "logS %{x}<br>%{y} molecules<extra></extra>" }],
    plotLayout({ shapes, annotations, bargap: 0.04,
      xaxis: { title: "Measured logS (log mol/L)", gridcolor: css("--line"), range: [-13, 3] },
      yaxis: { title: "Number of molecules", gridcolor: css("--line") } }), PLOT_CONFIG);
}
