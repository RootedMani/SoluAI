/* Trends page: how solubility changes along a homologous series (one more CH2 each step). */

let SERIES = [];
let active = new Set(["alcohols", "alkanes"]);
let showClassic = false;
const COLORS = ["#0891b2", "#dc2626", "#7c3aed", "#059669", "#d97706", "#db2777", "#475569"];

document.addEventListener("DOMContentLoaded", async () => {
  const out = document.getElementById("plot");
  out.innerHTML = loadingBox();
  try { SERIES = (await api("/api/series")).series; } catch (err) { out.innerHTML = errorBox(err); return; }
  out.innerHTML = "";
  const chips = document.getElementById("series-chips");
  chips.innerHTML = SERIES.map((s, i) => `<button class="chip" data-k="${s.key}" style="--c:${COLORS[i]}">${esc(s.title)} <span class="ltr">${esc(s.group)}</span></button>`).join("");
  chips.onclick = (e) => {
    const b = e.target.closest("[data-k]"); if (!b) return;
    active.has(b.dataset.k) ? active.delete(b.dataset.k) : active.add(b.dataset.k);
    draw();
  };
  document.getElementById("classic").onchange = (e) => { showClassic = e.target.checked; draw(); };
  draw();
});
document.addEventListener("themechange", () => SERIES.length && draw());

function draw() {
  document.querySelectorAll("#series-chips .chip").forEach((c) => c.classList.toggle("active", active.has(c.dataset.k)));
  const traces = [];
  SERIES.forEach((s, i) => {
    if (!active.has(s.key)) return;
    const c = COLORS[i], pts = s.points;
    traces.push({ type: "scatter", mode: "lines+markers", name: `${s.title} · SoluAI`, x: pts.map((p) => p.n), y: pts.map((p) => p.predicted),
      text: pts.map((p) => p.name), customdata: pts.map((p) => p.smiles), line: { color: c, width: 3 }, marker: { size: 7, color: c },
      hovertemplate: "<b>%{text}</b><br>SoluAI: %{y:.2f}<extra></extra>" });
    const m = pts.filter((p) => p.measured !== null);
    traces.push({ type: "scatter", mode: "markers", name: `${s.title} · ${T("measured", "اندازه‌گیری‌شده")}`, x: m.map((p) => p.n), y: m.map((p) => p.measured),
      text: m.map((p) => p.name), customdata: m.map((p) => p.smiles),
      marker: { size: 13, symbol: "diamond-open", color: c, line: { width: 2.5, color: c } },
      hovertemplate: `<b>%{text}</b><br>${T("measured", "اندازه‌گیری")}: %{y:.2f}<extra></extra>` });
    if (showClassic) traces.push({ type: "scatter", mode: "lines", name: `${s.title} · ${T("classic formula", "فرمول کلاسیک")}`, x: pts.map((p) => p.n), y: pts.map((p) => p.classic),
      line: { color: c, width: 1.5, dash: "dot" }, hoverinfo: "skip" });
  });
  if (!traces.length) { Plotly.purge("plot"); document.getElementById("plot").innerHTML = `<p class="muted">${T("Pick at least one family above.", "دست‌کم یک خانواده را از بالا انتخاب کنید.")}</p>`; document.getElementById("cards").innerHTML = ""; return; }
  if (!document.getElementById("plot").classList.contains("js-plotly-plot")) document.getElementById("plot").innerHTML = "";
  Plotly.newPlot("plot", traces, plotLayout({
    xaxis: { title: T("Number of carbon atoms", "تعداد اتم‌های کربن"), dtick: 1, gridcolor: css("--line") },
    yaxis: { title: T("logS (log mol/L)", "logS (لگاریتم مول بر لیتر)"), gridcolor: css("--line"), zerolinecolor: css("--ink-3") },
    legend: { orientation: "h", y: -0.22 }, margin: { l: 60, r: 20, t: 10, b: 60 } }), PLOT_CONFIG);
  const el = document.getElementById("plot");
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on("plotly_click", (ev) => { const p = ev.points[0]; if (p && p.customdata) location.href = `/?q=${encodeURIComponent(p.customdata)}`; });

  document.getElementById("cards").innerHTML = SERIES.filter((s) => active.has(s.key)).map((s) => {
    const i = SERIES.indexOf(s);
    const fold = (sl) => (sl === null ? "–" : `${Number(Math.pow(10, -sl).toPrecision(2))}×`);
    return `<div class="card"><div class="kicker" style="color:${COLORS[i]}">${esc(s.title)} <span class="ltr">${esc(s.group)}</span></div>
      <div class="big-number" style="font-size:2.2rem;margin-top:6px">${fold(s.slope_predicted)}</div>
      <div class="small muted">${T("less soluble for every extra CH₂ (SoluAI)", "کاهش انحلال‌پذیری با هر ‎CH₂‎ اضافه (SoluAI)")}</div>
      <div class="small" style="margin-top:8px">${T("Lab measurements say", "اندازه‌گیری‌ها می‌گویند")}: <b class="ltr">${fold(s.slope_measured)}</b></div></div>`;
  }).join("");
}
