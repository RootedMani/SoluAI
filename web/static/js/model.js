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
  household();
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
const V2 = "SoluAI v2";
// The deployed model: SoluAI v2 when the results contain it, otherwise the best LightGBM (model v1).
const deployed = (d, s) => rows(d, s).find((r) => r.model === V2) || lgbmBest(d, s);
const pct1 = (v) => (v == null ? "–" : `${(100 * v).toFixed(0)}%`);

function kpis() {
  const r = deployed("aqsoldb", "random"), s = deployed("aqsoldb", "scaffold");
  const del = rows("aqsoldb", "random").find((x) => x.model === "Delaney (original)");
  const unc = REP.splits.aqsoldb.scaffold.uncertainty;
  const tr = REP.splits.aqsoldb.random.trust;
  const box = (k, v, sub) => `<div class="card"><div class="kicker">${k}</div><div class="big-number" style="font-size:2.3rem;margin-top:6px">${v}</div><div class="small muted">${sub}</div></div>`;
  document.getElementById("kpis").innerHTML =
    box(T("Typical error · random", "خطای معمول · تصادفی"), r.test_rmse.toFixed(2), T(`RMSE in log units (R² ${r.test_r2.toFixed(2)})`, `RMSE به واحد لگاریتمی (‎R² ${r.test_r2.toFixed(2)}‎)`)) +
    box(T("Typical error · new skeletons", "خطای معمول · اسکلت‌های تازه"), s.test_rmse.toFixed(2), T(`scaffold split (R² ${s.test_r2.toFixed(2)})`, `تقسیم اسکلتی (‎R² ${s.test_r2.toFixed(2)}‎)`)) +
    box(T("Delaney's formula", "فرمول دلانی"), del.test_rmse.toFixed(2), T("same test molecules, random split", "همان مولکول‌های آزمون، تقسیم تصادفی")) +
    (tr && tr.tiers.high
      ? box(T("When it says “high confidence”", "وقتی می‌گوید «اطمینان بالا»"), pct1(tr.tiers.high.within_1), T(`are within a factor of 10 (${pct1(tr.tiers.high.share)} of test molecules, random split)`, `در محدودهٔ 10 برابر هستند (${pct1(tr.tiers.high.share)} از مولکول‌های آزمون، تقسیم تصادفی)`))
      : box(T("90% range really covers", "پوشش واقعی بازهٔ 90%"), `${Math.round(unc.coverage_all * 100)}%`, T("of unseen scaffold-test molecules", "از مولکول‌های دیده‌نشدهٔ آزمون اسکلتی")));
}

function drawAll() { drawSplit(); drawHuman(); drawImportance(); drawDependence(); drawUncertainty(); }


function drawSplit() {
  const list = rows(ds, sp);
  const best = bestBy(list, "cv_rmse");
  document.getElementById("table").innerHTML = `<table><thead><tr><th>${T("Model", "مدل")}</th><th>${T("Features", "ویژگی‌ها")}</th><th class="num">CV RMSE</th><th class="num">${T("Test RMSE", "RMSE آزمون")}</th><th class="num">MAE</th><th class="num">R²</th></tr></thead><tbody>${
    list.map((r) => `<tr class="${r === best ? "best" : ""}"><td>${esc(r.model)}</td><td>${esc(r.features)}</td><td class="num">${r.cv_rmse.toFixed(2)} <span class="muted small">±${r.cv_rmse_sd.toFixed(2)}</span></td><td class="num">${r.test_rmse.toFixed(2)}</td><td class="num">${r.test_mae.toFixed(2)}</td><td class="num">${r.test_r2.toFixed(2)}</td></tr>`).join("")
  }</tbody></table><p class="small muted" style="margin-top:8px">${T(`${list[0].n_train} training molecules · ${list[0].n_test} test molecules`, `${list[0].n_train} مولکول آموزشی · ${list[0].n_test} مولکول آزمون`)}</p>`;
  const preds = REP.splits[ds][sp].predictions;
  const sel = document.getElementById("parity-model");
  const names = Object.keys(preds);
  if (!parityModel) parityModel = names.find((n) => n.startsWith(V2)) || names.find((n) => n.startsWith("LightGBM")) || names[0];
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
    xaxis: { title: T("Measured logS", "logS اندازه‌گیری‌شده"), range: [lo, hi], gridcolor: css("--line"), zeroline: false },
    yaxis: { title: T("Predicted logS", "logS پیش‌بینی‌شده"), range: [lo, hi], gridcolor: css("--line"), zeroline: false, scaleanchor: "x" },
    annotations: [{ x: lo + 0.3, y: hi - 0.4, xanchor: "left", showarrow: false, text: `RMSE ${rmse.toFixed(2)}`, font: { size: 15, color: css("--ink") } }] }), PLOT_CONFIG);
}

function drawHuman() {
  const hasV2 = rows("aqsoldb", "random").some((r) => r.model === V2);
  const order = ["Delaney (original)", "Delaney (refit)", "Linear regression", "Random forest", "LightGBM"].concat(hasV2 ? [V2] : []);
  const top = hasV2 ? V2 : "LightGBM";
  const pick = (d, s, m) => { const l = rows(d, s).filter((r) => r.model === m); return l.length ? bestBy(l, "cv_rmse").test_rmse : null; };
  const labels = order.map((o) => o.replace(" (", "<br>(").replace("Linear regression", "Linear<br>regression").replace("Random forest", "Random<br>forest"));
  const trace = (s, color) => ({ type: "bar", name: s === "random" ? T("Random split", "تقسیم تصادفی") : T("Scaffold split", "تقسیم اسکلتی"), x: labels,
    y: order.map((m) => pick("aqsoldb", s, m)), marker: { color }, text: order.map((m) => (pick("aqsoldb", s, m) || 0).toFixed(2)), textposition: "outside", cliponaxis: false });
  const gain = (s) => Math.round(100 * (1 - pick("aqsoldb", s, top) / pick("aqsoldb", s, "Delaney (original)")));
  document.getElementById("gain").textContent = `${gain("scaffold")}–${gain("random")}%`;
  Plotly.newPlot("p-human", [trace("random", css("--accent")), trace("scaffold", css("--ink-3"))],
    plotLayout({ barmode: "group", yaxis: { title: T("Test RMSE on AqSolDB (lower = better)", "RMSE آزمون روی AqSolDB (کمتر = بهتر)"), gridcolor: css("--line"), rangemode: "tozero" },
      xaxis: { tickangle: 0 }, legend: { orientation: "h", y: 1.12 }, margin: { l: 60, r: 10, t: 30, b: 80 } }), PLOT_CONFIG);
}

function drawImportance() {
  const imp = [...REP.importance].reverse();
  Plotly.newPlot("p-importance", [{ type: "bar", orientation: "h", x: imp.map((d) => d.mean_abs_shap), y: imp.map((d) => d.label),
    marker: { color: imp.map((d) => (d.feature.startsWith("FP_") ? css("--ink-3") : css("--accent"))) },
    hovertemplate: "%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>" }],
    plotLayout({ margin: { l: 210, r: 20, t: 10, b: 50 }, xaxis: { title: T("mean |SHAP| (log units)", "میانگین |SHAP| (واحد لگاریتمی)"), gridcolor: css("--line") }, yaxis: { automargin: true } }), PLOT_CONFIG);
}

function drawDependence() {
  const labels = { MolLogP: "logP", TPSA: T("Polar surface (Å²)", "سطح قطبی (Å²)"), MolWt: T("Molecular weight", "جرم مولی"), NumHDonors: T("H-bond donors", "دهنده‌های پیوند H"), NumHAcceptors: T("H-bond acceptors", "پذیرنده‌های پیوند H"), AromaticProportion: T("Aromatic proportion", "سهم آروماتیک") };
  const keys = ["MolLogP", "TPSA", "MolWt", "NumHDonors"].filter((k) => REP.dependence[k]);
  const el = document.getElementById("dep");
  el.innerHTML = keys.map((k) => `<div><div id="dep-${k}" class="plot" style="min-height:260px"></div><p class="small muted" style="text-align:center">${REP.shap_trend_correlation[k] < 0 ? T("trend: ↓ lowers solubility as it grows", "روند: ↓ با بزرگ‌تر شدن، انحلال‌پذیری کم می‌شود") : T("trend: ↑ raises solubility as it grows", "روند: ↑ با بزرگ‌تر شدن، انحلال‌پذیری زیاد می‌شود")}</p></div>`).join("");
  keys.forEach((k) => {
    const d = REP.dependence[k];
    const sorted = [...d.x].sort((a, b) => a - b);
    const q = (p) => sorted[Math.floor(p * (sorted.length - 1))];
    const pad = (q(0.99) - q(0.01)) * 0.05 || 0.5;
    Plotly.newPlot(`dep-${k}`, [{ type: "scattergl", mode: "markers", x: d.x, y: d.shap,
      marker: { size: 4, opacity: 0.5, color: d.shap.map((v) => (v >= 0 ? css("--up") : css("--down"))) }, hoverinfo: "skip" }],
      plotLayout({ margin: { l: 50, r: 10, t: 10, b: 45 }, showlegend: false,
        xaxis: { title: labels[k], gridcolor: css("--line"), range: [q(0.01) - pad, q(0.99) + pad] }, yaxis: { title: T("push on logS", "اثر بر logS"), gridcolor: css("--line"), zerolinecolor: css("--ink-3") } }), PLOT_CONFIG);
  });
}

function drawUncertainty() {
  const u = REP.splits.aqsoldb.scaffold.uncertainty, ur = REP.splits.aqsoldb.random.uncertainty;
  const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  const t = REP.splits.aqsoldb.random.trust, ts = REP.splits.aqsoldb.scaffold.trust;
  if (t && ts) {
    // Model v2: the referee sizes the range for each molecule and gives a confidence label.
    const cell = (x) => (x ? `<td class="num">${pct1(x.share)}</td><td class="num"><b>${pct1(x.within_1)}</b></td>` : `<td class="num" colspan="2">–</td>`);
    const line = (label, a, b) => `<tr><td>${label}</td>${cell(a)}${cell(b)}</tr>`;
    document.getElementById("unc").innerHTML = `
      <p class="muted">${T("A second small model, the <b>referee</b>, estimates how large the error usually is for a molecule like this one (from its closest training molecules and from the prediction itself). The <b>90% range</b> is scaled with that estimate, so easy molecules get a narrow range and hard ones a wide range.",
        "یک مدل کوچک دوم، یعنی <b>داور</b>، برآورد می‌کند خطا برای مولکولی شبیه این معمولاً چقدر است (با نگاه به نزدیک‌ترین مولکول‌های آموزشی و خودِ پیش‌بینی). <b>بازهٔ 90%</b> با همین برآورد تنظیم می‌شود؛ پس مولکول‌های آسان بازهٔ باریک و مولکول‌های سخت بازهٔ پهن می‌گیرند.")}</p>
      <div class="table-wrap"><table><thead>
        <tr><th></th><th class="num" colspan="2">${T("Random split", "تقسیم تصادفی")}</th><th class="num" colspan="2">${T("Scaffold split", "تقسیم اسکلتی")}</th></tr>
        <tr><th>${T("Test molecules", "مولکول‌های آزمون")}</th>${`<th class="num">${T("share", "سهم")}</th><th class="num">${T("within ×10", "در محدودهٔ ×10")}</th>`.repeat(2)}</tr></thead><tbody>
        ${line(T("High confidence", "اطمینان بالا"), t.tiers.high, ts.tiers.high)}
        ${line(T("Medium confidence", "اطمینان متوسط"), t.tiers.medium, ts.tiers.medium)}
        ${line(T("Low confidence", "اطمینان پایین"), t.tiers.low, ts.tiers.low)}
        ${line(T("All", "همه"), t.all, ts.all)}
        ${line(T("Measured 2+ times, labs agree", "دو بار یا بیشتر اندازه‌گیری‌شده و هم‌خوان"), t.reliable_measurements, ts.reliable_measurements)}
      </tbody></table></div>
      <p class="small muted" style="margin-top:10px">${T(
        `"Within ×10" = predicted within one log unit of the lab value. High confidence = a 90% range of ± 0.8 or narrower, low = wider than ± 1.4. The 90% ranges really contained ${pct(t.all.coverage)} (random) and ${pct(ts.all.coverage)} (scaffold) of the test molecules. One fixed width for every molecule would need ± ${t.fixed_width.half_width.toFixed(2)} and ± ${ts.fixed_width.half_width.toFixed(2)}.`,
        `«در محدودهٔ ×10» یعنی پیش‌بینی حداکثر یک واحد لگاریتمی با مقدار آزمایشگاهی فاصله دارد. اطمینان بالا = بازهٔ 90% برابر ± 0.8 یا باریک‌تر، اطمینان پایین = پهن‌تر از ± 1.4. بازه‌های 90% در عمل ${pct(t.all.coverage)} (تصادفی) و ${pct(ts.all.coverage)} (اسکلتی) از مولکول‌های آزمون را در بر گرفتند. اگر برای همهٔ مولکول‌ها یک پهنای ثابت می‌گذاشتیم، ± ${t.fixed_width.half_width.toFixed(2)} و ± ${ts.fixed_width.half_width.toFixed(2)} لازم بود.`)}</p>`;
  } else {
  document.getElementById("unc").innerHTML = `
    <p class="muted">${T("Every prediction comes with a <b>90% range</b>. We looked at the model's mistakes on molecules it had not seen, and made the range wide enough to cover 90% of them. Unfamiliar molecules get a wider range.",
      "هر پیش‌بینی یک <b>بازهٔ 90%</b> دارد. خطاهای مدل را روی مولکول‌هایی که ندیده بود بررسی کردیم و بازه را آن‌قدر پهن گرفتیم که 90% آن‌ها را بپوشاند. مولکول‌های ناآشنا بازهٔ پهن‌تری می‌گیرند.")}</p>
    <table><thead><tr><th></th><th class="num">${T("Random", "تصادفی")}</th><th class="num">${T("Scaffold", "اسکلتی")}</th></tr></thead><tbody>
      <tr><td>${T("Range (familiar molecule)", "بازه (مولکول آشنا)")}</td><td class="num">± ${ur.half_width_in.toFixed(2)}</td><td class="num">± ${u.half_width_in.toFixed(2)}</td></tr>
      <tr><td>${T("Range (unfamiliar molecule)", "بازه (مولکول ناآشنا)")}</td><td class="num">± ${ur.half_width_out.toFixed(2)}</td><td class="num">± ${u.half_width_out.toFixed(2)}</td></tr>
      <tr><td>${T("Test molecules inside their range", "مولکول‌های آزمون درون بازهٔ خود")}</td><td class="num">${pct(ur.coverage_all)}</td><td class="num">${pct(u.coverage_all)}</td></tr>
      <tr><td>${T('Share flagged "unfamiliar"', "سهم مولکول‌های «ناآشنا»")}</td><td class="num">${pct(ur.share_out_of_domain)}</td><td class="num">${pct(u.share_out_of_domain)}</td></tr>
      <tr><td>${T("Error (RMSE) familiar / unfamiliar", "خطا (RMSE) آشنا / ناآشنا")}</td><td class="num">${ur.rmse_in?.toFixed(2)} / ${ur.rmse_out?.toFixed(2) ?? "–"}</td><td class="num">${u.rmse_in?.toFixed(2)} / ${u.rmse_out?.toFixed(2) ?? "–"}</td></tr>
    </tbody></table>`;
  }
  if (t && ts) {
    // The more confident the referee, the more accurate the prediction should be.
    const curve = (c, name, color) => ({ type: "scatter", mode: "lines+markers", name, x: c.map((d) => 100 * d.fraction), y: c.map((d) => 100 * d.within_1),
      line: { color, width: 2.5 }, marker: { size: 7 }, hovertemplate: `${name}<br>most confident %{x:.0f}%<br>within ×10: %{y:.1f}%<extra></extra>` });
    Plotly.newPlot("p-sim", [curve(t.confidence_curve, T("Random split", "تقسیم تصادفی"), css("--accent")), curve(ts.confidence_curve, T("Scaffold split", "تقسیم اسکلتی"), css("--ink-3"))],
      plotLayout({ legend: { orientation: "h", y: 1.12 },
        xaxis: { title: T("Most confident … % of test molecules", "… درصد از مطمئن‌ترین مولکول‌های آزمون"), gridcolor: css("--line"), range: [5, 102] },
        yaxis: { title: T("Predicted within a factor of 10 (%)", "پیش‌بینی در محدودهٔ 10 برابر (%)"), gridcolor: css("--line"), range: [50, 101] } }), PLOT_CONFIG);
    return;
  }
  const b = u.error_by_similarity;
  Plotly.newPlot("p-sim", [{ type: "bar", x: b.map((d) => d.range), y: b.map((d) => d.rmse), text: b.map((d) => `n=${d.n}`), textposition: "outside", cliponaxis: false,
    marker: { color: b.map((d) => (parseFloat(d.range) < 0.3 ? css("--cls-slight") : css("--accent"))) },
    hovertemplate: "similarity %{x}<br>RMSE %{y:.2f}<br>%{text}<extra></extra>" }],
    plotLayout({ xaxis: { title: T("Similarity to closest training molecule", "شباهت به نزدیک‌ترین مولکول آموزشی") }, yaxis: { title: T("Test RMSE", "RMSE آزمون"), gridcolor: css("--line"), rangemode: "tozero" } }), PLOT_CONFIG);
}

/* "Kitchen test": everyday substances, lab value vs SoluAI vs the classic formula (salts: the rules). */
async function household() {
  const el = document.getElementById("household");
  try {
    const h = await api("/api/household");
    const err = (a, b) => `<span class="ltr">${Math.abs(a - b).toFixed(2)}</span>`;
    el.innerHTML = `<div class="table-wrap"><table><thead><tr><th>${T("Where you meet it", "کجا می‌بینیمش")}</th><th>${T("Substance", "ماده")}</th>
      <th class="num">${T("Lab (logS)", "آزمایشگاه (logS)")}</th><th class="num">SoluAI</th><th class="num">${T("Classic formula", "فرمول کلاسیک")}</th><th>${T("Verdict", "نتیجه")}</th></tr></thead><tbody>${h.rows.map((r) => {
        if (r.kind === "ionic") return `<tr><td>${esc(r.where)}</td><td>${esc(capitalize(r.name))} <span class="ltr muted">${esc(r.formula)}</span></td>
          <td class="num ltr">${r.measured_logS === null ? "–" : num(r.measured_logS)}</td><td class="num" colspan="2"><span class="pill">${T("solubility rules", "قواعد انحلال‌پذیری")}</span></td>
          <td><b class="verdict ${r.status}" style="font-size:1rem">${esc(r.status_label)}</b> ${r.agrees ? "✓" : ""}</td></tr>`;
        const has = r.measured !== null;
        return `<tr><td>${esc(r.where)}</td><td><a href="/?q=${encodeURIComponent(r.query)}">${esc(capitalize(r.name))}</a></td>
          <td class="num ltr">${has ? num(r.measured) : "–"}</td><td class="num ltr"><b>${num(r.logS)}</b></td><td class="num ltr">${num(r.classic)}</td>
          <td class="small">${has ? T(`off by ${Math.abs(r.logS - r.measured).toFixed(2)}`, `خطا ${Math.abs(r.logS - r.measured).toFixed(2)}`) : T("no lab value in our data", "مقدار آزمایشگاهی نداریم")}</td></tr>`;
      }).join("")}</tbody></table></div>
      <p class="note" style="margin:16px 0 0">${T(`On the ${h.n_compared} organic substances with a lab value, SoluAI is off by ${h.mean_abs_error_model} log units on average and the classic formula by ${h.mean_abs_error_classic}. Honest note: these molecules were part of the training data, so this is a sanity check, not an exam. The real exam is at the top of this page.`,
        `برای ${h.n_compared} مادهٔ آلی که مقدار آزمایشگاهی دارند، میانگین خطای SoluAI برابر ${h.mean_abs_error_model} و خطای فرمول کلاسیک ${h.mean_abs_error_classic} واحد لگاریتمی است. نکتهٔ صادقانه: این مولکول‌ها در داده‌های آموزشی بوده‌اند؛ پس این یک آزمونِ سلامت است، نه امتحان واقعی. امتحان واقعی در بالای همین صفحه است.`)}</p>`;
  } catch (err) { el.innerHTML = errorBox(err); }
}
