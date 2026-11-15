/* Predict page: search, molecule picker, drawing editor and the detailed result view. */

let CURRENT = null;   // the last result shown (used by the editor and the "more soluble" ideas)

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("search-icon").innerHTML = ICONS.search;
  document.getElementById("draw-icon").innerHTML = ICONS.pencil;
  const form = document.getElementById("search-form");
  const input = document.getElementById("query");
  form.addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });
  setupEditor();

  // Molecule picker (offline list), grouped by category, plus autocomplete for every known name.
  try {
    const { categories, names: all } = await api("/api/molecules");
    const tabs = document.getElementById("picker-tabs");
    const chips = document.getElementById("chips");
    const names = Object.keys(categories);
    const show = (cat) => {
      tabs.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.cat === cat));
      chips.innerHTML = categories[cat].map((m) => `<button class="chip" data-name="${esc(m.name)}">${esc(m.label)}</button>`).join("");
    };
    tabs.innerHTML = names.map((c) => `<button class="tab" data-cat="${esc(c)}">${esc(c)}</button>`).join("");
    tabs.onclick = (e) => e.target.dataset.cat && show(e.target.dataset.cat);
    chips.onclick = (e) => { if (e.target.dataset.name) { input.value = e.target.textContent; run(e.target.dataset.name); } };
    show(names[0]);
    document.getElementById("names").innerHTML = all.map((n) => `<option value="${esc(n)}">`).join("");
  } catch { /* picker is optional */ }

  const q = new URLSearchParams(location.search).get("q");
  if (q) { input.value = q; run(q); }
});
// A different model was chosen: show the same molecule again.
document.addEventListener("modelchange", () => { if (CURRENT && CURRENT.kind === "organic") run(CURRENT.query); });

async function run(query) {
  const out = document.getElementById("result");
  const btn = document.getElementById("go");
  btn.disabled = true;
  out.innerHTML = loadingBox();
  try {
    const r = await api("/api/predict", { query });
    CURRENT = r;
    if (r.kind === "ionic") {
      out.innerHTML = `<div style="margin-top:28px">${printHeader()}${ionicCard(r, { link: true })}</div>`;
    } else {
      out.innerHTML = renderResult(r);
      wire(out, r);
    }
    history.replaceState(null, "", `?q=${encodeURIComponent(query)}`);
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.name === (r.name_en || r.name)));
  } catch (err) {
    CURRENT = null;
    out.innerHTML = `<div style="margin-top:24px;max-width:760px">${errorBox(err)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

/* Only visible on paper / in the saved PDF. */
function printHeader() {
  const today = new Date().toLocaleDateString(FA ? "fa-IR" : "en-GB", { year: "numeric", month: "long", day: "numeric" });
  return `<div class="print-only" style="margin-bottom:14px">
    <div class="row spread"><b style="font-size:1.3rem">SoluAI · ${T("solubility report", "گزارش انحلال‌پذیری")}</b><span>${esc(today)}</span></div>
    <p class="small muted" style="margin:4px 0 0">${T("Machine-learning estimate of water solubility at about 25 °C. Not a lab measurement.",
      "برآورد یادگیری ماشین از انحلال‌پذیری در آب در حدود 25 درجهٔ سلسیوس؛ اندازه‌گیری آزمایشگاهی نیست.")}</p></div>`;
}

const WIDE_FA = { "Overall size": "اندازهٔ کلی", "Compared with the average molecule": "در مقایسه با مولکول میانگین",
  "Fragments not present": "قطعه‌هایی که وجود ندارند", "Other": "سایر" };

/* Measured value, SoluAI and the classic hand-made formula on one scale. */
function threeAnswers(r) {
  const m = r.measured, c = r.classic.delaney;
  const row = (who, sub, v) => `<div class="tri-row"><span class="who">${who}<small>${sub}</small></span>
    <span class="tri-track"><span class="tri-dot" style="left:${pct(v)}%"></span></span><span class="val ltr">${num(v)}</span></div>`;
  let verdict;
  if (m) {
    const ea = Math.abs(r.logS - m.logS), ec = Math.abs(c - m.logS);
    const who = ea <= ec ? T("the model is closer", "مدل نزدیک‌تر است") : T("the classic formula is closer here", "این‌جا فرمول کلاسیک نزدیک‌تر است");
    verdict = T(`Compared with the lab value, ${r.model.name} is off by ${ea.toFixed(2)} and the classic formula by ${ec.toFixed(2)} log units: ${who}. `,
      `در مقایسه با مقدار آزمایشگاهی، خطای ${r.model.name} برابر ${ea.toFixed(2)} و خطای فرمول کلاسیک ${ec.toFixed(2)} واحد لگاریتمی است: ${who}. `) +
      T("Honest note: this molecule was in the training data, so the model has seen its answer before.",
        "نکتهٔ صادقانه: این مولکول در داده‌های آموزشی بوده، پس مدل پاسخ آن را قبلاً دیده است.") +
      (m.exact ? "" : T(" (The lab value is for the same molecule with its 3D shape ignored.)", " (مقدار آزمایشگاهی برای همین مولکول بدون در نظر گرفتن شکل سه‌بعدی است.)"));
  } else {
    const d = Math.abs(r.logS - c);
    verdict = T(`No lab value for this molecule in our data. ${r.model.name} and the classic formula differ by ${d.toFixed(2)} log units${d >= 1 ? " - a big gap, so treat the number with care." : "."}`,
      `برای این مولکول مقدار آزمایشگاهی در داده‌های ما نیست. اختلاف ${r.model.name} و فرمول کلاسیک ${d.toFixed(2)} واحد لگاریتمی است${d >= 1 ? "؛ اختلاف بزرگی است، پس با احتیاط به عدد نگاه کنید." : "."}`);
  }
  return `<div class="card fade-in" style="margin-top:20px">
    <div class="card-title"><h3>${T("Three answers to the same question", "سه پاسخ به یک پرسش")}</h3><span class="small muted">logS</span></div>
    <div class="tri">
      ${m ? row(T("Measured in the lab", "اندازه‌گیری آزمایشگاهی"), esc(prettyName(m.name)), m.logS) : ""}
      ${row(esc(r.model.name), T("machine learning, 2026", "یادگیری ماشین"), r.logS)}
      ${row(T("Classic formula", "فرمول کلاسیک"), T("Delaney's equation, 2004", "معادلهٔ دلانی، 2004"), c)}
    </div>
    <p class="small muted" style="margin:14px 0 0">${verdict}</p>
  </div>`;
}

function renderResult(r) {
  const source = { list: T("from the built-in list", "از فهرست داخلی"), smiles: T("from SMILES", "از SMILES"), pubchem: T("found on PubChem", "یافت‌شده در PubChem") }[r.source];
  const wide = Object.entries(r.molecule_wide).map(([k, v]) =>
    `<li><b>${esc(FA ? WIDE_FA[k] || k : k)}</b>: <span class="ltr">${signed(v)}</span></li>`).join("");
  const neighbours = r.domain.neighbours.map((n) => `
    <div class="neighbour"><span class="n-name ltr" title="${esc(n.smiles)}">${esc(prettyName(n.name))}</span>
    <span class="muted">${Math.round(n.similarity * 100)}% ${T("similar", "شباهت")} · logS <b class="ltr">${num(n.logS)}</b></span></div>`).join("");
  return `
  ${printHeader()}
  <div class="result fade-in">
    <div class="card">
      <div class="card-title">
        <div><p class="mol-name">${esc(capitalize(r.name))}</p>
          <div class="mol-meta">${formula(r.formula)} · ${r.mol_weight.toFixed(1)} g/mol · ${esc(source)}</div></div>
        <div class="segmented" role="tablist">
          <button class="active" data-view="explained">${T("Why", "چرا")}</button><button data-view="plain">${T("Structure", "ساختار")}</button>
        </div>
      </div>
      <div class="mol-paper" id="mol-svg">${r.svg_explained}</div>
      ${r.model.id === "v1" ? "" : `<p class="small muted" style="margin:10px 0 0">${T(`Colours: SoluAI v1.0's explanation (v1.0 predicts ${num(r.explained_by.logS)}).`, `رنگ‌ها: توضیح SoluAI v1.0 (پیش‌بینی v1.0: ‎${num(r.explained_by.logS)}‎).`)}</p>`}
      <div class="legend" style="margin-top:12px">
        <span><i style="background:var(--up)"></i>${T("raises solubility", "انحلال‌پذیری را بالا می‌برد")}</span>
        <span><i style="background:var(--down)"></i>${T("lowers solubility", "انحلال‌پذیری را پایین می‌آورد")}</span>
        <span class="mono small">${esc(r.smiles)}</span>
      </div>
      <div class="row no-print" style="margin-top:14px">
        <button class="btn ghost small" id="edit-btn">${ICONS.pencil} ${T("Edit this molecule", "ویرایش همین مولکول")}</button>
        <button class="btn ghost small" id="print-btn">${T("Save as PDF / print", "ذخیره به‌صورت PDF / چاپ")}</button>
      </div>
    </div>

    <div class="card">
      <div class="kicker">${T("Predicted solubility in water (25 °C)", "انحلال‌پذیری پیش‌بینی‌شده در آب (25°C)")} · ${esc(r.model.name)}</div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:8px">
        <div class="big-number">${num(r.logS)}<small>log mol/L</small></div>
        ${badge(r.category)}
      </div>
      <div class="stat-row four">
        <div class="stat"><div class="label">${T("In grams", "به گرم")}</div><div class="value ltr">${fmtConc(r.g_per_L)}</div></div>
        <div class="stat"><div class="label">${T("In a glass of water", "در یک لیوان آب")}</div><div class="value ltr">${glassAmount(r.g_per_L)}</div></div>
        <div class="stat"><div class="label">${T(`${Math.round(r.coverage * 100)}% range`, `بازهٔ ${Math.round(r.coverage * 100)}%`)}</div><div class="value ltr">${num(r.interval[0], 1)} … ${num(r.interval[1], 1)}</div></div>
        <div class="stat"><div class="label">${T("Moles per litre", "مول بر لیتر")}</div><div class="value ltr">${sci(Math.pow(10, r.logS))}</div></div>
      </div>
      <p class="note" style="margin:16px 0 0">${waterNeeded(r.g_per_L)} ${T("Pharmacists would call it", "در داروسازی به آن می‌گویند")}: <b>${uspLabel(r.usp)}</b>.</p>
      ${scaleBar(r)}
      <div style="margin-top:20px">${domainBox(r)}</div>
      <div class="props">${propBoxes(r.descriptors)}</div>
    </div>
  </div>

  ${threeAnswers(r)}
  ${everyModel(r)}

  <div class="grid grid-2 fade-in" style="margin-top:20px">
    <div class="card">
      <h3>${T("Why? In plain words", "چرا؟ به زبان ساده")}</h3>
      <div class="why"><ul>${r.explanation.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>
      ${wide ? `<p class="small muted" style="margin-top:14px">${T("Whole-molecule effects not drawn on atoms (log units):", "اثرهای کلِ مولکول که روی اتم‌ها رسم نشده‌اند (واحد لگاریتمی):")}</p><ul class="small muted">${wide}</ul>` : ""}
    </div>
    <div class="card">
      <div class="card-title"><h3>${T("What pushed the prediction", "چه چیزی پیش‌بینی را جابه‌جا کرد")}</h3><span class="small muted">${T("SHAP, log units", "SHAP، واحد لگاریتمی")}</span></div>
      ${shapBars(r.contributions)}
      <p class="small muted" style="margin-top:14px">${T(`Starts from the average molecule (${num(r.base_value)}); blue bars push solubility up, red bars push it down. They add up to ${num(r.explained_by.logS)}.`,
        `از مولکول میانگین (‎${num(r.base_value)}‎) شروع می‌شود؛ میله‌های آبی انحلال‌پذیری را بالا و میله‌های قرمز پایین می‌برند. جمع آن‌ها ‎${num(r.explained_by.logS)}‎ می‌شود.`)}</p>
      ${explainNote(r)}
    </div>
  </div>

  <div class="card fade-in no-print" style="margin-top:20px" id="ideas-card">
    <div class="card-title"><h3>${T("Make it more soluble", "چطور محلول‌تر شود؟")}</h3><span class="small muted">${T("what drug designers do every day", "کاری که طراحان دارو هر روز انجام می‌دهند")}</span></div>
    <p class="muted" style="margin-bottom:14px">${T("SoluAI can try small changes to this molecule (add an –OH, swap a ring CH for N, shorten a chain…) and rank them by predicted solubility.",
      "SoluAI می‌تواند تغییرهای کوچکی را روی این مولکول امتحان کند (افزودن ‎–OH‎، جایگزینی ‎CH‎ حلقه با ‎N‎، کوتاه کردن زنجیر…) و آن‌ها را بر پایهٔ انحلال‌پذیری پیش‌بینی‌شده مرتب کند.")}</p>
    <button class="btn" id="ideas-btn">${T("Suggest changes", "پیشنهاد تغییر")}</button>
    <div id="ideas"></div>
  </div>

  <div class="card fade-in" style="margin-top:20px">
    <div class="card-title"><h3>${T("Closest molecules the model learned from", "نزدیک‌ترین مولکول‌هایی که مدل از آن‌ها یاد گرفته")}</h3><span class="small muted">${T("measured values", "مقدارهای اندازه‌گیری‌شده")}</span></div>
    <div class="neighbours">${neighbours}</div>
  </div>`;
}

function propBoxes(d) {
  const items = [["logP", d.MolLogP.toFixed(1)], [T("Polar surface", "سطح قطبی"), `${d.TPSA.toFixed(0)} Å²`],
    [T("H-bond donors", "دهندهٔ پیوند H"), d.NumHDonors], [T("H-bond acceptors", "پذیرندهٔ پیوند H"), d.NumHAcceptors],
    [T("Rotatable bonds", "پیوند چرخش‌پذیر"), d.NumRotatableBonds], [T("Rings", "حلقه"), d.RingCount],
    [T("Aromatic rings", "حلقهٔ آروماتیک"), d.NumAromaticRings], [T("Heavy atoms", "اتم سنگین"), d.HeavyAtomCount]];
  return items.map(([k, v]) => `<div><div class="label">${k}</div><div class="value ltr">${v}</div></div>`).join("");
}

function wire(root, r) {
  root.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(".segmented button").forEach((x) => x.classList.toggle("active", x === b));
    root.querySelector("#mol-svg").innerHTML = b.dataset.view === "plain" ? r.svg : r.svg_explained;
  }));
  root.querySelector("#print-btn").onclick = () => window.print();
  root.querySelector("#edit-btn").onclick = () => openEditor(r.smiles);
  root.querySelector("#ideas-btn").onclick = () => loadIdeas(r);
}

/* ------------------------------------------------------------------ "make it more soluble" */
async function loadIdeas(r) {
  const box = document.getElementById("ideas");
  document.getElementById("ideas-btn").remove();
  box.innerHTML = loadingBox(T("Trying small changes…", "در حال امتحان تغییرهای کوچک…"));
  try {
    const s = await api("/api/suggest", { query: r.smiles });
    if (!s.ideas.length) {
      box.innerHTML = `<p class="note">${T("None of the simple changes I know makes this molecule clearly more soluble - it is already very polar, or it has no place for these edits.",
        "هیچ‌کدام از تغییرهای ساده‌ای که بلدم این مولکول را به‌روشنی محلول‌تر نمی‌کند؛ یا همین حالا بسیار قطبی است یا جایی برای این تغییرها ندارد.")}</p>`;
      return;
    }
    box.innerHTML = `<div class="ideas">${s.ideas.map((i) => `
      <button class="idea" data-smiles="${esc(i.smiles)}" title="${esc(i.smiles)}">
        <div class="mol-paper">${i.svg}</div>
        <b>${esc(i.label)}</b>
        <span class="delta-up ltr">${signed(i.delta)} logS</span> <span class="muted small">· ${T(`about ${i.fold}× more soluble`, `حدود ${i.fold} برابر محلول‌تر`)}</span>
        <span class="small muted" style="display:block;margin-top:4px">${esc(i.why)}</span>
      </button>`).join("")}</div>
      <p class="small muted" style="margin:14px 0 0">${T(`Tried ${s.tried} variants; showing the best of each kind. Click one to see its full prediction. These are ideas from a model, not a recipe: a real chemist would also check that the molecule still does its job and can actually be made.`,
        `${s.tried} حالت امتحان شد و بهترینِ هر نوع نشان داده شده است. برای دیدن پیش‌بینی کامل روی هر کدام کلیک کنید. این‌ها ایده‌های یک مدل‌اند، نه دستور ساخت: شیمی‌دان واقعی بررسی می‌کند که مولکول همچنان کار خودش را انجام بدهد و اصلاً قابل ساخت باشد.`)}</p>`;
    box.querySelectorAll(".idea").forEach((b) => { b.onclick = () => { document.getElementById("query").value = b.dataset.smiles; run(b.dataset.smiles); window.scrollTo({ top: 0, behavior: "smooth" }); }; });
  } catch (err) {
    box.innerHTML = errorBox(err);
  }
}

/* ------------------------------------------------------------------ drawing editor (JSME, loaded on first use) */
let EDITOR = null, editorLoading = false, pendingSmiles = null;

function setupEditor() {
  const modal = document.getElementById("draw-modal");
  document.getElementById("draw-btn").onclick = () => openEditor(CURRENT && CURRENT.kind === "organic" ? CURRENT.smiles : "");
  document.getElementById("draw-cancel").onclick = () => modal.classList.remove("open");
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("open"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.classList.remove("open"); });
  document.getElementById("draw-use").onclick = () => {
    const smi = EDITOR ? EDITOR.smiles() : "";
    if (!smi) { document.getElementById("draw-smiles").textContent = T("Draw something first.", "اول چیزی رسم کنید."); return; }
    modal.classList.remove("open");
    document.getElementById("query").value = smi;
    run(smi);
  };
}

function openEditor(smiles) {
  document.getElementById("draw-modal").classList.add("open");
  pendingSmiles = smiles || "";
  if (EDITOR) { loadIntoEditor(); return; }
  if (editorLoading) return;
  editorLoading = true;
  document.getElementById("jsme-box").innerHTML = loadingBox(T("Loading the editor…", "در حال بارگذاری ویرایشگر…"));
  window.jsmeOnLoad = () => {
    document.getElementById("jsme-box").innerHTML = "";
    EDITOR = new JSApplet.JSME("jsme-box", "100%", "380px", { options: "newlook,nohydrogens,nopaste" });
    EDITOR.setCallBack("AfterStructureModified", () => { document.getElementById("draw-smiles").textContent = EDITOR.smiles(); });
    loadIntoEditor();
  };
  const s = document.createElement("script");
  s.src = "/static/vendor/jsme/jsme.nocache.js";
  s.onerror = () => { document.getElementById("jsme-box").innerHTML = errorBox({ message: T("The drawing editor could not be loaded.", "ویرایشگر رسم بارگذاری نشد.") }); editorLoading = false; };
  document.head.appendChild(s);
}

function loadIntoEditor() {
  if (pendingSmiles) EDITOR.readGenericMolecularInput(pendingSmiles); else EDITOR.reset();
  document.getElementById("draw-smiles").textContent = pendingSmiles;
}

/* ------------------------------------------------------------------ every model's answer */
/* The atom colours and SHAP bars always come from SoluAI v1.0: say so when another model is chosen. */
function explainNote(r) {
  if (r.model.id === "v1") return "";
  return `<p class="small muted" style="margin-top:8px">${T(`This explanation is from SoluAI v1.0, the only model that can be explained atom by atom. ${r.model.name} predicts ${num(r.logS)}.`,
    `این توضیح از SoluAI v1.0 است؛ تنها مدلی که می‌توان سهم هر اتم را در آن توضیح داد. پیش‌بینی ${r.model.name} برابر ‎${num(r.logS)}‎ است.`)}</p>`;
}

function everyModel(r) {
  const m = r.measured;
  const row = (x) => `<div class="tri-row zoo-row ${x.id === r.model.id ? "chosen" : ""}" data-model="${x.id}" title="${T("Use this model", "استفاده از این مدل")}">
      <span class="who">${esc(x.name)}<small class="ltr">${T("test", "آزمون")} R² ${x.test_r2.toFixed(2)}${m ? ` · ${T("off by", "خطا")} ${Math.abs(x.logS - m.logS).toFixed(2)}` : ""}</small></span>
      <span class="tri-track"><span class="tri-dot" style="left:${pct(x.logS)}%"></span></span><span class="val ltr">${num(x.logS)}</span></div>`;
  const a = r.agreement;
  const verdict = a.spread <= 0.5
    ? T(`The machine-learning models agree within ${a.spread.toFixed(2)} log units: a good sign.`, `مدل‌های یادگیری ماشین تا ${a.spread.toFixed(2)} واحد لگاریتمی با هم توافق دارند: نشانهٔ خوبی است.`)
    : a.spread <= 1
      ? T(`The models differ by up to ${a.spread.toFixed(2)} log units: the usual amount of disagreement.`, `مدل‌ها تا ${a.spread.toFixed(2)} واحد لگاریتمی با هم اختلاف دارند: اختلافی معمولی.`)
      : T(`The models disagree by ${a.spread.toFixed(2)} log units (more than 10×). Treat any single number with care.`, `اختلاف مدل‌ها ${a.spread.toFixed(2)} واحد لگاریتمی است (بیش از ۱۰ برابر). به هیچ عدد تکی بیش از حد اعتماد نکنید.`);
  return `<div class="card fade-in no-print-break" style="margin-top:20px" id="every-model">
    <div class="card-title"><h3>${T("What every model predicts", "پیش‌بینی همهٔ مدل‌ها")}</h3><span class="small muted">${T("logS · click a row to use that model", "logS · برای انتخاب مدل روی ردیف کلیک کنید")}</span></div>
    <div class="tri">
      ${m ? `<div class="tri-row measured"><span class="who">${T("Measured in the lab", "اندازه‌گیری آزمایشگاهی")}<small>${esc(prettyName(m.name))}</small></span>
        <span class="tri-track"><span class="tri-dot" style="left:${pct(m.logS)}%"></span></span><span class="val ltr">${num(m.logS)}</span></div>` : ""}
      ${r.models.map(row).join("")}
    </div>
    <p class="small muted" style="margin:14px 0 0">${verdict} ${T("\"Test R²\" is each model's score on 1,718 molecules it never saw (1.00 would be perfect).",
      "«R² آزمون» امتیاز هر مدل روی ۱۷۱۸ مولکولی است که هرگز ندیده (۱٫۰۰ یعنی بی‌نقص).")}</p>
  </div>`;
}
document.addEventListener("click", (e) => {
  const row = e.target.closest(".zoo-row");
  if (row && row.dataset.model && row.dataset.model !== MODEL) setModel(row.dataset.model);
});
