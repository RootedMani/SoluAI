/* Shared code for every SoluAI page:
   - language (English / Persian) and the T() helper
   - header/footer (so the HTML pages stay short)
   - light/dark theme toggle (remembered in the browser)
   - a tiny API helper with friendly errors
   - render helpers used by several pages */

/* ------------------------------------------------------------------ language */
const LANG = document.documentElement.lang === "fa" ? "fa" : "en";
const FA = LANG === "fa";
/* T("Predict", "پیش‌بینی") returns the text for the current language. */
const T = (en, fa) => (FA && fa !== undefined ? fa : en);
function setLang(l) {
  try { localStorage.setItem("soluai-lang", l); } catch { /* private mode: ignore */ }
  const url = new URL(location.href);
  url.searchParams.delete("lang");
  location.replace(url);
}

const NAV = [
  ["/", T("Predict", "پیش‌بینی")],
  ["/compare", T("Compare", "مقایسه")],
  ["/series", T("Trends", "روندها")],
  ["/salts", T("Salts", "نمک‌ها")],
  ["/quiz", T("Quiz", "مسابقه")],
  ["/batch", T("Batch", "گروهی")],
  ["/explore", T("Data", "داده‌ها")],
  ["/model", T("Model", "مدل")],
  ["/chemistry", T("Chemistry", "شیمی")],
];

const ICONS = {
  logo: `<svg viewBox="0 0 40 40" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#0e7490"/></linearGradient></defs><path d="M20 3c6 8 12 14.5 12 21.5A12 12 0 0 1 8 24.5C8 17.5 14 11 20 3z" fill="url(#lg)"/><g fill="none" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"><path d="M20 17l5 2.9v5.8L20 28.6l-5-2.9v-5.8z"/></g><circle cx="20" cy="17" r="1.6" fill="#fff"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>`,
  menu: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`,
  warn: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>`,
  pencil: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>`,
};

/* ------------------------------------------------------------------ theme */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("soluai-theme", t); } catch { /* private mode: ignore */ }
  const btn = document.getElementById("theme-btn");
  if (btn) {
    btn.innerHTML = t === "dark" ? ICONS.sun : ICONS.moon;
    btn.title = t === "dark" ? T("Switch to light mode", "حالت روشن") : T("Switch to dark mode", "حالت تیره");
  }
  document.dispatchEvent(new CustomEvent("themechange", { detail: t }));
}

/* ------------------------------------------------------------------ layout */
function buildLayout() {
  const here = location.pathname.replace(/\/$/, "") || "/";
  const links = NAV.map(([href, label]) =>
    `<a href="${href}" ${href === here ? 'aria-current="page"' : ""}>${label}</a>`).join("");
  document.body.insertAdjacentHTML("afterbegin", `
    <header class="topbar"><div class="wrap">
      <a class="brand" href="/">${ICONS.logo}<span class="ltr">Solu<b>AI</b></span></a>
      <nav class="nav" id="nav">${links}</nav>
      <button class="icon-btn menu-btn" id="menu-btn" aria-label="${T("Menu", "فهرست")}">${ICONS.menu}</button>
      <button class="icon-btn lang-btn" id="lang-btn" title="${T("نسخهٔ فارسی", "English version")}">${T("فارسی", "English")}</button>
      <button class="icon-btn" id="theme-btn" aria-label="${T("Toggle dark mode", "حالت تیره/روشن")}"></button>
    </div></header>`);
  document.body.insertAdjacentHTML("beforeend", `
    <footer><div class="wrap">
      <span>${T("SoluAI · a chemistry × computer science project · predictions are estimates at ~25 °C",
                "SoluAI · پروژهٔ شیمی × علوم کامپیوتر · پیش‌بینی‌ها برآوردی در حدود 25 درجهٔ سلسیوس‌اند")}</span>
      <span>${T("Data", "داده‌ها")}: <a href="https://doi.org/10.1038/s41597-019-0151-1" target="_blank" rel="noopener">AqSolDB</a> ·
        <a href="https://doi.org/10.1021/ci034243x" target="_blank" rel="noopener">ESOL</a> ·
        <a href="/docs" target="_blank">API</a></span>
    </div></footer>`);
  document.getElementById("menu-btn").onclick = () => document.getElementById("nav").classList.toggle("open");
  document.getElementById("lang-btn").onclick = () => setLang(FA ? "en" : "fa");
  document.getElementById("theme-btn").onclick = () =>
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  setTheme(document.documentElement.dataset.theme || "light");
  const faTitle = document.documentElement.dataset.titleFa;
  if (FA && faTitle) document.title = `SoluAI · ${faTitle}`;
  // Placeholders cannot hold two languages in HTML, so they are set here.
  document.querySelectorAll("[data-ph-fa]").forEach((el) => { if (FA) el.placeholder = el.dataset.phFa; });
}
document.addEventListener("DOMContentLoaded", buildLayout);

/* ------------------------------------------------------------------ API */
async function api(path, body) {
  let opts = {};
  if (body) {
    opts = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lang: LANG, ...body }) };
  } else {
    path += (path.includes("?") ? "&" : "?") + "lang=" + LANG;
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw { message: T("Can't reach the SoluAI server.", "اتصال به سرور SoluAI برقرار نشد."),
            hint: T("Check your connection. On the free hosting plan the server may need ~30 s to wake up - try again.",
                    "اتصال خود را بررسی کنید. در میزبانی رایگان، سرور ممکن است حدود 30 ثانیه برای بیدار شدن زمان بخواهد؛ دوباره امتحان کنید.") };
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw (data.detail && data.detail.message) ? data.detail : { message: T("Something went wrong.", "مشکلی پیش آمد."), hint: `${res.status}` };
  return data;
}

/* ------------------------------------------------------------------ helpers */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const CLASS_KEY = { "Highly soluble": "cls-high", "Soluble": "cls-sol", "Slightly soluble": "cls-slight", "Insoluble": "cls-insol" };
const CLASS_FA = { "Highly soluble": "بسیار محلول", "Soluble": "محلول", "Slightly soluble": "کم‌محلول", "Insoluble": "نامحلول" };
const className = (c) => (FA ? CLASS_FA[c] || c : c);
const SCALE_MIN = -10, SCALE_MAX = 2;
const pct = (v) => 100 * (Math.min(SCALE_MAX, Math.max(SCALE_MIN, v)) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN);
const signed = (v, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const num = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const big = (x) => Math.round(x).toLocaleString("en-US");

function fmtConc(gL) {
  if (gL >= 1000) return `${(gL / 1000).toPrecision(2)} kg/L`;
  if (gL >= 1) return `${gL.toPrecision(3)} g/L`;
  if (gL >= 1e-3) return `${(gL * 1000).toPrecision(3)} mg/L`;
  return `${(gL * 1e6).toPrecision(2)} µg/L`;
}
function fmtMass(g) {
  if (g >= 1) return `${Number(g.toPrecision(2))} g`;
  if (g >= 1e-3) return `${Number((g * 1e3).toPrecision(2))} mg`;
  if (g >= 1e-6) return `${Number((g * 1e6).toPrecision(2))} µg`;
  return `${Number((g * 1e9).toPrecision(2))} ng`;
}
/* How much dissolves in one glass of water (250 mL). */
function glassAmount(gL) {
  return gL >= 1000 ? T("over 250 g", "بیش از 250 g") : fmtMass(gL * 0.25);
}
/* "To dissolve 1 g you need about 3,300 L of water (about 22 bathtubs)." */
function waterNeeded(gL) {
  if (gL >= 1000) return T("It mixes with water in almost any amount.", "تقریباً به هر نسبتی با آب مخلوط می‌شود.");
  const L = 1 / gL;
  let amount, picture = "";
  if (L < 1) amount = `${Number((L * 1000).toPrecision(2))} mL`;
  else amount = `${big(Number(L.toPrecision(2)))} L`;
  if (L >= 2.5e6) picture = T(`about ${big(Number((L / 2.5e6).toPrecision(2)))} Olympic swimming pools`, `حدود ${big(Number((L / 2.5e6).toPrecision(2)))} استخر المپیک`);
  else if (L >= 300) picture = T(`about ${big(Number((L / 150).toPrecision(2)))} bathtubs`, `حدود ${big(Number((L / 150).toPrecision(2)))} وان حمام`);
  else if (L >= 3) picture = T(`about ${big(Number((L / 1.5).toPrecision(2)))} large bottles`, `حدود ${big(Number((L / 1.5).toPrecision(2)))} بطری بزرگ`);
  else if (L >= 0.02 && L < 1) picture = T(`about ${Number((L / 0.005).toPrecision(2))} teaspoons`, `حدود ${Number((L / 0.005).toPrecision(2))} قاشق چای‌خوری`);
  const tail = picture ? ` (${picture})` : "";
  return T(`To dissolve 1 g you need about ${amount} of water${tail}.`, `برای حل کردن 1 گرم از آن، حدود ‎${amount}‎ آب لازم است${tail}.`);
}
const USP = {
  very_soluble: ["Very soluble", "بسیار محلول"], freely_soluble: ["Freely soluble", "به‌آسانی محلول"],
  soluble: ["Soluble", "محلول"], sparingly_soluble: ["Sparingly soluble", "نسبتاً کم‌محلول"],
  slightly_soluble: ["Slightly soluble", "کم‌محلول"], very_slightly_soluble: ["Very slightly soluble", "بسیار کم‌محلول"],
  practically_insoluble: ["Practically insoluble", "عملاً نامحلول"],
};
const uspLabel = (k) => (USP[k] ? T(USP[k][0], USP[k][1]) : k);

/* 0.00123 -> "1.2 × 10⁻³" */
function sci(x) {
  if (x >= 0.01 && x < 1000) return x.toPrecision(2);
  const e = Math.floor(Math.log10(x));
  const sup = String(e).split("").map((c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]).join("");
  return `${(x / 10 ** e).toFixed(1)} × 10${sup}`;
}
/* "C9H8O4" -> C<sub>9</sub>H<sub>8</sub>O<sub>4</sub> */
function formula(f) { return esc(f).replace(/(\d+)/g, "<sub>$1</sub>"); }
/* Dataset names sometimes use underscores ("Acetylsalicylic_acid"). */
function prettyName(n) { return capitalize(String(n).replace(/_/g, " ")); }
function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

function errorBox(err) {
  return `<div class="callout error fade-in">${ICONS.warn}<div><p><b>${esc(err.message)}</b></p>${err.hint ? `<p class="small">${esc(err.hint)}</p>` : ""}</div></div>`;
}
function loadingBox(text = T("Thinking like a chemist…", "در حال فکر کردن مثل یک شیمی‌دان…")) {
  return `<div class="loading"><div class="spinner"></div>${esc(text)}</div>`;
}

function badge(category) {
  return `<span class="badge ${CLASS_KEY[category]}">${esc(className(category))}</span>`;
}

function scaleBar(r) {
  const [lo, hi] = r.interval;
  return `<div class="scale" title="logS −10 … +2">
    <div class="scale-bar">
      <div class="scale-range" style="left:${pct(lo)}%;width:${Math.max(1, pct(hi) - pct(lo))}%"></div>
      <div class="scale-marker" style="left:${pct(r.logS)}%"></div>
    </div>
    <div class="scale-labels"><span>${T("Insoluble", "نامحلول")}</span><span>${T("Slightly", "کم‌محلول")}</span><span>${T("Soluble", "محلول")}</span><span>${T("Highly", "بسیار محلول")}</span></div>
  </div>`;
}

/* Model v2: the "referee" gives every prediction a confidence label from the width of its 90% range. */
const CONFIDENCE = {
  high: { en: "High confidence", fa: "اطمینان بالا", cls: "ok" },
  medium: { en: "Medium confidence", fa: "اطمینان متوسط", cls: "ok" },
  low: { en: "Low confidence", fa: "اطمینان پایین", cls: "warn" },
};
function confidencePill(level) {
  const c = CONFIDENCE[level];
  if (!c) return "";
  return level === "low" ? `<span class="pill" style="color:var(--warn-ink)">⚠ ${T("low", "پایین")}</span>`
    : `<span class="pill ${level === "high" ? "accent" : ""}">${level === "high" ? T("high", "بالا") : T("medium", "متوسط")}</span>`;
}
function confidenceBox(r) {
  const d = r.domain, c = CONFIDENCE[d.confidence], sim = Math.round(d.max_similarity * 100);
  const hw = (r.interval[1] - r.interval[0]) / 2, factor = Math.pow(10, hw);
  const f = factor < 10 ? factor.toFixed(1) : String(Math.round(factor / 5) * 5);
  const why = !d.in_domain
    ? T(`The closest training molecule is only ${sim}% similar (or a property is outside the training range), so the model has almost no experience here. Treat the number as a rough guide.`,
      `نزدیک‌ترین مولکول آموزشی فقط ${sim}% شباهت دارد (یا یکی از ویژگی‌ها بیرون از بازهٔ آموزش است)؛ پس مدل این‌جا تقریباً تجربه‌ای ندارد. این عدد را فقط یک راهنمای تقریبی بدانید.`)
    : d.confidence === "high"
    ? T("Molecules like this one were predicted well when the model had not seen them.", "مدل مولکول‌هایی شبیه این را، وقتی آن‌ها را ندیده بود، خوب پیش‌بینی کرده است.")
    : d.confidence === "medium"
      ? T("The model has some experience with molecules like this one.", "مدل با مولکول‌هایی شبیه این تا حدی تجربه دارد.")
      : T("Molecules like this one were hard for the model, or it has seen few of them. Treat the number as a rough guide.", "مولکول‌هایی شبیه این برای مدل سخت بوده‌اند یا نمونه‌های کمی از آن‌ها دیده است. این عدد را فقط یک راهنمای تقریبی بدانید.");
  const extra = d.out_of_range_descriptors.length ? T(` Some properties (${d.out_of_range_descriptors.join(", ")}) are outside the training range.`,
    ` برخی ویژگی‌ها (‎${d.out_of_range_descriptors.join(", ")}‎) بیرون از بازهٔ داده‌های آموزشی‌اند.`) : "";
  return `<div class="callout ${c.cls}">${c.cls === "ok" ? ICONS.check : ICONS.warn}<p><b>${T(c.en, c.fa)}.</b> ${T(
    `The 90% range is ± ${hw.toFixed(1)} log units: the true solubility is probably within a factor of ${f} of the prediction.`,
    `بازهٔ 90% برابر ± ${hw.toFixed(1)} واحد لگاریتمی است: انحلال‌پذیری واقعی احتمالاً حداکثر ${f} برابر با پیش‌بینی فاصله دارد.`)} ${why}${
    d.in_domain ? ` ${T(`The most similar training molecule is ${sim}% similar.`, `شبیه‌ترین مولکول آموزشی ${sim}% شباهت دارد.`)}` : ""}${esc(extra)}</p></div>`;
}

function domainBox(r) {
  const d = r.domain, sim = Math.round(d.max_similarity * 100);
  if (d.confidence) return confidenceBox(r);
  if (d.in_domain) {
    return `<div class="callout ok">${ICONS.check}<p>${T(`Familiar territory: the most similar training molecule is ${sim}% similar.`,
      `محدودهٔ آشنا: شبیه‌ترین مولکول آموزشی ${sim}% به این مولکول شباهت دارد.`)}</p></div>`;
  }
  const extra = d.out_of_range_descriptors.length ? T(` Some properties (${d.out_of_range_descriptors.join(", ")}) are outside the training range.`,
    ` برخی ویژگی‌ها (‎${d.out_of_range_descriptors.join(", ")}‎) بیرون از بازهٔ داده‌های آموزشی‌اند.`) : "";
  return `<div class="callout warn">${ICONS.warn}<p><b>${T("Outside the model's experience.", "بیرون از تجربهٔ مدل.")}</b> ${T(
    `The closest training molecule is only ${sim}% similar, so this prediction is less reliable and the range is wider.`,
    `نزدیک‌ترین مولکول آموزشی فقط ${sim}% شباهت دارد؛ پس این پیش‌بینی کمتر قابل‌اعتماد و بازهٔ آن پهن‌تر است.`)}${esc(extra)}</p></div>`;
}

const FEATURE_FA = {
  MolLogP: "‎logP‎ (چربی‌دوستی)", MolWt: "جرم مولی", TPSA: "سطح قطبی", NumHDonors: "دهنده‌های پیوند هیدروژنی",
  NumHAcceptors: "پذیرنده‌های پیوند هیدروژنی", NumRotatableBonds: "پیوندهای چرخش‌پذیر", AromaticProportion: "سهم اتم‌های آروماتیک",
  FractionCSP3: "سهم کربن‌های ‎sp³‎", RingCount: "تعداد حلقه‌ها", NumAromaticRings: "حلقه‌های آروماتیک", HeavyAtomCount: "اتم‌های سنگین",
  NumHeteroatoms: "هترواتم‌ها", MolMR: "شکست مولی", NumValenceElectrons: "الکترون‌های ظرفیت",
  fp_present: "قطعه‌های موجود", fp_absent: "قطعه‌های غایب",
  family_surface: "سطح مولکول بر پایهٔ قطبیت", family_groups: "شمار گروه‌های عاملی", family_shape: "شکل و شاخه‌داری",
  family_charge: "بارها و حالت‌های الکترونی", family_size: "سنجه‌های دیگر اندازه", family_other: "توصیف‌گرهای دیگر",
};
function shapBars(contribs, n = 8) {
  const items = contribs.slice(0, n);
  const max = Math.max(0.2, ...items.map((c) => Math.abs(c.shap)));
  return `<div class="bars">${items.map((c) => {
    const w = 50 * Math.abs(c.shap) / max;
    const left = c.shap >= 0 ? 50 : 50 - w;
    const color = c.shap >= 0 ? "var(--up)" : "var(--down)";
    const val = c.value === null || c.value === undefined ? "" : ` = ${Number.isInteger(c.value) ? c.value : Number(c.value).toFixed(2)}`;
    const label = FA ? FEATURE_FA[c.key] || c.feature : c.feature;
    return `<div class="bar-row"><span class="name" title="${esc(label)}${esc(val)}">${esc(label)}<span class="muted ltr">${esc(val)}</span></span>
      <span class="bar-track"><span class="bar-fill" style="left:${left}%;width:${w}%;background:${color}"></span></span>
      <span class="num ltr">${signed(c.shap)}</span></div>`;
  }).join("")}</div>`;
}

/* A salt answered by the solubility rules (used on the Predict, Compare and Salts pages). */
function ionicCard(r, opts = {}) {
  const m = r.measured;
  const measured = m ? `<div class="stat-row">
      <div class="stat"><div class="label">${T("Measured (20 °C)", "اندازه‌گیری‌شده (20°C)")}</div><div class="value ltr">${m.g_per_100mL >= 0.01 ? m.g_per_100mL : sci(m.g_per_100mL)} g / 100 mL</div></div>
      <div class="stat"><div class="label">${T("Moles per litre", "مول بر لیتر")}</div><div class="value ltr">${sci(m.mol_per_L)}</div></div>
      <div class="stat"><div class="label">logS</div><div class="value ltr">${num(m.logS)}</div></div>
    </div>
    <p class="small muted" style="margin-top:10px">${r.agrees_with_measurement
      ? T("✓ The rule agrees with the lab measurement.", "✓ قاعده با اندازه‌گیری آزمایشگاهی هم‌خوانی دارد.")
      : T("The rule and the measurement do not quite agree here: rules are a simplification.", "این‌جا قاعده و اندازه‌گیری کاملاً هم‌خوان نیستند؛ قاعده‌ها ساده‌سازی‌اند.")}
      ${T("(Source of the value: solubility table, 20 °C.)", "(منبع عدد: جدول انحلال‌پذیری، 20°C.)")}</p>` : "";
  return `<div class="card fade-in">
    <div class="row spread"><div>
      <p class="mol-name"><span class="ltr">${esc(r.formula)}</span> · ${esc(capitalize(r.name))}</p>
      <div class="mol-meta">${r.molar_mass.toFixed(1)} g/mol · ${T("ionic compound (salt)", "ترکیب یونی (نمک)")}</div></div>
      <span class="pill accent">${T("Answered by the solubility rules, not by the AI model", "پاسخ با قواعد انحلال‌پذیری، نه با مدل هوش مصنوعی")}</span>
    </div>
    <div class="row" style="margin-top:16px;gap:18px"><div class="verdict ${r.status}">${esc(r.status_label)}</div>
      ${r.exception ? `<span class="pill">${T("exception", "استثنا")}</span>` : ""}</div>
    <p style="margin-top:10px"><b>${T("Rule", "قاعدهٔ")} ${r.rule_number}.</b> ${esc(r.rule_text)} <span class="muted">${esc(r.why)}</span></p>
    ${r.note ? `<div class="callout warn" style="margin-bottom:14px">${ICONS.warn}<p>${esc(r.note)}</p></div>` : ""}
    <div class="equation">${esc(r.equation)}</div>
    <p class="small muted" style="margin-top:8px">${r.status === "soluble"
      ? T("In water the crystal breaks apart into separate ions, each surrounded by water molecules.", "در آب، بلور به یون‌های جدا از هم تفکیک می‌شود و مولکول‌های آب دور هر یون را می‌گیرند.")
      : T("Only a tiny part dissolves: the attraction between the ions in the crystal is stronger than water's pull on them.", "فقط مقدار بسیار کمی حل می‌شود: جاذبهٔ میان یون‌ها در بلور از کشش آب بر آن‌ها قوی‌تر است.")}</p>
    ${measured}
    ${opts.link ? `<p class="no-print" style="margin:14px 0 0"><a href="/salts">${T("Open the solubility table and the mixing lab →", "جدول انحلال‌پذیری و آزمایشگاه اختلاط ←")}</a></p>` : ""}
  </div>`;
}

/* Plotly look that follows the current theme. */
function plotLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font"), color: css("--ink-2"), size: 14 },
    margin: { l: 60, r: 20, t: 20, b: 55 },
    xaxis: { gridcolor: css("--line"), zerolinecolor: css("--line"), linecolor: css("--line") },
    yaxis: { gridcolor: css("--line"), zerolinecolor: css("--line"), linecolor: css("--line") },
    hoverlabel: { font: { family: css("--font") } },
    legend: { orientation: "h", y: -0.2 },
  }, extra);
}
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/* Save text as a file (used for CSV downloads). */
function downloadText(filename, text, type = "text/csv") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + text], { type: `${type};charset=utf-8` }));
  a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
