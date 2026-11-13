/* Shared code for every SoluAI page:
   - header/footer (so the HTML pages stay short)
   - light/dark theme toggle (remembered in the browser)
   - a tiny API helper with friendly errors
   - render helpers used by both the Predict and Compare pages */

const NAV = [
  ["/", "Predict"],
  ["/compare", "Compare"],
  ["/explore", "Explore the data"],
  ["/model", "How the model works"],
  ["/chemistry", "Chemistry corner"],
];

const ICONS = {
  logo: `<svg viewBox="0 0 40 40" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#0e7490"/></linearGradient></defs><path d="M20 3c6 8 12 14.5 12 21.5A12 12 0 0 1 8 24.5C8 17.5 14 11 20 3z" fill="url(#lg)"/><g fill="none" stroke="#fff" stroke-width="1.8" stroke-linejoin="round"><path d="M20 17l5 2.9v5.8L20 28.6l-5-2.9v-5.8z"/></g><circle cx="20" cy="17" r="1.6" fill="#fff"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>`,
  menu: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`,
  warn: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>`,
};

/* ------------------------------------------------------------------ theme */
function storedTheme() {
  try { return localStorage.getItem("soluai-theme"); } catch { return null; }
}
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("soluai-theme", t); } catch { /* private mode: ignore */ }
  const btn = document.getElementById("theme-btn");
  if (btn) {
    btn.innerHTML = t === "dark" ? ICONS.sun : ICONS.moon;
    btn.title = t === "dark" ? "Switch to light mode" : "Switch to dark mode";
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
      <a class="brand" href="/">${ICONS.logo}<span>Solu<b>AI</b></span></a>
      <nav class="nav" id="nav">${links}</nav>
      <button class="icon-btn menu-btn" id="menu-btn" aria-label="Menu">${ICONS.menu}</button>
      <button class="icon-btn" id="theme-btn" aria-label="Toggle dark mode"></button>
    </div></header>`);
  document.body.insertAdjacentHTML("beforeend", `
    <footer><div class="wrap">
      <span>SoluAI · a chemistry × computer science project · predictions are estimates at ~25 °C</span>
      <span>Data: <a href="https://doi.org/10.1038/s41597-019-0151-1" target="_blank" rel="noopener">AqSolDB</a> ·
        <a href="https://doi.org/10.1021/ci034243x" target="_blank" rel="noopener">ESOL</a> ·
        <a href="/docs" target="_blank">API</a></span>
    </div></footer>`);
  document.getElementById("menu-btn").onclick = () => document.getElementById("nav").classList.toggle("open");
  document.getElementById("theme-btn").onclick = () =>
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  setTheme(document.documentElement.dataset.theme || "light");
}
document.addEventListener("DOMContentLoaded", buildLayout);

/* ------------------------------------------------------------------ API */
async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw { message: "Can't reach the SoluAI server.", hint: "Check your connection. On the free hosting plan the server may need ~30 s to wake up - try again." };
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw (data.detail && data.detail.message) ? data.detail : { message: "Something went wrong.", hint: `Server answered ${res.status}.` };
  return data;
}

/* ------------------------------------------------------------------ helpers */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const CLASS_KEY = { "Highly soluble": "cls-high", "Soluble": "cls-sol", "Slightly soluble": "cls-slight", "Insoluble": "cls-insol" };
const SCALE_MIN = -10, SCALE_MAX = 2;
const pct = (v) => 100 * (Math.min(SCALE_MAX, Math.max(SCALE_MIN, v)) - SCALE_MIN) / (SCALE_MAX - SCALE_MIN);

function fmtConc(gL) {
  if (gL >= 1000) return `${(gL / 1000).toPrecision(2)} kg/L`;
  if (gL >= 1) return `${gL.toPrecision(3)} g/L`;
  if (gL >= 1e-3) return `${(gL * 1000).toPrecision(3)} mg/L`;
  return `${(gL * 1e6).toPrecision(2)} µg/L`;
}
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
function loadingBox(text = "Thinking like a chemist…") {
  return `<div class="loading"><div class="spinner"></div>${esc(text)}</div>`;
}

function badge(category) {
  return `<span class="badge ${CLASS_KEY[category]}">${esc(category)}</span>`;
}

function scaleBar(r) {
  const [lo, hi] = r.interval;
  return `<div class="scale" title="Solubility scale (logS from -10 to +2)">
    <div class="scale-bar">
      <div class="scale-range" style="left:${pct(lo)}%;width:${Math.max(1, pct(hi) - pct(lo))}%"></div>
      <div class="scale-marker" style="left:${pct(r.logS)}%"></div>
    </div>
    <div class="scale-labels"><span>Insoluble</span><span>Slightly</span><span>Soluble</span><span>Highly</span></div>
  </div>`;
}

function domainBox(r) {
  const d = r.domain;
  if (d.in_domain) {
    return `<div class="callout ok">${ICONS.check}<p>Familiar territory: the most similar training molecule is ${Math.round(d.max_similarity * 100)}% similar.</p></div>`;
  }
  const extra = d.out_of_range_descriptors.length ? ` Some properties (${d.out_of_range_descriptors.join(", ")}) are outside the training range.` : "";
  return `<div class="callout warn">${ICONS.warn}<p><b>Outside the model's experience.</b> The closest training molecule is only ${Math.round(d.max_similarity * 100)}% similar, so this prediction is less reliable and the range is wider.${esc(extra)}</p></div>`;
}

function shapBars(contribs, n = 8) {
  const items = contribs.slice(0, n);
  const max = Math.max(0.2, ...items.map((c) => Math.abs(c.shap)));
  return `<div class="bars">${items.map((c) => {
    const w = 50 * Math.abs(c.shap) / max;
    const left = c.shap >= 0 ? 50 : 50 - w;
    const color = c.shap >= 0 ? "var(--up)" : "var(--down)";
    const val = c.value === null || c.value === undefined ? "" : ` = ${Number.isInteger(c.value) ? c.value : Number(c.value).toFixed(2)}`;
    return `<div class="bar-row"><span class="name" title="${esc(c.feature)}${esc(val)}">${esc(c.feature)}<span class="muted">${esc(val)}</span></span>
      <span class="bar-track"><span class="bar-fill" style="left:${left}%;width:${w}%;background:${color}"></span></span>
      <span class="num">${c.shap >= 0 ? "+" : ""}${c.shap.toFixed(2)}</span></div>`;
  }).join("")}</div>`;
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
