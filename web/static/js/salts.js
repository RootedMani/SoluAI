/* Salts page: the solubility-rules table, a salt look-up and the "mix two solutions" lab. */

let TABLE = null;
const MIX_PRESETS = [["Ag", "NO3", "Na", "Cl"], ["Pb", "NO3", "K", "I"], ["Ba", "Cl", "Na", "SO4"], ["Cu", "SO4", "Na", "OH"], ["Ca", "Cl", "Na", "CO3"], ["Na", "Cl", "K", "NO3"]];

document.addEventListener("DOMContentLoaded", async () => {
  try { TABLE = await api("/api/salts"); } catch (err) { document.getElementById("table").innerHTML = errorBox(err); return; }
  drawTable();
  document.getElementById("rules").innerHTML = TABLE.rules.map((r) => `<li data-rule="${r.key}">${esc(r.text)}</li>`).join("");
  document.getElementById("salt-form").addEventListener("submit", (e) => { e.preventDefault(); lookup(document.getElementById("salt-q").value); });

  const opt = (list) => list.map((i) => `<option value="${i.key}">${i.label} · ${esc(i.name)}</option>`).join("");
  ["c1", "c2"].forEach((id) => { document.getElementById(id).innerHTML = opt(TABLE.cations); });
  ["a1", "a2"].forEach((id) => { document.getElementById(id).innerHTML = opt(TABLE.anions); });
  const f = (k) => TABLE.cells[k].formula;
  document.getElementById("mix-presets").innerHTML = MIX_PRESETS.map((p, i) =>
    `<button class="chip ltr" data-i="${i}">${f(`${p[0]}|${p[1]}`)} + ${f(`${p[2]}|${p[3]}`)}</button>`).join("");
  document.getElementById("mix-presets").onclick = (e) => { if (e.target.dataset.i) { setMix(MIX_PRESETS[e.target.dataset.i]); mix(); } };
  document.getElementById("mix-go").onclick = mix;
  setMix(MIX_PRESETS[0]); mix();

  const q = new URLSearchParams(location.search).get("q");
  if (q) { document.getElementById("salt-q").value = q; lookup(q); } else showCell("Ag", "Cl");
});

function drawTable() {
  const L = TABLE.status_labels, short = { soluble: "S", slightly: "sl", insoluble: "I" };
  const head = `<tr><th></th>${TABLE.anions.map((a) => `<th title="${esc(a.name)}">${a.label}</th>`).join("")}</tr>`;
  const rows = TABLE.cations.map((c) => `<tr><th title="${esc(c.name)}">${c.label}</th>${TABLE.anions.map((a) => {
    const cell = TABLE.cells[`${c.key}|${a.key}`];
    return `<td><button class="cell ${cell.unstable ? "unstable" : cell.status}" data-c="${c.key}" data-a="${a.key}" title="${esc(cell.formula)} · ${esc(cell.unstable ? T("reacts with water", "با آب واکنش می‌دهد") : L[cell.status])}">${cell.unstable ? "–" : short[cell.status]}</button></td>`;
  }).join("")}</tr>`).join("");
  document.getElementById("table").innerHTML = `<table class="salt-table">${head}${rows}</table>`;
  document.getElementById("table").onclick = (e) => { const b = e.target.closest(".cell"); if (b) showCell(b.dataset.c, b.dataset.a); };
  document.getElementById("key").innerHTML = ["soluble", "slightly", "insoluble"].map((s) =>
    `<span><i class="cell ${s}" style="display:inline-grid;width:30px;min-width:30px;height:22px;vertical-align:middle">${short[s]}</i> ${esc(L[s])}</span>`).join("") +
    `<span><i class="cell unstable" style="display:inline-grid;width:30px;min-width:30px;height:22px;vertical-align:middle">–</i> ${T("reacts with water / not stable", "با آب واکنش می‌دهد / ناپایدار")}</span>`;
}

function mark(r) {
  document.querySelectorAll(".cell.active").forEach((c) => c.classList.remove("active"));
  const b = document.querySelector(`.cell[data-c="${r.cation.key}"][data-a="${r.anion.key}"]`);
  if (b) b.classList.add("active");
  document.querySelectorAll("#rules li").forEach((li) => li.classList.toggle("active", li.dataset.rule === r.rule));
}

async function showCell(c, a) {
  const out = document.getElementById("detail");
  try { const r = await api(`/api/salts/${c}/${a}`); out.innerHTML = ionicCard(r); mark(r); } catch (err) { out.innerHTML = errorBox(err); }
}

async function lookup(query) {
  const out = document.getElementById("detail");
  out.innerHTML = loadingBox();
  try {
    const r = await api("/api/predict", { query });
    if (r.kind === "ionic") { out.innerHTML = ionicCard(r); mark(r); history.replaceState(null, "", `?q=${encodeURIComponent(query)}`); }
    else out.innerHTML = `<div class="callout ok">${ICONS.check}<p>${T(`${esc(capitalize(r.name))} is an organic molecule, not a salt. `, `${esc(r.name)} یک مولکول آلی است، نه نمک. `)}
      <a href="/?q=${encodeURIComponent(query)}">${T("See the AI prediction →", "پیش‌بینی هوش مصنوعی را ببینید ←")}</a></p></div>`;
  } catch (err) {
    out.innerHTML = errorBox({ message: T("I don't know this salt.", "این نمک را نمی‌شناسم."),
      hint: T("Try a formula such as AgCl or Ca(OH)2, or a name such as silver nitrate. The table covers 17 cations and 12 anions.",
              "فرمولی مثل AgCl یا Ca(OH)2، یا نامی مثل «نقره نیترات» را امتحان کنید. جدول 17 کاتیون و 12 آنیون را پوشش می‌دهد.") });
  }
}

function setMix(p) { ["c1", "a1", "c2", "a2"].forEach((id, i) => { document.getElementById(id).value = p[i]; }); }

async function mix() {
  const v = (id) => document.getElementById(id).value;
  const out = document.getElementById("mix-out");
  try {
    const r = await api("/api/salts/mix", { a: [v("c1"), v("a1")], b: [v("c2"), v("a2")] });
    const solid = r.precipitates.length > 0;
    out.innerHTML = `<div class="grid grid-2 fade-in" style="align-items:center">
      <div><div class="beaker ${solid ? "" : "clear"}" id="beaker"><div class="liquid"></div><div class="solid"></div></div>
        <p class="small muted" style="text-align:center;margin-top:8px">${solid ? T("cloudy: a solid settles", "کدر: جامد ته‌نشین می‌شود") : T("stays clear", "شفاف می‌ماند")}</p></div>
      <div>
        <div class="callout ${solid ? "warn" : "ok"}" style="margin-bottom:12px">${solid ? ICONS.warn : ICONS.check}<p><b>${esc(r.message)}</b></p></div>
        ${r.equation ? `<p class="kicker">${T("Balanced equation", "معادلهٔ موازنه‌شده")}</p><div class="equation">${esc(r.equation)}</div>` : ""}
        ${r.net_ionic.length ? `<p class="kicker" style="margin-top:12px">${T("Net ionic equation", "معادلهٔ یونی خالص")}</p>${r.net_ionic.map((e) => `<div class="equation" style="margin-bottom:6px">${esc(e)}</div>`).join("")}` : ""}
        ${r.precipitates.map((p) => `<p class="small" style="margin-top:10px"><b class="ltr">${esc(p.formula)}</b>: ${esc(p.rule_text)}</p>`).join("")}
      </div></div>`;
    if (solid) requestAnimationFrame(() => setTimeout(() => document.getElementById("beaker").classList.add("cloudy"), 60));
  } catch (err) { out.innerHTML = errorBox(err); }
}
