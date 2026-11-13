/* Predict page: search, molecule picker and the detailed result view. */

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("search-icon").innerHTML = ICONS.search;
  const form = document.getElementById("search-form");
  const input = document.getElementById("query");
  form.addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });

  // Molecule picker (offline list), grouped by category.
  try {
    const { categories } = await api("/api/molecules");
    const tabs = document.getElementById("picker-tabs");
    const chips = document.getElementById("chips");
    const names = Object.keys(categories);
    const show = (cat) => {
      tabs.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.cat === cat));
      chips.innerHTML = categories[cat].map((m) => `<button class="chip" data-name="${esc(m.name)}">${esc(m.name)}</button>`).join("");
    };
    tabs.innerHTML = names.map((c) => `<button class="tab" data-cat="${esc(c)}">${esc(c)}</button>`).join("");
    tabs.onclick = (e) => e.target.dataset.cat && show(e.target.dataset.cat);
    chips.onclick = (e) => { if (e.target.dataset.name) { input.value = e.target.dataset.name; run(e.target.dataset.name); } };
    show(names[0]);
  } catch { /* picker is optional */ }

  const q = new URLSearchParams(location.search).get("q");
  if (q) { input.value = q; run(q); }
});

async function run(query) {
  const out = document.getElementById("result");
  const btn = document.getElementById("go");
  btn.disabled = true;
  out.innerHTML = loadingBox();
  try {
    const r = await api("/api/predict", { query });
    out.innerHTML = renderResult(r);
    wireToggle(out, r);
    history.replaceState(null, "", `?q=${encodeURIComponent(query)}`);
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.name === r.name));
  } catch (err) {
    out.innerHTML = `<div style="margin-top:24px;max-width:760px">${errorBox(err)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderResult(r) {
  const source = { list: "from the built-in list", smiles: "from SMILES", pubchem: "found on PubChem" }[r.source];
  const wide = Object.entries(r.molecule_wide).map(([k, v]) =>
    `<li><b>${esc(k)}</b>: ${v >= 0 ? "+" : ""}${v.toFixed(2)}</li>`).join("");
  const neighbours = r.domain.neighbours.map((n) => `
    <div class="neighbour"><span class="n-name" title="${esc(n.smiles)}">${esc(prettyName(n.name))}</span>
    <span class="muted">${Math.round(n.similarity * 100)}% similar · logS <b>${n.logS.toFixed(2)}</b></span></div>`).join("");
  return `
  <div class="result fade-in">
    <div class="card">
      <div class="card-title">
        <div><p class="mol-name">${esc(capitalize(r.name))}</p>
          <div class="mol-meta">${formula(r.formula)} · ${r.mol_weight.toFixed(1)} g/mol · ${esc(source)}</div></div>
        <div class="segmented" role="tablist">
          <button class="active" data-view="explained">Why</button><button data-view="plain">Structure</button>
        </div>
      </div>
      <div class="mol-paper" id="mol-svg">${r.svg_explained}</div>
      <div class="legend" style="margin-top:12px">
        <span><i style="background:var(--up)"></i>raises solubility</span>
        <span><i style="background:var(--down)"></i>lowers solubility</span>
        <span class="mono small">${esc(r.smiles)}</span>
      </div>
    </div>

    <div class="card">
      <div class="kicker">Predicted solubility in water (25 °C)</div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:8px">
        <div class="big-number">${r.logS.toFixed(2)}<small>log mol/L</small></div>
        ${badge(r.category)}
      </div>
      <div class="stat-row">
        <div class="stat"><div class="label">In grams</div><div class="value">${fmtConc(r.g_per_L)}</div></div>
        <div class="stat"><div class="label">${Math.round(r.coverage * 100)}% range</div><div class="value">${r.interval[0].toFixed(1)} to ${r.interval[1].toFixed(1)}</div></div>
        <div class="stat"><div class="label">Moles per litre</div><div class="value">${sci(Math.pow(10, r.logS))}</div></div>
      </div>
      ${scaleBar(r)}
      <div style="margin-top:20px">${domainBox(r)}</div>
      <div class="props">${propBoxes(r.descriptors)}</div>
    </div>
  </div>

  <div class="grid grid-2 fade-in" style="margin-top:20px">
    <div class="card">
      <h3>Why? In plain words</h3>
      <div class="why"><ul>${r.explanation.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>
      ${wide ? `<p class="small muted" style="margin-top:14px">Whole-molecule effects not drawn on atoms (log units):</p><ul class="small muted">${wide}</ul>` : ""}
    </div>
    <div class="card">
      <div class="card-title"><h3>What pushed the prediction</h3><span class="small muted">SHAP, log units</span></div>
      ${shapBars(r.contributions)}
      <p class="small muted" style="margin-top:14px">Starts from the average molecule (${r.base_value.toFixed(2)}); blue bars push solubility up, red bars push it down. They add up to ${r.logS.toFixed(2)}.</p>
    </div>
  </div>

  <div class="card fade-in" style="margin-top:20px">
    <div class="card-title"><h3>Closest molecules the model learned from</h3><span class="small muted">measured values</span></div>
    <div class="neighbours">${neighbours}</div>
  </div>`;
}

function propBoxes(d) {
  const items = [["logP", d.MolLogP.toFixed(1)], ["Polar surface", `${d.TPSA.toFixed(0)} Å²`],
    ["H-bond donors", d.NumHDonors], ["H-bond acceptors", d.NumHAcceptors],
    ["Rotatable bonds", d.NumRotatableBonds], ["Rings", d.RingCount],
    ["Aromatic rings", d.NumAromaticRings], ["Heavy atoms", d.HeavyAtomCount]];
  return items.map(([k, v]) => `<div><div class="label">${k}</div><div class="value">${v}</div></div>`).join("");
}

function wireToggle(root, r) {
  root.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(".segmented button").forEach((x) => x.classList.toggle("active", x === b));
    root.querySelector("#mol-svg").innerHTML = b.dataset.view === "plain" ? r.svg : r.svg_explained;
  }));
}
