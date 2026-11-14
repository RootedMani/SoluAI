/* Quiz page: which of two molecules dissolves better? Answers come from lab measurements. */

let ROUNDS = [], idx = 0, you = 0, ai = 0, level = "easy";

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("level").onclick = (e) => {
    if (!e.target.dataset.v) return;
    level = e.target.dataset.v;
    document.querySelectorAll("#level button").forEach((b) => b.classList.toggle("active", b === e.target));
    start();
  };
  start();
});

async function start() {
  const out = document.getElementById("quiz");
  out.innerHTML = loadingBox(T("Choosing molecules…", "در حال انتخاب مولکول‌ها…"));
  try { ROUNDS = (await api(`/api/quiz?n=8&level=${level}`)).rounds; } catch (err) { out.innerHTML = errorBox(err); return; }
  idx = 0; you = 0; ai = 0;
  show();
}

function show() {
  const out = document.getElementById("quiz");
  if (idx >= ROUNDS.length) { finish(); return; }
  const r = ROUNDS[idx];
  const card = (m, i) => `<button class="quiz-card" data-i="${i}">
      <div class="mol-paper">${m.svg}</div>
      <div class="mol-name ltr" style="font-size:1.25rem" dir="auto">${esc(capitalize(m.name))}</div>
      <div class="mol-meta" style="text-align:center">${formula(m.formula)}</div>
      <div class="reveal" style="margin-top:10px"></div></button>`;
  out.innerHTML = `<div class="card fade-in">
    <div class="row spread" style="margin-bottom:10px"><b>${T("Question", "پرسش")} ${idx + 1} / ${ROUNDS.length}</b>
      <span class="muted">${T("You", "شما")} <b>${you}</b> · SoluAI <b>${ai}</b></span></div>
    <div class="progress"><i style="width:${100 * idx / ROUNDS.length}%"></i></div>
    <h2 style="margin:18px 0">${T("Which one dissolves better in water?", "کدام‌یک در آب بهتر حل می‌شود؟")}</h2>
    <div class="quiz-pair">${card(r.molecules[0], 0)}<span class="quiz-vs">${T("or", "یا")}</span>${card(r.molecules[1], 1)}</div>
    <div id="after" style="margin-top:18px"></div></div>`;
  out.querySelectorAll(".quiz-card").forEach((b) => { b.onclick = () => answer(Number(b.dataset.i)); });
}

function answer(choice) {
  const r = ROUNDS[idx], right = choice === r.answer, aiRight = r.model_answer === r.answer;
  if (right) you++;
  if (aiRight) ai++;
  document.querySelectorAll(".quiz-card").forEach((b, i) => {
    b.disabled = true;
    if (i === r.answer) b.classList.add("right"); else if (i === choice) b.classList.add("wrong");
    const m = r.molecules[i];
    b.querySelector(".reveal").innerHTML = `<div>${T("measured", "اندازه‌گیری")}: <b class="ltr">logS ${num(m.measured)}</b></div>
      <div class="small muted">SoluAI: <span class="ltr">${num(m.predicted)}</span></div>`;
  });
  const gap = Math.abs(r.molecules[0].measured - r.molecules[1].measured);
  const times = gap >= 6 ? T("more than a million", "بیش از یک میلیون") : big(Number(Math.pow(10, gap).toPrecision(2)));
  const w = r.molecules[r.answer], l = r.molecules[1 - r.answer];
  document.getElementById("after").innerHTML = `
    <div class="callout ${right ? "ok" : "warn"}">${right ? ICONS.check : ICONS.warn}<p><b>${right ? T("Correct!", "درست است!") : T("Not this time.", "این بار نه.")}</b>
      ${T(`${esc(capitalize(w.name))} is about ${times} times more soluble (in moles per litre).`, `${esc(capitalize(w.name))} حدود ${times} برابر محلول‌تر است (بر حسب مول بر لیتر).`)}
      ${aiRight ? T("SoluAI also got it right.", "SoluAI هم درست جواب داد.") : T("SoluAI got this one wrong!", "SoluAI این یکی را اشتباه کرد!")}</p></div>
    <div class="row" style="margin-top:14px"><button class="btn" id="next">${idx + 1 < ROUNDS.length ? T("Next question", "پرسش بعدی") : T("See my score", "دیدن امتیاز")}</button>
      <a class="btn ghost" href="/compare?m=${encodeURIComponent(w.query)}&m=${encodeURIComponent(l.query)}" target="_blank">${T("See why (compare)", "چرا؟ (مقایسه)")}</a></div>`;
  document.getElementById("next").onclick = () => { idx++; show(); };
  document.getElementById("next").focus();
}

function finish() {
  const n = ROUNDS.length;
  const msg = you > ai ? T("You beat the AI!", "شما هوش مصنوعی را شکست دادید!") : you === ai ? T("A draw with the AI.", "با هوش مصنوعی مساوی شدید.") : T("The AI wins this round.", "این دور را هوش مصنوعی برد.");
  document.getElementById("quiz").innerHTML = `<div class="card fade-in" style="text-align:center">
    <div class="kicker">${T("Final score", "امتیاز نهایی")}</div>
    <div class="grid grid-2" style="max-width:520px;margin:14px auto">
      <div><div class="score ltr">${you} / ${n}</div><div class="muted">${T("You", "شما")}</div></div>
      <div><div class="score ltr">${ai} / ${n}</div><div class="muted">SoluAI</div></div></div>
    <h2>${msg}</h2>
    <p class="muted" style="max-width:640px;margin:0 auto 18px">${T("To be fair: SoluAI studied these molecules during training, so it has an advantage. On molecules it has never seen, its typical error is about 1 log unit.",
      "منصفانه بگوییم: SoluAI این مولکول‌ها را هنگام آموزش دیده و برتری دارد. برای مولکول‌هایی که هرگز ندیده، خطای معمول آن حدود 1 واحد لگاریتمی است.")}</p>
    <button class="btn" id="again">${T("Play again", "دوباره بازی کن")}</button></div>`;
  document.getElementById("again").onclick = start;
}
