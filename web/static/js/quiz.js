/* Quiz page: which of two molecules dissolves better? Answers come from lab measurements.
   Four levels; your opponent is the model chosen in the picker, and at the end every model's score is shown. */

let ROUNDS = [], idx = 0, you = 0, ai = 0, level = "easy";
try { level = localStorage.getItem("soluai-quiz-level") || "easy"; } catch { /* private mode */ }

const LEVEL_HELP = {
  easy: T("Everyday molecules, and the two differ at least 100×.", "مولکول‌های روزمره، و اختلاف دو مولکول دست‌کم ۱۰۰ برابر است."),
  medium: T("Everyday molecules, but the gap can be as small as 10×.", "مولکول‌های روزمره، اما اختلاف می‌تواند فقط ۱۰ برابر باشد."),
  hard: T("Any molecule from the dataset (many you have never heard of), gap at least 10×.", "هر مولکولی از مجموعه‌داده (بسیاری را هرگز نشنیده‌اید)، اختلاف دست‌کم ۱۰ برابر."),
  expert: T("Look-alike pairs: the two molecules share most of their structure, and differ only about 5×. Spot the small change that matters.",
    "جفت‌های شبیه هم: دو مولکول بیشتر ساختارشان مشترک است و فقط حدود ۵ برابر اختلاف دارند. تغییر کوچکِ مهم را پیدا کنید."),
};

document.addEventListener("DOMContentLoaded", () => {
  const seg = document.getElementById("level");
  seg.onclick = (e) => {
    const b = e.target.closest("[data-v]");
    if (!b) return;
    level = b.dataset.v;
    try { localStorage.setItem("soluai-quiz-level", level); } catch { /* private mode */ }
    start();
  };
  start();
});
document.addEventListener("modelchange", () => start());

const opponent = () => modelName(MODEL);

async function start() {
  document.querySelectorAll("#level button").forEach((b) => b.classList.toggle("active", b.dataset.v === level));
  document.getElementById("level-help").textContent = LEVEL_HELP[level];
  const out = document.getElementById("quiz");
  out.innerHTML = loadingBox(T("Choosing molecules…", "در حال انتخاب مولکول‌ها…"));
  await loadModels().then((r) => { if (r.models.length) MODEL_LIST = r.models; });
  try { ROUNDS = (await api(`/api/quiz?n=8&level=${level}`)).rounds; } catch (err) { out.innerHTML = errorBox(err); return; }
  if (!ROUNDS.length) { out.innerHTML = errorBox({ message: T("No questions could be made for this level.", "برای این سطح پرسشی ساخته نشد.") }); return; }
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
      <span class="muted">${T("You", "شما")} <b>${you}</b> · ${esc(opponent())} <b>${ai}</b></span></div>
    <div class="progress"><i style="width:${100 * idx / ROUNDS.length}%"></i></div>
    <h2 style="margin:18px 0">${T("Which one dissolves better in water?", "کدام‌یک در آب بهتر حل می‌شود؟")}</h2>
    <div class="quiz-pair">${card(r.molecules[0], 0)}<span class="quiz-vs">${T("or", "یا")}</span>${card(r.molecules[1], 1)}</div>
    <div id="after" style="margin-top:18px"></div></div>`;
  out.querySelectorAll(".quiz-card").forEach((b) => { b.onclick = () => answer(Number(b.dataset.i)); });
}

function answer(choice) {
  const r = ROUNDS[idx], right = choice === r.answer, aiRight = r.model_answer === r.answer;
  r.you = choice;
  if (right) you++;
  if (aiRight) ai++;
  document.querySelectorAll(".quiz-card").forEach((b, i) => {
    b.disabled = true;
    if (i === r.answer) b.classList.add("right"); else if (i === choice) b.classList.add("wrong");
    const m = r.molecules[i];
    b.querySelector(".reveal").innerHTML = `<div>${T("measured", "اندازه‌گیری")}: <b class="ltr">logS ${num(m.measured)}</b></div>
      <div class="small muted">${esc(opponent())}: <span class="ltr">${num(m.predicted)}</span></div>`;
  });
  const gap = Math.abs(r.molecules[0].measured - r.molecules[1].measured);
  const times = gap >= 6 ? T("more than a million", "بیش از یک میلیون") : big(Number(Math.pow(10, gap).toPrecision(2)));
  const w = r.molecules[r.answer], l = r.molecules[1 - r.answer];
  const others = r.model_answers ? Object.entries(r.model_answers).filter(([id]) => id !== MODEL) : [];
  const nRight = others.filter(([, a]) => a === r.answer).length;
  document.getElementById("after").innerHTML = `
    <div class="callout ${right ? "ok" : "warn"}">${right ? ICONS.check : ICONS.warn}<p><b>${right ? T("Correct!", "درست است!") : T("Not this time.", "این بار نه.")}</b>
      ${T(`${esc(capitalize(w.name))} is about ${times} times more soluble (in moles per litre).`, `${esc(capitalize(w.name))} حدود ${times} برابر محلول‌تر است (بر حسب مول بر لیتر).`)}
      ${aiRight ? T(`${esc(opponent())} also got it right.`, `${esc(opponent())} هم درست جواب داد.`) : T(`${esc(opponent())} got this one wrong!`, `${esc(opponent())} این یکی را اشتباه کرد!`)}
      ${others.length ? `<span class="muted">${T(`Other models: ${nRight} of ${others.length} right.`, `مدل‌های دیگر: ${nRight} از ${others.length} درست.`)}</span>` : ""}</p></div>
    <div class="row" style="margin-top:14px"><button class="btn" id="next">${idx + 1 < ROUNDS.length ? T("Next question", "پرسش بعدی") : T("See my score", "دیدن امتیاز")}</button>
      <a class="btn ghost" href="/compare?m=${encodeURIComponent(w.query)}&m=${encodeURIComponent(l.query)}" target="_blank">${T("See why (compare)", "چرا؟ (مقایسه)")}</a></div>`;
  document.getElementById("next").onclick = () => { idx++; show(); };
  document.getElementById("next").focus();
}

/* Everyone's score on the same questions: you and every model. */
function leaderboard() {
  const n = ROUNDS.length;
  const rows = [{ id: "you", name: T("You", "شما"), score: you }];
  if (ROUNDS[0].model_answers) {
    Object.keys(ROUNDS[0].model_answers).forEach((id) => {
      rows.push({ id, name: modelName(id), score: ROUNDS.filter((r) => r.model_answers[id] === r.answer).length });
    });
  } else rows.push({ id: MODEL, name: opponent(), score: ai });
  rows.sort((a, b) => b.score - a.score || (a.id === "you" ? -1 : b.id === "you" ? 1 : 0));
  let rank = 0, prev = null;
  return `<div class="table-wrap" style="max-width:560px;margin:18px auto;text-align:start"><table class="leader"><thead><tr><th>#</th><th>${T("Player", "بازیکن")}</th><th class="num">${T("Score", "امتیاز")}</th></tr></thead><tbody>${rows.map((r, i) => {
    if (r.score !== prev) { rank = i + 1; prev = r.score; }
    return `<tr class="${r.id === "you" || r.id === MODEL ? "me" : ""}"><td class="rank">${rank}</td><td>${esc(r.name)}${r.id === MODEL ? ` <span class="pill accent">${T("opponent", "حریف")}</span>` : ""}</td><td class="num ltr">${r.score} / ${n}</td></tr>`;
  }).join("")}</tbody></table></div>`;
}

function finish() {
  const n = ROUNDS.length;
  const msg = you > ai ? T(`You beat ${opponent()}!`, `شما ${opponent()} را شکست دادید!`) : you === ai ? T(`A draw with ${opponent()}.`, `با ${opponent()} مساوی شدید.`) : T(`${opponent()} wins this round.`, `این دور را ${opponent()} برد.`);
  const levels = ["easy", "medium", "hard", "expert"], nextLevel = levels[levels.indexOf(level) + 1];
  document.getElementById("quiz").innerHTML = `<div class="card fade-in" style="text-align:center">
    <div class="kicker">${T("Final score", "امتیاز نهایی")} · ${esc(document.querySelector(`#level [data-v="${level}"]`).textContent)}</div>
    <div class="grid grid-2" style="max-width:520px;margin:14px auto">
      <div><div class="score ltr">${you} / ${n}</div><div class="muted">${T("You", "شما")}</div></div>
      <div><div class="score ltr">${ai} / ${n}</div><div class="muted">${esc(opponent())}</div></div></div>
    <h2>${msg}</h2>
    ${leaderboard()}
    <p class="muted" style="max-width:640px;margin:0 auto 18px">${T("To be fair: the machine-learning models studied these exact molecules during training, so here they win about 99% of rounds at every level. On molecules they have never seen, their typical error is about 1 log unit (10×). Want a fair fight? Choose the classic formula as your opponent: it is right about 85% of the time.",
      "منصفانه بگوییم: مدل‌های یادگیری ماشین همین مولکول‌ها را هنگام آموزش دیده‌اند، پس این‌جا در هر سطحی حدود ۹۹٪ دورها را می‌برند. برای مولکول‌هایی که هرگز ندیده‌اند، خطای معمول آن‌ها حدود ۱ واحد لگاریتمی (۱۰ برابر) است. مسابقهٔ منصفانه می‌خواهید؟ فرمول کلاسیک را حریف خود کنید: حدود ۸۵٪ مواقع درست می‌گوید.")}</p>
    <div class="row" style="justify-content:center"><button class="btn" id="again">${T("Play again", "دوباره بازی کن")}</button>
      ${nextLevel && you >= n - 1 ? `<button class="btn ghost" id="harder">${T("Try a harder level", "سطح سخت‌تر را امتحان کن")}</button>` : ""}</div></div>`;
  document.getElementById("again").onclick = start;
  const h = document.getElementById("harder");
  if (h) h.onclick = () => { level = nextLevel; try { localStorage.setItem("soluai-quiz-level", level); } catch { /* */ } start(); };
}
