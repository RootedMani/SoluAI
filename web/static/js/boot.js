/* Runs before the page paints: applies the saved theme and language so nothing flashes.
   Language: "en" (default) or "fa" (Persian, right-to-left). `?lang=fa` in the address also works. */
(function () {
  var d = document.documentElement, t = null, l = null;
  try { t = localStorage.getItem("soluai-theme"); l = localStorage.getItem("soluai-lang"); } catch (e) { /* private mode */ }
  if (!t) t = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  var q = new URLSearchParams(location.search).get("lang");
  if (q === "fa" || q === "en") { l = q; try { localStorage.setItem("soluai-lang", l); } catch (e) { /* ignore */ } }
  if (!l) l = /^fa\b/i.test(navigator.language || "") ? "fa" : "en";
  if (l !== "fa") l = "en";
  d.dataset.theme = t; d.lang = l; d.dir = l === "fa" ? "rtl" : "ltr";
})();
