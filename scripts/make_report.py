#!/usr/bin/env python3
"""Render exported PIBE results as a self-contained HTML report.

Reads one or more JSON files written by ``export_results.py`` and emits a
single HTML page with the data inlined, so the page needs no network access.

Usage::

    python scripts/make_report.py --results a.json b.json --out report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = """<title>PIBE Estimation Dossier</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  .viz-root {
    color-scheme: light;
    /* Neutrals carry a slight cool bias toward the blue truth series. */
    --surface-0: #eef1f5;
    --surface-1: #fbfcfe;
    --surface-2: #e4e9f0;
    --border:    #d3dae4;
    --text-primary:   #10151c;
    --text-secondary: #4a5464;
    --text-muted:     #75808f;
    --truth:   #2a78d6;
    --est:     #eb6834;
    --meas:    #1baf7a;
    --good:    #1a7f4b;
    --bad:     #c0392b;
    --grid:    #dde3ec;
    --serif: "IBM Plex Serif", Georgia, serif;
    --sans:  "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    --mono:  "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-0: #0d1016;
      --surface-1: #161b23;
      --surface-2: #1e242e;
      --border:    #2b323d;
      --text-primary:   #f2f5f9;
      --text-secondary: #b3bcc9;
      --text-muted:     #8593a3;
      --truth:  #3987e5;
      --est:    #d95926;
      --meas:   #199e70;
      --good:   #45b97c;
      --bad:    #e66767;
      --grid:   #262d38;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-0: #0d1016;
    --surface-1: #161b23;
    --surface-2: #1e242e;
    --border:    #2b323d;
    --text-primary:   #f2f5f9;
    --text-secondary: #b3bcc9;
    --text-muted:     #8593a3;
    --truth:  #3987e5;
    --est:    #d95926;
    --meas:   #199e70;
    --good:   #45b97c;
    --bad:    #e66767;
    --grid:   #262d38;
  }

  .viz-root {
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.55;
    padding: 34px 20px 72px;
    min-height: 100vh;
    box-sizing: border-box;
  }
  .wrap { max-width: 1060px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; }

  .masthead { border-bottom: 2px solid var(--text-primary); padding-bottom: 16px; margin-bottom: 8px; }
  .eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.13em; text-transform: uppercase;
             color: var(--text-muted); margin: 0 0 8px; }
  h1 { font-family: var(--serif); font-size: 30px; font-weight: 600; margin: 0 0 6px;
       letter-spacing: -0.015em; text-wrap: balance; }
  .sub { color: var(--text-secondary); margin: 0; font-size: 13.5px; max-width: 68ch; }
  .eqn { font-family: var(--mono); font-size: 11.5px; line-height: 1.75; color: var(--text-secondary);
         background: var(--surface-2); border-radius: 8px; padding: 12px 14px; margin: 14px 0 0;
         overflow-x: auto; white-space: pre; }

  h2 { font-family: var(--serif); font-size: 17px; font-weight: 600; margin: 30px 0 2px; letter-spacing: -0.01em; }
  .note { color: var(--text-muted); font-size: 12.5px; margin: 0 0 12px; max-width: 74ch; }
  code, .num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
  code { font-size: 0.9em; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 10px; margin: 18px 0 6px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .tile .k { font-family: var(--mono); font-size: 10.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.09em; }
  .tile .v { font-family: var(--mono); font-size: 20px; font-weight: 500; margin-top: 4px; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
  .tile .d { font-size: 11.5px; color: var(--text-secondary); margin-top: 3px; }

  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 14px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px 8px; }
  .card h3 { font-size: 13px; margin: 0 0 2px; font-weight: 600; letter-spacing: -0.005em; }
  .card .cap { font-family: var(--mono); font-size: 11px; color: var(--text-muted); margin: 0 0 8px; font-variant-numeric: tabular-nums; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .axis text { font-family: var(--mono); font-size: 10px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
  .axis line, .axis path { stroke: var(--grid); stroke-width: 1; }
  .serieslabel { font-family: var(--mono); font-size: 10.5px; font-weight: 500; }

  .legend { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin: 2px 0 8px; font-size: 12px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 13px; height: 3px; border-radius: 2px; display: inline-block; }
  .swatch.dash { background: none !important; border-top: 3px dashed currentColor; height: 0; }

  .tblwrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--text-muted); font-weight: 500; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; }
  .pill { display: inline-block; padding: 1px 9px; border-radius: 999px; font-family: var(--mono); font-size: 11px; font-weight: 500; }
  .pill.good { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }
  .pill.bad  { background: color-mix(in srgb, var(--bad) 16%, transparent);  color: var(--bad); }

  .tip { position: fixed; font-family: var(--mono); pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
         border-radius: 7px; padding: 7px 9px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.18);
         opacity: 0; transition: opacity .08s; z-index: 30; font-variant-numeric: tabular-nums; }
  .tip b { display: block; margin-bottom: 3px; color: var(--text-muted); font-weight: 600; font-size: 11px; }
  .tip .row { display: flex; gap: 8px; justify-content: space-between; }
  .tip .row i { font-style: normal; color: var(--text-secondary); }
  details { margin-top: 10px; }
  summary { cursor: pointer; font-size: 12.5px; color: var(--text-secondary); }
</style>

<div class="viz-root"><div class="wrap" id="app"></div></div>
<div class="tip" id="tip"></div>

<script>
const RESULTS = __DATA__;

const fmt = (v, n = 4) => {
  if (v === null || v === undefined || !isFinite(v)) return "n/a";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(n - 2);
  return v.toFixed(n);
};
const el = (tag, attrs = {}, kids = []) => {
  const NS = "http://www.w3.org/2000/svg";
  const svgTags = new Set(["svg","g","path","line","circle","rect","text","polyline"]);
  const node = svgTags.has(tag) ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.setAttribute("class", v);
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) if (kid) node.appendChild(kid);
  return node;
};

const tip = document.getElementById("tip");
function showTip(evt, title, rows) {
  tip.innerHTML = "<b>" + title + "</b>" + rows.map(
    r => '<div class="row"><i style="color:' + r.c + '">' + r.k + '</i><span>' + r.v + "</span></div>"
  ).join("");
  const pad = 14;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  const box = tip.getBoundingClientRect();
  if (x + box.width > innerWidth - 8) x = evt.clientX - box.width - pad;
  if (y + box.height > innerHeight - 8) y = evt.clientY - box.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px"; tip.style.opacity = 1;
}
const hideTip = () => { tip.style.opacity = 0; };

/* ---------- line chart: several series over a shared time grid ---------- */
function lineChart({ t, series, height = 168, ylabel = "", unit = "" }) {
  const W = 560, H = height, M = { t: 14, r: 64, b: 26, l: 46 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const xs = t, x0 = xs[0], x1 = xs[xs.length - 1];
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const X = v => M.l + ((v - x0) / (x1 - x0)) * iw;
  const Y = v => M.t + ih - ((v - lo) / (hi - lo)) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const axis = el("g", { class: "axis" });
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (i / ticks) * (hi - lo), y = Y(v);
    axis.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y }));
    axis.appendChild(el("text", { x: M.l - 7, y: y + 3.5, "text-anchor": "end", text: fmt(v, 2) }));
  }
  for (let i = 0; i <= 4; i++) {
    const v = x0 + (i / 4) * (x1 - x0);
    axis.appendChild(el("text", { x: X(v), y: H - 8, "text-anchor": "middle", text: fmt(v, 1) }));
  }
  axis.appendChild(el("text", { x: M.l + iw / 2, y: H + 6, "text-anchor": "middle", text: "" }));
  svg.appendChild(axis);
  if (ylabel) svg.appendChild(el("text", {
    class: "axis", x: 4, y: M.t - 3, "text-anchor": "start", text: ylabel,
    style: "font-size:10.5px;fill:var(--text-muted)"
  }));

  for (const s of series) {
    const d = s.values.map((v, i) => (i ? "L" : "M") + X(xs[i]).toFixed(2) + " " + Y(v).toFixed(2)).join(" ");
    svg.appendChild(el("path", {
      d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(s.dash ? { "stroke-dasharray": s.dash } : {})
    }));
    // Direct label at the right edge: identity is never colour-alone.
    const last = s.values[s.values.length - 1];
    svg.appendChild(el("text", {
      class: "serieslabel", x: M.l + iw + 7, y: Y(last) + 3.5, fill: s.color, text: s.label
    }));
  }

  const cross = el("line", { y1: M.t, y2: M.t + ih, stroke: "var(--text-muted)", "stroke-width": 1, opacity: 0 });
  svg.appendChild(cross);
  const dots = series.map(s => {
    const c = el("circle", { r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2, opacity: 0 });
    svg.appendChild(c); return c;
  });
  const hit = el("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
  svg.appendChild(hit);
  hit.addEventListener("pointermove", evt => {
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * W;
    const frac = Math.max(0, Math.min(1, (px - M.l) / iw));
    let i = Math.round(frac * (xs.length - 1));
    cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("opacity", 0.5);
    series.forEach((s, k) => {
      dots[k].setAttribute("cx", X(xs[i])); dots[k].setAttribute("cy", Y(s.values[i])); dots[k].setAttribute("opacity", 1);
    });
    showTip(evt, "t = " + fmt(xs[i], 2) + " s",
      series.map(s => ({ k: s.label, v: fmt(s.values[i], 4) + unit, c: s.color })));
  });
  hit.addEventListener("pointerleave", () => {
    hideTip(); cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0));
  });
  return svg;
}

/* ---------- paired dot plot for the parameters ---------- */
function dumbbell({ names, truth, est, bounds }) {
  const W = 560, rowH = 40, H = names.length * rowH + 40, M = { t: 22, r: 66, b: 18, l: 54 };
  const iw = W - M.l - M.r;
  let lo = Math.min(...truth, ...est, ...bounds.map(b => b[0]));
  let hi = Math.max(...truth, ...est, ...bounds.map(b => b[1]));
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const X = v => M.l + ((v - lo) / (hi - lo)) * iw;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  const axis = el("g", { class: "axis" });
  for (let i = 0; i <= 4; i++) {
    const v = lo + (i / 4) * (hi - lo);
    axis.appendChild(el("line", { x1: X(v), x2: X(v), y1: M.t - 8, y2: H - M.b }));
    axis.appendChild(el("text", { x: X(v), y: H - 5, "text-anchor": "middle", text: fmt(v, 2) }));
  }
  svg.appendChild(axis);

  names.forEach((name, i) => {
    const y = M.t + i * rowH + rowH / 2 - 6;
    svg.appendChild(el("text", { class: "axis", x: 6, y: y + 4, text: name, style: "font-size:12px;fill:var(--text-secondary)" }));
    // admissible interval as a recessive rule
    svg.appendChild(el("line", {
      x1: X(bounds[i][0]), x2: X(bounds[i][1]), y1: y, y2: y,
      stroke: "var(--grid)", "stroke-width": 6, "stroke-linecap": "round"
    }));
    svg.appendChild(el("line", {
      x1: X(truth[i]), x2: X(est[i]), y1: y, y2: y,
      stroke: "var(--text-muted)", "stroke-width": 2, opacity: 0.55
    }));
    const mk = (v, color, label) => {
      const c = el("circle", { cx: X(v), cy: y, r: 6, fill: color, stroke: "var(--surface-1)", "stroke-width": 2 });
      c.addEventListener("pointerenter", e => showTip(e, name, [{ k: label, v: fmt(v, 5), c: color }]));
      c.addEventListener("pointerleave", hideTip);
      svg.appendChild(c);
    };
    mk(truth[i], "var(--truth)", "true");
    mk(est[i], "var(--est)", "estimated");
    svg.appendChild(el("text", {
      class: "serieslabel", x: W - M.r + 10, y: y + 4,
      fill: Math.abs(est[i] - truth[i]) < 0.02 ? "var(--good)" : "var(--est)",
      text: (est[i] - truth[i] >= 0 ? "+" : "") + fmt(est[i] - truth[i], 3)
    }));
  });
  svg.appendChild(el("text", {
    class: "axis", x: W - M.r + 10, y: 14, fill: "var(--text-muted)",
    text: "error", style: "font-size:10.5px"
  }));
  return svg;
}

function card(title, caption, node, legend) {
  const c = el("div", { class: "card" }, [el("h3", { text: title })]);
  if (caption) c.appendChild(el("p", { class: "cap", text: caption }));
  if (legend) c.appendChild(legend);
  c.appendChild(node);
  return c;
}
function legendOf(items) {
  return el("div", { class: "legend" }, items.map(
    it => el("span", {}, [
      el("i", { class: "swatch" + (it.dash ? " dash" : ""), style: "background:" + it.c + ";color:" + it.c }),
      el("span", { text: it.k })
    ])
  ));
}

/* ---------- page ---------- */
function render() {
  const app = document.getElementById("app");
  app.innerHTML = "";
  const R = RESULTS[0];

  const head = el("div", { class: "masthead" });
  head.appendChild(el("p", { class: "eyebrow", text:
    "Physics-Informed Bank of Estimators  ·  " + R.system + "  ·  held-out trajectories" }));
  head.appendChild(el("h1", { text: "Estimation vs Truth" }));
  head.appendChild(el("p", { class: "sub", html:
    "Joint recovery of the unmeasured states x₂–x₄, the constant parameters θ, and the "
    + "time-varying disturbance d(t), from the single noisy output y = x₁ + ω. "
    + "Horizon T = " + R.horizon + " s, measurement noise σ = " + R.noise_sigma + ", schedule: "
    + R.schedule + "." }));
  head.appendChild(el("pre", { class: "eqn", text: [
      "ẋ₁ = x₂ − 0.35 tanh x₁ + θ₁(0.8 + 0.2 cos x₁)",
      "ẋ₂ = x₃ − 0.30 tanh x₂ + 0.10 sin x₁ + θ₂(0.9 + 0.1 sin x₁ + 0.1 cos x₂)",
      "ẋ₃ = x₄ − 0.25 tanh x₃ + 0.10 sin(x₁+x₂) + θ₃(1 + 0.1 sin(x₁+x₃))",
      "ẋ₄ = −[0.576  2.736  4.76  3.6]·x + 0.15 sin x₁ + 0.08 tanh(x₂x₃)",
      "       + 0.08θ₁ sin x₂ + 0.06θ₂ sin x₃ + 0.05θ₃ tanh x₄ + d(t)",
      "y  = x₁ + ω        Γ₄(t) = [sin Ωt  cos Ωt  sin 2Ωt  cos 2Ωt]ᵀ,  Ω = 2π/5,  W_Γ = 10·I₄"
    ].join(String.fromCharCode(10)) }));
  app.appendChild(head);

  const m = R.metrics;
  const thetaErr = Math.max(...m.theta_abs_error);
  const dErr = m.disturbance_sup;
  const tiles = el("div", { class: "tiles" });
  const tile = (k, v, d) => tiles.appendChild(el("div", { class: "tile" }, [
    el("div", { class: "k", text: k }), el("div", { class: "v", text: v }), el("div", { class: "d", text: d })
  ]));
  tile("state error x₁", fmt(m.state_l2[0], 4), "normalised L², noise σ = " + R.noise_sigma);
  tile("worst state error", fmt(Math.max(...m.state_l2), 4), "over x₁..x" + "₄".slice(0,1));
  tile("worst |θ error|", fmt(thetaErr, 4), "true |θ| ≤ 0.25");
  tile("disturbance sup err", fmt(dErr, 4), "‖d‖∞ ≈ 0.23");
  app.appendChild(tiles);

  const verdictGood = !R.oracle.truth_scores_better;
  app.appendChild(el("p", { class: "note", html:
    "<span class='pill " + (verdictGood ? "good" : "bad") + "'>" +
    (verdictGood ? "objective reached" : "objective not reached") + "</span> &nbsp;" +
    "L<sub>Tot</sub> at the true solution = <b>" + fmt(R.oracle.total_truth, 4) + "</b>; " +
    "at the trained bank = <b>" + fmt(R.oracle.total_trained, 4) + "</b>. " +
    (verdictGood
      ? "Training reached the objective value the exact solution attains, so any residual error is at the noise floor rather than an optimisation gap."
      : "The exact solution scores better on the very objective being minimised, so the gap is optimisation, not loss design.") }));

  /* states */
  app.appendChild(el("h2", { text: "State trajectories" }));
  app.appendChild(el("p", { class: "note", text:
    "One held-out trajectory. Only x₁ is measured (and only through noisy y); "
    + "x₂..x₄ are never observed — they are reconstructed by the cascade." }));
  const traj = R.trajectories[0];
  const g = el("div", { class: "grid2" });
  for (let j = 0; j < R.n_states; j++) {
    const series = [
      { label: "true", values: traj.true[j], color: "var(--truth)" },
      { label: "est", values: traj.est[j], color: "var(--est)", dash: "5 3" },
    ];
    if (j === 0) series.unshift({ label: "y (noisy)", values: traj.y, color: "var(--meas)", width: 1, opacity: .5 });
    const err = Math.sqrt(traj.true[j].reduce((a, v, i) => a + (v - traj.est[j][i]) ** 2, 0) / traj.true[j].length);
    g.appendChild(card(
      "x" + ["₁","₂","₃","₄"][j] + (j === 0 ? "  (measured)" : "  (unmeasured)"),
      "RMSE " + fmt(err, 4),
      lineChart({ t: R.t, series, ylabel: "x" + ["₁","₂","₃","₄"][j] }),
      legendOf(j === 0
        ? [{ k: "y (noisy)", c: "var(--meas)" }, { k: "true", c: "var(--truth)" }, { k: "estimated", c: "var(--est)", dash: 1 }]
        : [{ k: "true", c: "var(--truth)" }, { k: "estimated", c: "var(--est)", dash: 1 }])
    ));
  }
  app.appendChild(g);

  /* disturbance */
  app.appendChild(el("h2", { text: "Disturbance" }));
  app.appendChild(el("p", { class: "note", html:
    "d̂(t) = Γ₄(t)ᵀ â against the true d(t). ε<sub>d,q</sub> = " + fmt(R.eps_dq, 3) +
    (R.eps_dq > 0 ? " — part of d lies outside the basis span, so exact recovery is impossible."
                  : " — d lies exactly in the basis span, so exact recovery is possible.") }));
  app.appendChild(card("d(t)", "sup error " + fmt(R.metrics.disturbance_sup, 4),
    lineChart({
      t: R.t, height: 190,
      series: [
        { label: "true", values: R.disturbance.true, color: "var(--truth)" },
        { label: "est", values: R.disturbance.est, color: "var(--est)", dash: "5 3" },
      ], ylabel: "d"
    }),
    legendOf([{ k: "true d", c: "var(--truth)" }, { k: "estimated d̂", c: "var(--est)", dash: 1 }])));

  /* parameters */
  app.appendChild(el("h2", { text: "Parameters" }));
  app.appendChild(el("p", { class: "note", text:
    "Constant unknowns. The grey rule is the admissible set Θⱼ the head is constrained to; "
    + "an estimate sitting on its end means the tanh reparameterisation saturated." }));
  app.appendChild(card("θ recovery", "",
    dumbbell({
      names: ["θ₁", "θ₂", "θ₃"],
      truth: R.theta.true, est: R.theta.est, bounds: R.theta.bounds
    }),
    legendOf([{ k: "true", c: "var(--truth)" }, { k: "estimated", c: "var(--est)" }])));

  app.appendChild(card("basis coefficients â", "",
    dumbbell({
      names: R.coefficients.true.map((_, i) => "a" + (i + 1)),
      truth: R.coefficients.true, est: R.coefficients.est, bounds: R.coefficients.bounds
    }),
    legendOf([{ k: "true", c: "var(--truth)" }, { k: "estimated", c: "var(--est)" }])));

  /* tables */
  app.appendChild(el("h2", { text: "Numbers" }));
  const rows = [];
  for (let j = 0; j < R.n_states; j++)
    rows.push(["x" + (j + 1), fmt(m.state_l2[j], 4), fmt(m.state_sup[j], 4), ""]);
  for (let j = 0; j < R.theta.true.length; j++)
    rows.push(["θ" + (j + 1), fmt(R.theta.true[j], 5), fmt(R.theta.est[j], 5), fmt(R.theta.est[j] - R.theta.true[j], 5)]);
  rows.push(["d", fmt(m.disturbance_l2, 4), fmt(m.disturbance_sup, 4), ""]);
  const tbl = el("table", {}, [el("thead", {}, [el("tr", {}, [
    el("th", { text: "quantity" }), el("th", { text: "L² / true" }),
    el("th", { text: "sup / estimated" }), el("th", { text: "error" })
  ])])]);
  tbl.appendChild(el("tbody", {}, rows.map(r => el("tr", {}, r.map(c => el("td", { text: c }))))));
  app.appendChild(el("div", { class: "tblwrap" }, [tbl]));

  const oc = el("table", {}, [el("thead", {}, [el("tr", {}, [
    el("th", { text: "cell" }), el("th", { text: "L_Loc @ truth" }),
    el("th", { text: "L_Loc @ trained" }), el("th", { text: "" })
  ])])]);
  oc.appendChild(el("tbody", {}, Object.entries(R.oracle.per_cell).map(([k, v]) =>
    el("tr", {}, [
      el("td", { text: "cell " + k }), el("td", { text: fmt(v.truth, 4) }), el("td", { text: fmt(v.trained, 4) }),
      el("td", {}, [el("span", { class: "pill " + (v.trained <= v.truth * 1.5 ? "good" : "bad"),
        text: v.trained <= v.truth * 1.5 ? "matched" : "gap" })])
    ]))));
  const det = el("details", {}, [el("summary", { text: "Oracle comparison — the objective at the exact solution" })]);
  det.appendChild(el("div", { class: "tblwrap" }, [oc]));
  app.appendChild(det);
}
render();
</script>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = [json.loads(path.read_text()) for path in args.results]
    html = TEMPLATE.replace("__DATA__", json.dumps(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
