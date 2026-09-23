"""Step 2 of the pump-curve pipeline: extract curves, fit polynomials, run QA.

Usage:
    python digitize.py WORKDIR/sheet.json [--out WORKDIR]

sheet.json (see SKILL.md for the full schema):
    image          rasterized sheet written by inspect_sheet.py
    frame          [x0, y0, x1, y1] plot area in pixels (search region)
    exclude_boxes  list of [x0, y0, x1, y1] regions to ignore (legend boxes, labels)
    x_axis         {"ticks": [[px, value], ...], "unit": "bpd" | "m3d"}
    y_axes         {"head": {...,"unit": "ft"|"m"}, "power": {...,"unit": "hp"|"kw"},
                    "eff": {...,"unit": "%"|"fraction"}}   ticks are [px, value] with px = y pixel
    curves         {"head": {"color": "#0000ff", "tol": 35, "style": "solid"|"dashed",
                             "x_range": [lo, hi] (optional, catalogue units)}, "power": ..., "eff": ...}
    fit            {"degree": 5, "sg": 1.0}
    meta           catalogue fields read from the sheet text (manufacturer, model, rpm, ...)

Writes to WORKDIR:
    points_<curve>.csv   extracted points in catalogue units (bpd, ft, hp, %)
    overlay.png          fitted curves drawn over the faded original sheet
    fit_plot.png         data vs fit, residuals, and the hydraulic-efficiency cross-check
    results.json         coefficients (C0..C5, raw units, ascending powers), QA metrics, status
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.polynomial import Polynomial

from common import load_json, save_json, line_mask, detect_lines

# conversions to catalogue units
FLOW = {"bpd": 1.0, "bbl/d": 1.0, "m3d": 6.289811, "m3/d": 6.289811}
HEAD = {"ft": 1.0, "m": 3.280840}
POWER = {"hp": 1.0, "kw": 1.341022}
EFF = {"%": 1.0, "pct": 1.0, "fraction": 100.0}
UNITS = {"head": HEAD, "power": POWER, "eff": EFF}
HYD_CONST = 135770.0          # Q[bpd] * H[ft] * SG / 135770 = hydraulic hp

# QA thresholds (fraction of the y-axis span unless stated)
QA = {"rmse_frac": 0.006, "max_resid_frac": 0.025, "max_gap_frac": 0.12,
      "min_points": 40,
      # hydraulic cross-check: flag above hyd_eff_pts, hard-reject above hyd_reject_pts
      "hyd_eff_pts": 5.0, "hyd_reject_pts": 10.0,
      # printed-BEP cross-check (meta.bep_head_ft / bep_bhp_hp / bep_eff_pct)
      "bep_box_pct": 3.0, "bep_box_reject_pct": 10.0,
      # physically impossible results -> reject
      "eff_min_pct": 20.0, "eff_max_pct": 90.0,
      # rpm implies the supply frequency: a 2-pole motor turns ~58.3 rpm per Hz
      "rpm_per_hz": 58.33, "rpm_hz_tol_pct": 10.0}


# ----------------------------------------------------------------------------- calibration
def axis_map(spec, name="axis"):
    """Least-squares linear pixel->value map from >= 2 [px, value] ticks.

    Validated: hand-entered ticks are a known source of silent, catastrophic errors
    (two values on one pixel, or a flow value pasted into a power axis). Those produce
    a plausible-looking fit and wrong-but-smooth coefficients, so they are refused here.
    """
    t = np.asarray(spec["ticks"], float)
    if len(t) < 2:
        raise ValueError(f"{name}: needs >= 2 ticks, got {len(t)}")
    order = np.argsort(t[:, 0])
    t = t[order]
    dpx, dval = np.diff(t[:, 0]), np.diff(t[:, 1])
    if np.any(np.abs(dpx) < 0.5):
        raise ValueError(f"{name}: two ticks share a pixel position -- {spec['ticks']}")
    if np.any(dval == 0):
        raise ValueError(f"{name}: repeated tick value -- {spec['ticks']}")
    if not (np.all(dval > 0) or np.all(dval < 0)):
        raise ValueError(f"{name}: tick values are not monotonic in pixel order -- {spec['ticks']}")
    b, a = np.polyfit(t[:, 0], t[:, 1], 1)
    resid = t[:, 1] - (a + b * t[:, 0])
    return (lambda p: a + b * np.asarray(p, float)), (lambda v: (np.asarray(v, float) - a) / b), \
        float(np.abs(resid / b).max())


# ----------------------------------------------------------------------------- extraction
def hex_to_bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.uint8)


def color_mask(img, color, tol):
    """Pixels within CIE-Lab distance `tol` of the target colour."""
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    tgt = cv2.cvtColor((hex_to_bgr(color).reshape(1, 1, 3).astype(np.float32) / 255.0), cv2.COLOR_BGR2LAB)[0, 0]
    return (np.linalg.norm(lab - tgt, axis=2) < tol).astype(np.uint8)


def curve_mask(img, frame, excl, spec):
    x0, y0, x1, y1 = frame
    sub = img[y0:y1 + 1, x0:x1 + 1]
    color = spec["color"]
    dark = color.lower() in ("#000000", "black")
    tol = spec.get("tol", 40 if dark else 35)
    m = color_mask(sub, "#000000" if dark else color, tol)
    fh, fw = m.shape
    # drop straight grid/marker lines of the same colour (span >= 25% of the frame)
    kv = max(25, int(0.25 * fh))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    m[cv2.dilate(vert, np.ones((1, 3), np.uint8)) > 0] = 0
    if dark:
        kh = max(25, int(0.25 * fw))
        hor = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
        m[cv2.dilate(hor, np.ones((3, 1), np.uint8)) > 0] = 0
        # A dark curve shares its colour with the grid. Morphology only removes grid lines
        # that stay unbroken; on sheets where the grid is chopped up (shaded bands, crossings)
        # the leftovers are indistinguishable from curve pixels and the tracker can follow a
        # grid line instead of the curve. So erase the grid where it was actually detected.
        lines = detect_lines(sub, min_frac=0.3)
        for L in lines["h"]:                      # erase a grid line only where the mask really
            r = int(round(L["pos"]))              # runs the width of the plot: a curve touches a
            if m[r, :].mean() > 0.5:              # row only over a limited span, a grid line does not
                m[r, :] = 0
        for L in lines["v"]:
            c_ = int(round(L["pos"]))
            if m[:, c_].mean() > 0.5:
                m[:, c_] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for bx0, by0, bx1, by1 in excl:
        m[max(0, by0 - y0):max(0, by1 - y0 + 1), max(0, bx0 - x0):max(0, bx1 - x0 + 1)] = 0
    # drop specks and text: components too small for a curve piece
    min_len = spec.get("min_len", 4 if spec.get("style") == "dashed" else 15)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= min_len
    return keep[lab].astype(np.uint8)


def column_candidates(m, max_run):
    """Per column: centres of vertical runs (each run = one crossing of a curve)."""
    cands = {}
    for x in range(m.shape[1]):
        col = m[:, x]
        idx = np.flatnonzero(col)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        c = [r.mean() for r in runs if len(r) <= max_run]
        if c:
            cands[x] = c
    return cands


def robust_track(cands, deg=7, iters=4):
    """Pick one y per column: iterate a smooth pixel-space fit and keep the nearest candidate."""
    xs = np.array(sorted(cands))
    if len(xs) < 10:
        return np.array([]), np.array([])
    single = np.array([len(cands[x]) == 1 for x in xs])
    px = xs[single] if single.sum() >= 10 else xs
    py = np.array([cands[x][0] if len(cands[x]) == 1 else np.median(cands[x]) for x in px])
    thr = None
    for _ in range(iters):
        keep = np.ones(len(px), bool)
        for _ in range(3):
            p = Polynomial.fit(px[keep], py[keep], min(deg, keep.sum() - 1))
            r = np.abs(py - p(px))
            mad = 1.4826 * np.median(r[keep])
            thr = max(2.5, 3.5 * mad)
            keep = r <= thr
        # re-select candidates in every column using the current fit
        sel_x, sel_y = [], []
        for x in xs:
            c = np.array(cands[x])
            j = int(np.argmin(np.abs(c - p(x))))
            if abs(c[j] - p(x)) <= max(thr * 2, 6):
                sel_x.append(x); sel_y.append(c[j])
        px, py = np.array(sel_x, float), np.array(sel_y, float)
    r = np.abs(py - p(px))
    ok = r <= max(thr, 2.5)
    return px[ok], py[ok]


# ----------------------------------------------------------------------------- fitting / QA
def fit_poly(x, y, deg):
    """Scaled-domain least squares, converted back to raw-unit coefficients C0..Cdeg."""
    p = Polynomial.fit(x, y, deg).convert()
    c = np.zeros(deg + 1)
    c[:len(p.coef)] = p.coef
    return c


def peval(c, x):
    return np.polynomial.polynomial.polyval(x, c)


def qa_curve(x, y, c, yspan):
    f = peval(c, x)
    res = y - f
    ss = ((y - y.mean()) ** 2).sum()
    xs = np.sort(x)
    gap = np.diff(xs).max() / (xs[-1] - xs[0]) if len(xs) > 2 else 1.0
    q = {"n_points": int(len(x)), "x_min": float(x.min()), "x_max": float(x.max()),
         "r2": float(1 - (res ** 2).sum() / ss) if ss > 0 else 1.0,
         "rmse": float(np.sqrt((res ** 2).mean())), "max_abs_resid": float(np.abs(res).max()),
         "rmse_pct_span": float(100 * np.sqrt((res ** 2).mean()) / yspan),
         "max_resid_pct_span": float(100 * np.abs(res).max() / yspan),
         "largest_gap_pct_range": float(100 * gap)}
    flags = []
    if len(x) < QA["min_points"]:
        flags.append(f"only {len(x)} points")
    if q["rmse_pct_span"] > 100 * QA["rmse_frac"]:
        flags.append(f"rmse {q['rmse_pct_span']:.2f}% of axis span")
    if q["max_resid_pct_span"] > 100 * QA["max_resid_frac"]:
        flags.append(f"max residual {q['max_resid_pct_span']:.2f}% of axis span")
    if gap > QA["max_gap_frac"]:
        flags.append(f"gap of {100 * gap:.0f}% of flow range in extracted points")
    # raw-unit coefficients are what the consuming program evaluates; make sure converting
    # from the scaled domain didn't lose precision to cancellation (worse at high degree/flow)
    xs_ = np.linspace(x.min(), x.max(), 200)
    scaled = Polynomial.fit(x, y, len(c) - 1)
    q["roundtrip_err_pct_span"] = float(100 * np.abs(peval(c, xs_) - scaled(xs_)).max() / yspan)
    if q["roundtrip_err_pct_span"] > 0.01:
        flags.append(f"raw-coefficient round-trip error {q['roundtrip_err_pct_span']:.3f}% of span")
    # would another degree do better? (catalogue has columns up to c9)
    scan = {}
    for d in range(3, 10):
        if len(x) > d + 5:
            pd_ = Polynomial.fit(x, y, d)
            scan[d] = float(100 * np.abs(y - pd_(x)).max() / yspan)
    q["degree_scan_max_resid_pct_span"] = scan
    if flags and any(k.startswith("max residual") or k.startswith("rmse") for k in flags):
        ok = [d for d, v in scan.items() if v <= 100 * QA["max_resid_frac"]]
        if ok:
            flags.append(f"degree {min(ok)} would meet the residual limit "
                         f"({scan[min(ok)]:.2f}% of span) -- extraction looks clean, 5th-degree form is limiting"
                         if len(c) - 1 < min(ok) else "")
    q["flags"] = [f for f in flags if f]
    return q


def derived_fields(res, sg):
    """Catalogue fields computed from the fits (shutoff head, BEP, flow max) + physics check."""
    d, flags = {}, []
    cur = {k: v for k, v in res["curves"].items() if v.get("coeffs")}
    if "head" in cur:
        ch, qh = np.array(cur["head"]["coeffs"]), cur["head"]["qa"]
        d["shutoff_head_ft_per_stage"] = float(peval(ch, 0.0))
        # flow max: last root of head in/just beyond the data, else the data end
        xs = np.linspace(qh["x_min"], qh["x_max"] * 1.05, 2000)
        hv = peval(ch, xs)
        neg = np.flatnonzero(hv <= 0)
        d["flow_max_bpd"] = float(xs[neg[0]]) if neg.size else float(qh["x_max"])
        d["flow_min_bpd"] = 0.0
    if "eff" in cur:
        ce, qe = np.array(cur["eff"]["coeffs"]), cur["eff"]["qa"]
        xs = np.linspace(qe["x_min"], qe["x_max"], 4000)
        i = int(np.argmax(peval(ce, xs)))
        d["bep_bpd_fit"] = float(xs[i])
        d["bep_eff_pct"] = float(peval(ce, xs[i]))
        if "power" in cur:
            d["bep_bhp_hp_per_stage"] = float(peval(np.array(cur["power"]["coeffs"]), xs[i]))
    if all(k in cur for k in ("head", "power", "eff")):
        ch, cp, ce = (np.array(cur[k]["coeffs"]) for k in ("head", "power", "eff"))
        qmax = d.get("flow_max_bpd", cur["head"]["qa"]["x_max"])
        lo = max(cur[k]["qa"]["x_min"] for k in cur)
        hi = min(cur[k]["qa"]["x_max"] for k in cur)
        xs = np.linspace(max(lo, 0.15 * qmax), min(hi, 0.9 * qmax), 60)
        eff_calc = 100 * xs * peval(ch, xs) * sg / (HYD_CONST * peval(cp, xs))
        diff = eff_calc - peval(ce, xs)
        d["hydraulic_check_max_diff_pts"] = float(np.abs(diff).max())
        d["hydraulic_check_mean_diff_pts"] = float(diff.mean())
        if np.abs(diff).max() > QA["hyd_eff_pts"]:
            flags.append(f"hydraulic check: Q*H/(135770*BHP) differs from digitized efficiency by up to "
                         f"{np.abs(diff).max():.1f} pts (axis assignment/units/SG?)")
    return d, flags


def bep_box_check(res, meta):
    """Cross-check the fits against a BEP box printed on the sheet (REDA prints Q/H/P/E).

    Set meta.bep_bpd plus any of bep_head_ft, bep_bhp_hp, bep_eff_pct from the sheet and the
    fitted curves are compared there. This is independent of every pixel step, so it catches
    a mis-scaled or mis-read axis that the fit statistics cannot see.
    """
    q = meta.get("bep_bpd")
    printed = {"head": meta.get("bep_head_ft"), "power": meta.get("bep_bhp_hp"),
               "eff": meta.get("bep_eff_pct")}
    if not q or not any(v for v in printed.values()):
        return {}, []
    out, flags = {}, []
    for name, want in printed.items():
        cv = res["curves"].get(name, {})
        if not want or not cv.get("coeffs"):
            continue
        got = float(peval(np.array(cv["coeffs"]), float(q)))
        err = 100 * (got - float(want)) / float(want) if want else float("nan")
        out[f"bep_{name}_err_pct"] = err
        if abs(err) > QA["bep_box_pct"]:
            flags.append(f"printed BEP {name} is {want:g}, fit gives {got:.4g} ({err:+.1f}%)")
    return out, flags


def frequency_check(cfg):
    """Frequency vs units and rpm.

    Metric sheets (m3/d, m, kW) are usually 50 Hz and imperial ones 60 Hz, and a metric sheet
    filed as 60 Hz is a silent error: the coefficients are fine but every affinity-law
    correction downstream is wrong. rpm settles it -- ~3500 rpm is 60 Hz, ~2917 rpm is 50 Hz.
    """
    meta = cfg.get("meta", {})
    hz, rpm = meta.get("base_frequency_hz"), meta.get("rpm")
    units = [cfg.get("x_axis", {}).get("unit", "")] + \
            [v.get("unit", "") for v in cfg.get("y_axes", {}).values()]
    metric = any(str(u).lower() in ("m3d", "m3/d", "m", "kw") for u in units)
    flags = []
    if hz is None:
        flags.append("base_frequency_hz not set -- read it from the sheet, do not assume 60 Hz")
    if metric and hz == 60:
        flags.append("metric axes (m3/d, m or kW) with base_frequency_hz = 60: metric sheets are "
                     "usually 50 Hz -- confirm against the sheet's stated Hz/rpm")
    if hz and rpm:
        implied = float(rpm) / QA["rpm_per_hz"]
        if abs(implied - float(hz)) > QA["rpm_hz_tol_pct"] / 100 * float(hz):
            flags.append(f"{rpm:g} rpm implies {implied:.0f} Hz but base_frequency_hz is {hz:g}")
    return flags


def hard_failures(res, d, meta):
    """Errors no human sign-off should be able to wave through (status 'reject').

    Each of these was observed producing a wrong catalogue row that looked fine in the
    overlay and had R2 > 0.999.
    """
    bad = []
    hyd = d.get("hydraulic_check_max_diff_pts")
    if hyd is not None and hyd > QA["hyd_reject_pts"]:
        bad.append(f"hydraulic check off by {hyd:.0f} pts (limit {QA['hyd_reject_pts']:.0f})")
    cur = {k: v for k, v in res["curves"].items() if v.get("coeffs")}
    qmax = d.get("flow_max_bpd")
    if qmax and "power" in cur:
        xs = np.linspace(0.2 * qmax, 0.9 * qmax, 60)
        pw = peval(np.array(cur["power"]["coeffs"]), xs)
        if pw.min() <= 0:
            bad.append(f"power curve reaches {pw.min():.3g} hp/stage inside the operating range")
    if "eff" in cur:
        e = d.get("bep_eff_pct")
        if e is not None and not (QA["eff_min_pct"] <= e <= QA["eff_max_pct"]):
            bad.append(f"peak efficiency {e:.1f}% outside {QA['eff_min_pct']:g}-{QA['eff_max_pct']:g}%")
    lo, hi = meta.get("ror_min_bpd"), meta.get("ror_max_bpd")
    bep = meta.get("bep_bpd") or d.get("bep_bpd_fit")
    if lo and hi and bep and not (float(lo) * 0.95 <= float(bep) <= float(hi) * 1.05):
        bad.append(f"BEP {float(bep):.0f} bpd falls outside the sheet's ROR {lo:g}-{hi:g} bpd")
    for name, err in ((k, v) for k, v in d.items() if k.endswith("_err_pct")):
        if abs(err) > QA["bep_box_reject_pct"]:
            bad.append(f"{name.replace('_err_pct','').replace('bep_','printed BEP ')} off by {err:+.0f}%")
    return bad


# ----------------------------------------------------------------------------- outputs
COLORS_BGR = {"head": (200, 0, 0), "power": (0, 0, 200), "eff": (0, 140, 0)}


def draw_overlay(img, frame, curves_px, fits_px, path):
    faded = cv2.addWeighted(img, 0.35, np.full_like(img, 255), 0.65, 0)
    x0, y0, x1, y1 = frame
    cv2.rectangle(faded, (x0, y0), (x1, y1), (200, 0, 200), 1)
    for name, (px, py) in curves_px.items():
        for x, y in zip(px, py):
            cv2.circle(faded, (int(round(x)), int(round(y))), 1, COLORS_BGR.get(name, (0, 0, 0)), -1)
    for name, pts in fits_px.items():
        cv2.polylines(faded, [pts.astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(faded, name, tuple(int(v) for v in pts[len(pts) // 2]) , cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), faded)


def fit_plot(res, data, path, sg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in ("head", "power", "eff") if k in data]
    fig, ax = plt.subplots(2, len(names), figsize=(4.2 * len(names), 6), squeeze=False,
                           gridspec_kw={"height_ratios": [3, 1.2]})
    units = {"head": "ft/stage", "power": "hp/stage", "eff": "%"}
    for j, k in enumerate(names):
        x, y = data[k]
        c = np.array(res["curves"][k]["coeffs"])
        xs = np.linspace(x.min(), x.max(), 400)
        ax[0, j].plot(x, y, ".", ms=2, color="0.5", label="extracted")
        ax[0, j].plot(xs, peval(c, xs), "k-", lw=1.2, label="5th-deg fit")
        if k == "eff" and all(n in data for n in ("head", "power")):
            ch, cp = (np.array(res["curves"][n]["coeffs"]) for n in ("head", "power"))
            with np.errstate(divide="ignore", invalid="ignore"):
                ec = 100 * xs * peval(ch, xs) * sg / (HYD_CONST * peval(cp, xs))
            ax[0, j].plot(xs, ec, "r--", lw=1, label="Q·H/(135770·BHP)")
        q = res["curves"][k]["qa"]
        ax[0, j].set_title(f"{k}  R²={q['r2']:.5f}  max|e|={q['max_resid_pct_span']:.2f}% span", fontsize=9)
        ax[0, j].set_ylabel(units[k]); ax[0, j].legend(fontsize=7); ax[0, j].grid(alpha=.3)
        ax[1, j].plot(x, y - peval(c, x), ".", ms=2, color="0.3")
        ax[1, j].axhline(0, color="k", lw=.8); ax[1, j].set_xlabel("flow, bpd"); ax[1, j].grid(alpha=.3)
        ax[1, j].set_ylabel("residual")
    fig.suptitle(f"{res['meta'].get('manufacturer')} {res['meta'].get('model')}  — status: {res['status']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(cfg_path, out=None):
    cfg = load_json(cfg_path)
    out = Path(out or Path(cfg_path).parent)
    img = cv2.imread(str(Path(cfg_path).parent / cfg["image"]) if not Path(cfg["image"]).is_absolute()
                     else cfg["image"])
    if img is None:
        img = cv2.imread(cfg["image"])
    frame = cfg["frame"]
    x0, y0, x1, y1 = frame
    excl = cfg.get("exclude_boxes", [])
    deg = cfg.get("fit", {}).get("degree", 5)
    sg = cfg.get("fit", {}).get("sg", 1.0)

    xf, xinv, xres = axis_map(cfg["x_axis"], "x_axis")
    xk = FLOW[cfg["x_axis"].get("unit", "bpd").lower()]
    res = {"config": str(cfg_path), "meta": cfg.get("meta", {}), "curves": {}, "flags": [],
           "calibration_resid_px": {"x": xres}}
    curves_px, fits_px, data = {}, {}, {}
    for name, spec in cfg["curves"].items():
        yspec = cfg["y_axes"][spec.get("axis", name)]
        yf, yinv, yres = axis_map(yspec, f"y_axes.{spec.get('axis', name)}")
        res["calibration_resid_px"][name] = yres
        yk = UNITS[name][yspec.get("unit").lower()]
        m = curve_mask(img, frame, excl, spec)
        max_run = spec.get("max_run", int(0.3 * (y1 - y0)))
        cands = column_candidates(m, max_run=max_run)
        px, py = robust_track(cands)
        if len(px) < 10:
            res["curves"][name] = {"coeffs": None, "qa": {"flags": ["curve not found"]}}
            res["flags"].append(f"{name}: curve not found (colour/tol?)")
            continue
        px, py = px + x0, py + y0
        x = xf(px) * xk
        y = yf(py) * yk
        if "x_range" in spec:
            lo, hi = spec["x_range"]
            sel = (x >= lo) & (x <= hi)
            x, y, px, py = x[sel], y[sel], px[sel], py[sel]
        c = fit_poly(x, y, deg)
        yspan = abs(float(yf(y0)) - float(yf(y1))) * yk      # full plotted span of this axis
        q = qa_curve(x, y, c, yspan)
        res["curves"][name] = {"coeffs": [float(v) for v in c], "units": {
            "x": "bpd", "y": {"head": "ft/stage", "power": "hp/stage", "eff": "%"}[name]}, "qa": q}
        res["flags"] += [f"{name}: {f}" for f in q["flags"]]
        curves_px[name] = (px, py)
        xs = np.linspace(x.min(), x.max(), 300)
        fits_px[name] = np.stack([xinv(xs / xk), yinv(peval(c, xs) / yk)], 1)
        data[name] = (x, y)
        np.savetxt(out / f"points_{name}.csv", np.stack([x, y], 1), delimiter=",",
                   header=f"flow_bpd,{name}", comments="", fmt="%.6g")
    d, flags = derived_fields(res, sg)
    bep_err, bep_flags = bep_box_check(res, cfg.get("meta", {}))
    d.update(bep_err)
    res["derived"] = d
    res["flags"] += flags + bep_flags + frequency_check(cfg)
    if max(res["calibration_resid_px"].values()) > 3:
        res["flags"].append("axis calibration residual > 3 px: check tick assignments")
    res["hard_failures"] = hard_failures(res, d, cfg.get("meta", {}))
    res["flags"] += [f"REJECT: {b}" for b in res["hard_failures"]]
    res["status"] = "reject" if res["hard_failures"] else ("review" if res["flags"] else "ok")
    draw_overlay(img, frame, curves_px, fits_px, out / "overlay.png")
    fit_plot(res, data, out / "fit_plot.png", sg)
    save_json(res, out / "results.json")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = run(a.config, a.out)
    print(f"status: {r['status']}")
    for k, v in r["curves"].items():
        q = v["qa"]
        if v["coeffs"] is None:
            print(f"  {k:<5} NOT FOUND"); continue
        print(f"  {k:<5} n={q['n_points']:<4} x={q['x_min']:.0f}..{q['x_max']:.0f}  R2={q['r2']:.5f}  "
              f"rmse={q['rmse_pct_span']:.2f}%  max={q['max_resid_pct_span']:.2f}%  gap={q['largest_gap_pct_range']:.0f}%")
    for k, v in r["derived"].items():
        print(f"  {k} = {v:.4g}")
    for f in r["flags"]:
        print("  FLAG:", f)
    if r.get("hard_failures"):
        print("  -> status REJECT: not writable to the catalogue. Fix the sheet.json "
              "(axis ticks/units) and re-run; --accept-review cannot override this.")


if __name__ == "__main__":
    main()
