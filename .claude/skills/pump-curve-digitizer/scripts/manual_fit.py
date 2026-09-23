"""Turn manually-traced pixel points (e.g. from same_color_tracer.py) into a
results.json compatible with catalogue.py, reusing digitize.py's exact fitting
and QA math so results stay consistent with the normal colour-mask pipeline.
"""
import json
import pathlib
import numpy as np
from digitize import (axis_map, fit_poly, qa_curve, derived_fields, bep_box_check,  # noqa: F401
                      hard_failures, peval, draw_overlay, fit_plot, FLOW, UNITS)


def build_results(out_dir, meta, x_axis_ticks, curve_axes, curve_pts_px, degree=5, sg=1.0,
                   fit_deg_override=None, sheet_png=None, frame=None,
                   x_unit="bpd", y_units=None):
    """
    out_dir: folder to write results.json into (created if missing)
    meta: dict, same shape as sheet.json's "meta"
    x_axis_ticks: [[px, value], ...] for the shared x axis
    curve_axes: {"head": [[px,value],...], "power": [...], "eff": [...]} -- the y
        axis ticks for each curve (from the sheet, or hand-read off a zoomed crop)
    curve_pts_px: {"head": [(x_px,y_px), ...], "power": [...], "eff": [...]} -- the
        traced pixel points per curve
    degree / sg: same meaning as sheet.json's fit.degree / fit.sg
    fit_deg_override: optional {"eff": 6} to fit one curve at a different degree
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_units = y_units or {}
    xf, xinv, xres = axis_map({"ticks": x_axis_ticks}, "x_axis")
    xk = FLOW[x_unit.lower()]
    res = {"meta": meta, "source": "manual_fit (traced points)", "curves": {}, "flags": [],
           "calibration_resid_px": {"x": xres}}
    curves_px, fits_px, data = {}, {}, {}
    for name, pts in curve_pts_px.items():
        yticks = curve_axes[name]
        yf, yinv, yres = axis_map({"ticks": yticks}, f"y_axes.{name}")
        res["calibration_resid_px"][name] = yres
        yk = UNITS[name][y_units.get(name, {"head": "ft", "power": "hp", "eff": "%"}[name]).lower()]
        xs_px = np.array([p[0] for p in pts], float)
        ys_px = np.array([p[1] for p in pts], float)
        x_val, y_val = xf(xs_px) * xk, yf(ys_px) * yk
        deg = (fit_deg_override or {}).get(name, degree)
        c = fit_poly(x_val, y_val, deg)
        # same span convention as digitize.py: the full plotted axis, not just the tick range
        if frame:
            yspan = abs(float(yf(frame[1])) - float(yf(frame[3]))) * yk
        else:
            yspan = float(np.ptp(yf(np.array([t[0] for t in yticks])))) * yk
        q = qa_curve(x_val, y_val, c, yspan)
        res["curves"][name] = {"coeffs": c.tolist(), "units": {
            "x": "bpd", "y": {"head": "ft/stage", "power": "hp/stage", "eff": "%"}[name]}, "qa": q}
        if q["flags"]:
            res["flags"].extend(f"{name}: {fl}" for fl in q["flags"])
        np.savetxt(out_dir / f"points_{name}.csv", np.stack([x_val, y_val], 1), delimiter=",",
                   header=f"flow_bpd,{name}", comments="", fmt="%.6g")
        curves_px[name] = (xs_px, ys_px)
        data[name] = (x_val, y_val)
        xg = np.linspace(x_val.min(), x_val.max(), 300)
        fits_px[name] = np.stack([xinv(xg / xk), yinv(peval(c, xg) / yk)], 1)
    d, flags = derived_fields(res, sg)
    bep_err, bep_flags = bep_box_check(res, meta)
    d.update(bep_err)
    res["derived"] = d
    res["flags"].extend(flags + bep_flags)
    res["hard_failures"] = hard_failures(res, d, meta)
    res["flags"].extend(f"REJECT: {b}" for b in res["hard_failures"])
    res["status"] = "reject" if res["hard_failures"] else ("review" if res["flags"] else "ok")
    if sheet_png and frame:
        import cv2
        img = cv2.imread(str(sheet_png))
        if img is not None:
            draw_overlay(img, frame, curves_px, fits_px, out_dir / "overlay.png")
            fit_plot(res, data, out_dir / "fit_plot.png", sg)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(out_dir.name, "status:", res["status"])
    for name, cv in res["curves"].items():
        q = cv["qa"]
        print(f"  {name:6s} n={q.get('n_points')} R2={q.get('r2', 0):.5f} "
              f"rmse%={q.get('rmse_pct_span', 0):.2f} flags={q.get('flags')}")
    print("  derived:", json.dumps(d, indent=2))
    print("  flags:", res["flags"])
    return res
