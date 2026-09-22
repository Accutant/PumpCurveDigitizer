"""Turn manually-traced pixel points (e.g. from same_color_tracer.py) into a
results.json compatible with catalogue.py, reusing digitize.py's exact fitting
and QA math so results stay consistent with the normal colour-mask pipeline.
"""
import json
import pathlib
import numpy as np
from digitize import axis_map, fit_poly, qa_curve, derived_fields  # noqa: F401 (HYD_CONST etc. live here)


def build_results(out_dir, meta, x_axis_ticks, curve_axes, curve_pts_px, degree=5, sg=1.0,
                   fit_deg_override=None):
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
    xf, _, xres = axis_map({"ticks": x_axis_ticks})
    res = {"meta": meta, "curves": {}, "flags": [], "calibration_resid_px": {"x": xres}}
    for name, pts in curve_pts_px.items():
        yticks = curve_axes[name]
        yf, _, yres = axis_map({"ticks": yticks})
        res["calibration_resid_px"][name] = yres
        xs_px = np.array([p[0] for p in pts], float)
        ys_px = np.array([p[1] for p in pts], float)
        x_val, y_val = xf(xs_px), yf(ys_px)
        deg = (fit_deg_override or {}).get(name, degree)
        c = fit_poly(x_val, y_val, deg)
        yspan = float(np.ptp(yf(np.array([t[0] for t in yticks]))))
        q = qa_curve(x_val, y_val, c, yspan)
        res["curves"][name] = {"coeffs": c.tolist(), "qa": q}
        if q["flags"]:
            res["flags"].extend(f"{name}: {fl}" for fl in q["flags"])
    d, flags = derived_fields(res, sg)
    res["derived"] = d
    res["flags"].extend(flags)
    res["status"] = "ok" if not res["flags"] else "review"
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(out_dir.name, "status:", res["status"])
    for name, cv in res["curves"].items():
        q = cv["qa"]
        print(f"  {name:6s} n={q.get('n_points')} R2={q.get('r2', 0):.5f} "
              f"rmse%={q.get('rmse_pct_span', 0):.2f} flags={q.get('flags')}")
    print("  derived:", json.dumps(d, indent=2))
    print("  flags:", res["flags"])
    return res
