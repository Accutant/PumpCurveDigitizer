"""Whole-catalogue audit: re-check every stored row, independent of how it was produced.

    python audit_catalogue.py ESPPump_Coefficients.csv [--runs runs runs_bhi ...] [--csv audit.csv]

Per-row checks (the coefficients are evaluated exactly as a consuming program would):
  * hydraulic   -- Q*H*SG/(135770*BHP) vs the stored efficiency curve over 25-85% of flow_max
  * power sign  -- BHP must stay positive across the operating range
  * efficiency  -- peak within 20-90%, and not rising at the run-out end
  * head        -- must fall from shutoff to flow_max (a rising head curve means a bad fit)
  * BEP vs ROR  -- stored BEP inside the sheet's own operating range
  * frequency   -- rpm must match base_frequency_hz (~58.3 rpm per Hz); metric-sourced rows
                   filed as 60 Hz are called out, since metric sheets are usually 50 Hz
  * derived     -- shutoff head equals head(0); stored BEP efficiency equals the curve there
  * family      -- head/stage vs flow against the whole-catalogue trend; a row several times
                   off the trend is usually an axis read at the wrong scale (this is how a
                   power axis printed 0-7 hp instead of 0-35 hp was found)
  * precision   -- coefficients must keep ~15 significant digits. Opening the catalogue in a
                   spreadsheet and saving it rounds them, which silently bends every curve:
                   c5 is ~1e-19 and is multiplied by Q^5, so dropped digits move the fit by
                   percent, not by rounding error. Edit the CSV with a text editor or a
                   script, never with Excel.
  * drift       -- with --runs, the row must still match the results.json it came from

Exit code 1 if any row fails, so it can gate a commit. Run it after every batch.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

HYD_CONST = 135770.0
LIM = {"hyd_pts": 5.0, "eff_min": 20.0, "eff_max": 90.0, "family_sigma": 3.0, "drift_pct": 1.0}


def coeffs(row, prefix, n=10):
    c = []
    for i in range(n):
        v = row.get(f"{prefix}_c{i}", "")
        c.append(float(v) if str(v).strip() not in ("", "nan") else 0.0)
    return np.array(c) if any(c) else None


def val(row, name):
    v = row.get(name, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def peval(c, x):
    return np.polynomial.polynomial.polyval(x, c)


def precision_check(row):
    """Most significant digits kept by any coefficient in the row.

    A file written by the pipeline has at least one coefficient carrying ~15 digits; after a
    spreadsheet round-trip none of them do. The maximum is the right statistic: individual
    coefficients can be legitimately short (0.75), so the minimum gives false alarms.
    """
    best = 0
    for p in ("head", "bhp", "eff"):
        for i in range(10):
            t = str(row.get(f"{p}_c{i}", "")).strip()
            if not t or t in ("0", "nan"):
                continue
            digits = len(t.split("E")[0].split("e")[0].replace("-", "").replace(".", "").lstrip("0"))
            best = max(best, digits)
    return best or None


def audit_row(row, sg=1.0):
    issues = []
    sig = precision_check(row)
    if sig is not None and sig < 14:
        issues.append(f"coefficients stored with only {sig} significant digits -- the catalogue "
                      f"was probably opened and saved in a spreadsheet; restore it from git or "
                      f"re-write it from the run folders")
    head, bhp, eff = (coeffs(row, p) for p in ("head", "bhp", "eff"))
    qmax, bep = val(row, "flow_max_bpd"), val(row, "bep_bpd")
    if head is None:
        return ["no head coefficients"]
    if not qmax or qmax <= 0:
        return ["flow_max_bpd missing or <= 0"]
    q = np.linspace(0.05 * qmax, qmax, 300)
    h = peval(head, q)
    sh = val(row, "shutoff_head_ft_per_stage")
    if sh is not None and abs(peval(head, 0.0) - sh) > max(0.02, 0.002 * abs(sh)):
        issues.append(f"shutoff_head {sh:.2f} != head(0) {peval(head, 0.0):.2f}")
    if h.min() < -0.5:
        issues.append(f"head reaches {h.min():.1f} ft inside the range")
    if h[-1] > h[0]:
        issues.append("head curve rises from shutoff to flow_max")
    if bhp is not None:
        p = peval(bhp, np.linspace(0.2 * qmax, 0.9 * qmax, 60))
        if p.min() <= 0:
            issues.append(f"power reaches {p.min():.3g} hp/stage in the operating range")
    if eff is not None:
        e = peval(eff, q)
        if not (LIM["eff_min"] <= e.max() <= LIM["eff_max"]):
            issues.append(f"peak efficiency {e.max():.1f}%")
        if bep is not None:
            stored = val(row, "bep_eff_pct")
            got = peval(eff, bep)
            if stored is not None and abs(got - stored) > 0.5:
                issues.append(f"bep_eff_pct {stored:.1f} != eff(BEP) {got:.1f}")
    if eff is not None and bhp is not None:
        qq = np.linspace(0.25 * qmax, 0.85 * qmax, 40)
        pw = peval(bhp, qq)
        ok = pw > 0
        if ok.sum() > 5:
            d = 100 * qq[ok] * peval(head, qq[ok]) * sg / (HYD_CONST * pw[ok]) - peval(eff, qq[ok])
            if np.abs(d).max() > LIM["hyd_pts"]:
                issues.append(f"hydraulic check off by up to {np.abs(d).max():.1f} pts")
    hz, rpm = val(row, "base_frequency_hz"), val(row, "rpm")
    if hz and rpm and abs(rpm / 58.33 - hz) > 0.1 * hz:
        issues.append(f"{rpm:.0f} rpm implies {rpm / 58.33:.0f} Hz but base_frequency_hz is {hz:.0f}")
    if hz and not rpm:
        issues.append("rpm blank: cannot confirm the stated frequency")
    if "m3/d" in str(row.get("notes", "")).lower() and hz == 60:
        issues.append("notes mention a metric sheet but the row is filed as 60 Hz (usually 50)")
    lo, hi = val(row, "ror_min_bpd"), val(row, "ror_max_bpd")
    if lo and hi and bep and not (lo * 0.95 <= bep <= hi * 1.05):
        issues.append(f"BEP {bep:.0f} outside ROR {lo:.0f}-{hi:.0f}")
    return issues


def family_check(rows, results):
    """Head per stage scales smoothly with flow across a catalogue; flag the outliers."""
    q, h, idx = [], [], []
    for i, r in enumerate(rows):
        c, b = coeffs(r, "head"), val(r, "bep_bpd")
        if c is None or not b or b <= 0:
            continue
        hv = peval(c, b)
        if hv <= 0:
            continue
        q.append(np.log10(b)); h.append(np.log10(hv)); idx.append(i)
    if len(q) < 20:
        return
    q, h = np.array(q), np.array(h)
    fit = np.polyfit(q, h, 1)
    dev = h - np.polyval(fit, q)
    sd = 1.4826 * np.median(np.abs(dev - np.median(dev)))
    for k, i in enumerate(idx):
        if abs(dev[k] - np.median(dev)) > LIM["family_sigma"] * sd:
            results[i].append(f"head/stage {10 ** h[k]:.0f} ft at {10 ** q[k]:.0f} bpd is "
                              f"{abs(dev[k] - np.median(dev)) / sd:.1f} sigma off the catalogue trend "
                              f"(check the axis scales on this sheet)")


def drift_check(rows, results, run_dirs):
    runs = {}
    for d in run_dirs:
        for f in glob.glob(str(Path(d) / "*" / "results.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            k = (str(r.get("meta", {}).get("manufacturer")).lower(),
                 str(r.get("meta", {}).get("model")).lower())
            runs.setdefault(k, []).append((f, r))
    for i, row in enumerate(rows):
        k = (str(row.get("manufacturer") or "").lower(), str(row.get("model") or "").lower())
        if k not in runs:
            results[i].append("no run folder found for this row")
            continue
        _, rr = sorted(runs[k])[-1]
        qmax = val(row, "flow_max_bpd") or 1
        xs = np.linspace(1, qmax, 100)
        for cur, pre in (("head", "head"), ("power", "bhp"), ("eff", "eff")):
            cat, run = coeffs(row, pre), rr["curves"].get(cur, {}).get("coeffs")
            if cat is None or not run:
                continue
            a, b = peval(cat, xs), peval(np.array(run + [0] * (10 - len(run))), xs)
            scale = max(np.abs(b).max(), 1e-9)
            if np.abs(a - b).max() / scale > LIM["drift_pct"] / 100:
                results[i].append(f"{cur} coefficients differ from {Path(_).parent.name} by "
                                  f"{100 * np.abs(a - b).max() / scale:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue")
    ap.add_argument("--runs", nargs="*", default=[], help="run folders, for the drift check")
    ap.add_argument("--sg", type=float, default=1.0)
    ap.add_argument("--csv", help="write the findings to this CSV as well")
    a = ap.parse_args()
    with open(a.catalogue, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    results = [audit_row(r, a.sg) for r in rows]
    family_check(rows, results)
    if a.runs:
        drift_check(rows, results, a.runs)
    bad = [(r, iss) for r, iss in zip(rows, results) if iss]
    for r, iss in bad:
        print(f"{r.get('manufacturer','?')} {r.get('model','?')} {r.get('base_frequency_hz','?')} Hz")
        for i in iss:
            print(f"    - {i}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} rows clean, {len(bad)} with findings")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["manufacturer", "model", "base_frequency_hz", "finding"])
            for r, iss in bad:
                for i in iss:
                    w.writerow([r.get("manufacturer"), r.get("model"), r.get("base_frequency_hz"), i])
        print("findings written to", a.csv)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
