"""Rewrite catalogue coefficients at full precision from the run folders.

    python refresh_from_runs.py CATALOGUE.csv --runs runs runs_bhi ... [--dry-run]

Use when the coefficient columns have lost precision -- most often because the CSV was
opened and saved in a spreadsheet. c5 is ~1e-19 and is multiplied by Q^5, so dropping a few
digits moves the fitted curve by percent, not by rounding error: a 12-significant-digit
round-trip took a 151-row catalogue from 143 clean rows to 22.

Each row is matched to its run by (manufacturer, model) and its head/bhp/eff coefficients and
the values derived from them (shutoff head, BEP efficiency, BEP BHP) are rewritten from that
run's results.json. Every other column -- ROR, limits, notes, anything typed by hand -- is
left exactly as it is. A timestamped .bak is written first.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import glob
import json
import shutil
from pathlib import Path

import numpy as np

CURVE_PREFIX = {"head": "head", "power": "bhp", "eff": "eff"}


def peval(c, x):
    return float(np.polynomial.polynomial.polyval(x, c))


def load_runs(dirs):
    runs = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "*" / "results.json")):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            k = (str(r.get("meta", {}).get("manufacturer")).strip().lower(),
                 str(r.get("meta", {}).get("model")).strip().lower())
            runs.setdefault(k, []).append((f, r))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cat = Path(a.catalogue)
    with cat.open(newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols, rows = list(rd.fieldnames), list(rd)
    runs = load_runs(a.runs)

    fixed = untouched = missing = 0
    for row in rows:
        k = (str(row.get("manufacturer") or "").strip().lower(),
             str(row.get("model") or "").strip().lower())
        if k not in runs:
            missing += 1
            continue
        _, res = sorted(runs[k])[-1]
        changed = False
        for name, pre in CURVE_PREFIX.items():
            c = (res.get("curves", {}).get(name) or {}).get("coeffs")
            if not c:
                continue
            for i, v in enumerate(c):
                col = f"{pre}_c{i}"
                new = f"{float(v):.15g}"
                if col in row and row[col].strip() != new:
                    row[col], changed = new, True
        if changed:      # keep the derived columns consistent with the restored coefficients
            head = (res["curves"].get("head") or {}).get("coeffs")
            eff = (res["curves"].get("eff") or {}).get("coeffs")
            power = (res["curves"].get("power") or {}).get("coeffs")
            try:
                bep = float(row.get("bep_bpd") or 0) or None
            except ValueError:
                bep = None
            if head:
                row["shutoff_head_ft_per_stage"] = f"{peval(head, 0.0):.4f}"
            if eff and bep:
                row["bep_eff_pct"] = f"{peval(eff, bep):.3f}"
            if power and bep:
                row["bep_bhp_hp_per_stage"] = f"{peval(power, bep):.5f}"
            fixed += 1
            print(f"refreshed {row['manufacturer']} {row['model']} from {Path(_).parent.name}")
        else:
            untouched += 1
    print(f"\n{fixed} rows refreshed, {untouched} already at full precision, "
          f"{missing} with no run folder (left alone)")
    if a.dry_run or not fixed:
        return
    shutil.copy2(cat, cat.with_suffix(f".{dt.datetime.now():%Y%m%d-%H%M%S}.bak"))
    with cat.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {cat} (backup alongside it)")


if __name__ == "__main__":
    main()
