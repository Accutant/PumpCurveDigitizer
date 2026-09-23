"""Step 3 of the pump-curve pipeline: write fitted coefficients into the catalogue CSV.

Usage:
    python catalogue.py CATALOGUE.csv WORKDIR/results.json [WORKDIR2/results.json ...]
                        [--accept-review] [--log digitize_log.csv] [--dry-run]

* Keeps the catalogue's existing column order (ESPPump_Coefficients template); unknown
  columns are left blank, c6..c9 are blank for 5th-degree fits.
* Upserts on (manufacturer, model, base_frequency_hz): an existing row for the same pump
  is replaced, everything else is untouched. A timestamped .bak copy is written first.
* Sheets with status "review" are skipped unless --accept-review is given (after a human
  has looked at overlay.png / fit_plot.png).
* Sheets with status "reject" are NEVER written: --accept-review does not override them.
  A reject means a check no sign-off should waive failed (hydraulic error > 10 pts, power
  negative in range, impossible efficiency, BEP outside the sheet's own ROR, printed-BEP
  mismatch > 10%). Fix sheet.json and re-run digitize.py instead.
* If the same pump is already in the log with a BETTER QA score, the worse re-run is skipped
  unless --force: duplicate scans of one sheet must not silently replace a cleaner run.
* A side log (default: digitize_log.csv next to the catalogue) records QA metrics and the
  work folder for every row written, so any coefficient can be traced back to its sheet.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import shutil
from pathlib import Path

import numpy as np

from common import load_json

DEFAULT_COLUMNS = (
    "manufacturer,model,base_frequency_hz,catalog_source,catalog_version,series,pump_od_in,"
    "stage_type,construction,rpm,stages_basis,ror_min_bpd,bep_bpd,ror_max_bpd,flow_min_bpd,"
    "flow_max_bpd,shutoff_head_ft_per_stage,bep_eff_pct,bep_bhp_hp_per_stage,shaft_limit_hp,"
    "housing_burst_psi,freq_min_hz,freq_max_hz,freq_step_hz,"
    + ",".join(f"{p}_c{i}" for p in ("head", "bhp", "eff") for i in range(10)) + ",notes").split(",")
CURVE_PREFIX = {"head": "head", "power": "bhp", "eff": "eff"}


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.15g}"
    return str(v)


def peval(c, x):
    return float(np.polynomial.polynomial.polyval(x, c))


def row_from_results(res, workdir):
    m = dict(res.get("meta", {}))
    d = res.get("derived", {})
    cur = res["curves"]
    row = {k: m.get(k) for k in DEFAULT_COLUMNS if k in m}
    bep = m.get("bep_bpd") or d.get("bep_bpd_fit")
    row["bep_bpd"] = round(bep, 1) if bep is not None else None
    row["flow_min_bpd"] = m.get("flow_min_bpd", 0)
    row["flow_max_bpd"] = round(m.get("flow_max_bpd") or d.get("flow_max_bpd"), 1) if (m.get("flow_max_bpd") or d.get("flow_max_bpd")) else None
    if cur.get("head", {}).get("coeffs"):
        row["shutoff_head_ft_per_stage"] = round(peval(cur["head"]["coeffs"], 0.0), 4)
    if bep is not None and cur.get("eff", {}).get("coeffs"):
        row["bep_eff_pct"] = round(peval(cur["eff"]["coeffs"], bep), 3)
    if bep is not None and cur.get("power", {}).get("coeffs"):
        row["bep_bhp_hp_per_stage"] = round(peval(cur["power"]["coeffs"], bep), 5)
    for name, pre in CURVE_PREFIX.items():
        c = cur.get(name, {}).get("coeffs")
        if c:
            for i, v in enumerate(c):
                row[f"{pre}_c{i}"] = float(v)
    today = dt.date.today().isoformat()
    src = m.get("catalog_source") or "curve sheet"
    row["catalog_source"] = f"{src}; digitized {today}"
    notes = [m.get("notes") or ""]
    missing = [n for n in CURVE_PREFIX if not cur.get(n, {}).get("coeffs")]
    if missing:
        notes.append("no " + "/".join(missing) + " curve")
    if res["status"] != "ok":
        notes.append("QA review accepted: " + "; ".join(res["flags"]))
    notes.append(f"work folder {Path(workdir).name}")
    row["notes"] = " | ".join(n for n in notes if n)
    return row


def log_row(res, workdir):
    out = {"date": dt.date.today().isoformat(), "manufacturer": res["meta"].get("manufacturer"),
           "model": res["meta"].get("model"), "base_frequency_hz": res["meta"].get("base_frequency_hz"),
           "status": res["status"], "workdir": str(Path(workdir).resolve()),
           "hydraulic_check_max_diff_pts": res.get("derived", {}).get("hydraulic_check_max_diff_pts"),
           "qa_score": round(qa_score(res), 3),
           "flags": "; ".join(res["flags"])}
    for n, v in res["curves"].items():
        q = v.get("qa", {})
        out[f"{n}_n"] = q.get("n_points")
        out[f"{n}_r2"] = q.get("r2")
        out[f"{n}_max_resid_pct_span"] = q.get("max_resid_pct_span")
    return out


def qa_score(res):
    """Lower is better: worst of the hydraulic error (pts) and the per-curve max residual
    (% of axis span). Used to decide which of two scans of the same pump to keep."""
    worst = 0.0
    hyd = (res.get("derived") or {}).get("hydraulic_check_max_diff_pts")
    if hyd is not None:
        worst = max(worst, float(hyd))
    for v in res.get("curves", {}).values():
        r = (v.get("qa") or {}).get("max_resid_pct_span")
        if r is not None:
            worst = max(worst, float(r))
    return worst


def previous_scores(logp):
    """{(mfr, model, hz): best qa score seen} from the digitize log, for duplicate handling."""
    out = {}
    if not Path(logp).exists():
        return out
    with Path(logp).open(newline="") as f:
        for row in csv.DictReader(f):
            k = (str(row.get("manufacturer") or "").strip().lower(),
                 str(row.get("model") or "").strip().lower(),
                 str(row.get("base_frequency_hz") or "").strip().rstrip("0").rstrip("."))
            try:
                sc = max([float(row.get("hydraulic_check_max_diff_pts") or 0)]
                         + [float(row.get(f"{c}_max_resid_pct_span") or 0)
                            for c in ("head", "power", "eff")])
            except ValueError:
                continue
            out[k] = min(out.get(k, sc), sc)
    return out


def key(r):
    return (str(r.get("manufacturer") or "").strip().lower(), str(r.get("model") or "").strip().lower(),
            str(r.get("base_frequency_hz") or "").strip().rstrip("0").rstrip("."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalogue")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--accept-review", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="write even if a previous scan of the same pump scored better")
    a = ap.parse_args()

    cat = Path(a.catalogue)
    if cat.exists():
        with cat.open(newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            cols = list(rd.fieldnames)
            rows = list(rd)
    else:
        cols, rows = DEFAULT_COLUMNS, []

    logp = Path(a.log) if a.log else cat.with_name("digitize_log.csv")
    prev = previous_scores(logp)

    new_rows, logs = [], []
    for rp in a.results:
        res = load_json(rp)
        wd = Path(rp).parent
        label = f"{res['meta'].get('manufacturer')} {res['meta'].get('model')}"
        if res.get("status") == "reject" or res.get("hard_failures"):
            print(f"REJECT {label}: {'; '.join(res.get('hard_failures') or res['flags'])}")
            print("       not written -- fix sheet.json and re-run digitize.py")
            continue
        k = key({"manufacturer": res["meta"].get("manufacturer"), "model": res["meta"].get("model"),
                 "base_frequency_hz": res["meta"].get("base_frequency_hz")})
        sc, old = qa_score(res), prev.get(k)
        if old is not None and sc > old + 0.5 and not a.force:
            print(f"SKIP  {label}: an earlier scan scored better (QA {old:.1f} vs {sc:.1f}); --force to overwrite")
            continue
        if res["status"] != "ok" and not a.accept_review:
            print(f"SKIP  {label}: status={res['status']} ({'; '.join(res['flags'])})")
            continue
        if not res["meta"].get("manufacturer") or not res["meta"].get("model"):
            print(f"SKIP  {label}: manufacturer/model missing in meta")
            continue
        new_rows.append(row_from_results(res, wd))
        logs.append(log_row(res, wd))

    for nr in new_rows:
        k = key(nr)
        hit = [i for i, r in enumerate(rows) if key(r) == k]
        rec = {c: fmt(nr.get(c)) for c in cols}
        if hit:
            rows[hit[0]] = rec
            print(f"UPDATE {nr['manufacturer']} {nr['model']} {nr.get('base_frequency_hz')} Hz")
        else:
            rows.append(rec)
            print(f"ADD    {nr['manufacturer']} {nr['model']} {nr.get('base_frequency_hz')} Hz")

    if a.dry_run or not new_rows:
        return
    if cat.exists():
        shutil.copy2(cat, cat.with_suffix(f".{dt.datetime.now():%Y%m%d-%H%M%S}.bak"))
    with cat.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    exists = logp.exists()
    fields = sorted({k for l in logs for k in l}, key=lambda k: (k not in ("date", "manufacturer", "model"), k))
    if exists:
        with logp.open(newline="") as f:
            old = list(csv.DictReader(f))
        old_fields = list(old[0].keys()) if old else []
        fields = old_fields + [k for k in fields if k not in old_fields]
        logs = old + logs
    with logp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(logs)
    print(f"wrote {cat} ({len(rows)} rows) and {logp}")


if __name__ == "__main__":
    main()
