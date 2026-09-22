"""Batch runner: inspect + digitize every sheet in a folder with one layout template.

Usage:
    python batch.py "SHEETS/*.pdf" --template templates/reda.json --out runs/ [--dpi 200]

For each sheet (each page of a multi-page PDF is its own sheet) it writes a work folder
runs/<name>[_pN]/ with sheet.png, review.png, sheet.json, overlay.png, fit_plot.png and
results.json, then a runs/batch_summary.csv listing status, flags and QA per sheet.

Per-sheet catalogue fields that differ between sheets (model, ROR, BEP, limits) are NOT
guessed: they stay null in sheet.json and must be filled from the sheet text before the
results go into the catalogue (re-run digitize.py after editing sheet.json).
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
from pathlib import Path

from inspect_sheet import run_inspect
from digitize import run as run_digitize


def pages_of(path):
    if Path(path).suffix.lower() != ".pdf":
        return [0]
    import pymupdf
    with pymupdf.open(path) as d:
        return list(range(d.page_count))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(a.pattern)):
        pages = pages_of(f)
        for p in pages:
            name = Path(f).stem + (f"_p{p + 1}" if len(pages) > 1 else "")
            wd = out / name
            row = {"sheet": f, "page": p + 1, "workdir": str(wd)}
            try:
                draft, _ = run_inspect(f, wd, p, a.dpi, a.template)
                missing = draft.get("_template_missing")
                cfg = {k: v for k, v in draft.items() if not k.startswith("_")}
                (wd / "sheet.json").write_text(json.dumps(cfg, indent=2))
                if missing:
                    row.update(status="needs-axes", flags=f"template roles not found: {missing}")
                else:
                    res = run_digitize(wd / "sheet.json")
                    row.update(status=res["status"], flags="; ".join(res["flags"]))
                    for k, v in res["curves"].items():
                        q = v.get("qa", {})
                        row[f"{k}_r2"] = q.get("r2")
                        row[f"{k}_max_resid_pct"] = q.get("max_resid_pct_span")
                    row["hyd_check_pts"] = res.get("derived", {}).get("hydraulic_check_max_diff_pts")
            except Exception as e:                      # keep going; report the failure
                row.update(status="error", flags=repr(e))
            rows.append(row)
            print(f"{row['status']:<10} {name}  {row.get('flags', '')[:120]}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "batch_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{n_ok}/{len(rows)} ok -> {out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
