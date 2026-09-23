"""Regression test: synthetic vector PDF with known curves, scored at several DPIs.

    python tests/regression.py [--out /tmp/pcd_regression] [--samples DIR --map map.json]

Pass criteria per DPI: max fit error vs truth < 0.5 % of axis span for head and power,
< 1.5 % for efficiency, and status not "reject". Run after any change to the scripts.

The limits are accuracy limits, not QA limits. The exact error depends on the render scale
the sheet normalises to, which shifts by a pixel or two with the local Tesseract/font build,
and the efficiency curve is the most sensitive because of its steep run-out: measured 0.26 %
to 1.06 % of span across environments for the same code. A "review" status is expected here
too -- the synthetic efficiency curve is sharper than a 5th-degree polynomial can follow, so
the degree-scan flag fires by design. Only a "reject" is a real failure.
Optionally also re-runs your own reference sheets (--samples + --map {"file.png": "template.json"})
and reports their status, so a code change that breaks a known layout is caught.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

src = (HERE / "make_truth_sheet.py").read_text().split("q = np.linspace")[0]
ns = {}
exec(src, ns)
head, power, eff = ns["head"], ns["power"], ns["eff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pcd_regression")
    ap.add_argument("--samples")
    ap.add_argument("--map")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pdf = out / "T2500_vector.pdf"
    subprocess.run([sys.executable, str(HERE / "make_truth_sheet.py"), str(pdf)], check=True,
                   capture_output=True)
    from inspect_sheet import run_inspect
    from digitize import run as run_digitize
    ok_all = True
    for dpi in (100, 150, 200, 300):
        wd = out / f"dpi{dpi}"
        draft, _ = run_inspect(str(pdf), wd, 0, dpi, str(HERE / "truth_template.json"))
        cfg = {k: v for k, v in draft.items() if not k.startswith("_")}
        (wd / "sheet.json").write_text(json.dumps(cfg, indent=2))
        if draft.get("_template_missing"):
            print(f"FAIL dpi={dpi}: axes not found {draft['_template_missing']}"); ok_all = False; continue
        r = run_digitize(wd / "sheet.json")
        q = np.linspace(100, 3400, 34)
        errs = {}
        for k, f, span, lim in [("head", head, 70, .5), ("power", power, 1.4, .5), ("eff", eff, 70, 1.5)]:
            e = np.polynomial.polynomial.polyval(q, r["curves"][k]["coeffs"]) - f(q)
            errs[k] = 100 * np.abs(e).max() / span
            ok_all &= errs[k] < lim
        ok_all &= r["status"] != "reject"
        rendered = json.loads((wd / "inspect.json").read_text()).get("dpi", dpi)
        print(f"dpi={dpi:<4}->{rendered:<4} status={r['status']:<7} max err % span: " +
              ", ".join(f"{k}={v:.2f}" for k, v in errs.items()) +
              (f"   flags: {'; '.join(r['flags'])[:70]}" if r["flags"] else ""))
    if a.samples and a.map:
        m = json.loads(Path(a.map).read_text())
        for f, tpl in m.items():
            wd = out / Path(f).stem
            draft, _ = run_inspect(str(Path(a.samples) / f), wd, 0, 200, tpl)
            cfg = {k: v for k, v in draft.items() if not k.startswith("_")}
            (wd / "sheet.json").write_text(json.dumps(cfg, indent=2))
            r = run_digitize(wd / "sheet.json")
            print(f"sample {f:<28} status={r['status']:<7} hyd={r['derived'].get('hydraulic_check_max_diff_pts', float('nan')):.2f}")
    print("PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
