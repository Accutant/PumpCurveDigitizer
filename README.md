# ESP Pump Curve Digitizer — Dataset & Catalogue

Digitized coefficient catalogue for Electrical Submersible Pump (ESP) performance
curves, built from vendor-published performance-curve sheets (PDF/PNG). Replaces a
manual WebPlotDigitizer → spreadsheet workflow with an OCR + pixel-tracing pipeline
that fits a 5th-degree polynomial per curve (head, brake horsepower, efficiency vs.
flow rate) and QA-checks the result against basic pump hydraulics before it's added
to the catalogue.

## Contents

| Path | What it is |
|---|---|
| `ESPPump_Coefficients.csv` | **The catalogue.** One row per pump model/frequency: manufacturer, model, series, mechanical limits (OD, shaft, burst pressure), operating range, BEP, and `head_c0..c9` / `bhp_c0..c9` / `eff_c0..c9` polynomial coefficients (ascending powers, fit against flow rate in bbl/d). |
| `digitize_log.csv` | Append-only log of every digitization run: sheet, QA metrics, and which `runs*` folder holds its audit trail. |
| `SLB/` | Source curve sheets from REDA/Schlumberger (58 sheets). |
| `BHI/` | Source curve sheets from Baker Hughes (98 files — single-pump sheets, multi-pump "vs" comparison charts, and a scanned PDF). |
| `AlKhorayef/`, `Borets/` | Reserved for additional manufacturers — not yet populated. |
| `runs/`, `runs_bhi/`, `runs_bhi2/` | Per-sheet audit trail (one subfolder per digitized sheet): the calibration config (`sheet.json`), a review image showing detected axes/frame, an overlay of the fitted curves on the source sheet, extracted point CSVs, and the QA result (`results.json`). Kept so every catalogue row can be traced back to exactly how it was produced and re-verified without re-digitizing. |

## How the catalogue was built

Each source sheet was processed through the **pump-curve-digitizer** toolkit
(a Claude Code skill, not included in this repo — see below):

1. **Inspect** — locate the plot frame, OCR the axis tick labels, and propose curve
   colors from the sheet image.
2. **Calibrate** — confirm axis roles (which axis is head/power/efficiency, what
   units), fix anything OCR got wrong, and read the catalogue metadata (model,
   series, mechanical limits) directly off the sheet text.
3. **Digitize & fit** — trace each curve by color, fit a 5th-degree polynomial, and
   run a hydraulic consistency check (`Q·H·SG / (135770·BHP)` should reconstruct the
   digitized efficiency curve) — this catches wrong axis assignments or unit
   mistakes that a simple R² wouldn't.
4. **Catalogue** — upsert the row into `ESPPump_Coefficients.csv` keyed on
   (manufacturer, model, base_frequency_hz), only after a human (or a documented,
   reviewed exception) has accepted any QA flags.

Several source sheets required non-standard handling — shared/combined axes,
OCR misreads, multi-pump comparison charts where curves share one color — which
is why the audit-trail folders and `notes` column exist: every non-obvious
judgment call made during digitization is recorded against the row it produced.

## Reproducing or extending this

The digitizer toolkit itself (Python scripts + per-vendor layout templates) lives
outside this repo, as a Claude Code skill. To add more sheets or re-run QA on an
existing row, you need that toolkit available locally; this repo holds the
inputs, outputs, and audit trail, not the pipeline code.

## Coefficient convention

For a curve variable `y` (head in ft/stage, BHP in hp/stage, or efficiency in %)
as a function of flow rate `Q` in bbl/d:

```
y(Q) = c0 + c1*Q + c2*Q^2 + c3*Q^3 + ... + c9*Q^9
```

Every fit in this catalogue uses degree 5 (`c6..c9` are always zero), even on a
handful of curves where a higher degree would reduce residual error, since not
every downstream consumer of this catalogue is guaranteed to evaluate beyond `c5`.
Those cases are flagged in the row's `notes` column instead of silently changing
degree.
