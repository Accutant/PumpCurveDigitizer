---
name: pump-curve-digitizer
description: Digitize ESP / centrifugal pump performance-curve sheets (PDF or image with head, power and efficiency curves on separate y-axes), fit 5th-degree polynomial coefficients and add them to the ESPPump_Coefficients catalogue CSV. Use when the user wants pump curve coefficients extracted from curve sheets, replacing a manual WebPlotDigitizer workflow, for one sheet or a batch.
---

# Pump curve digitizer

Replaces the manual WebPlotDigitizer → CSV → polynomial-fit → catalogue workflow.
Code does everything pixel-accurate; you (Claude) do the judgment steps: confirm axis
roles, pick curve colours, read catalogue fields off the sheet, and review QA images.
**Never estimate curve values or coefficients by eye** — vision is not precise enough.

## 0. Locate the toolkit and set up

The toolkit is the folder containing this file (`scripts/`, `templates/`, `tests/`). If this
SKILL.md was installed without it, look for a `pump-curve-digitizer` folder or zip in the
user's connected folders / attachments / repo; if you can't find it, ask for it.

```bash
pip install opencv-python-headless numpy pymupdf pytesseract matplotlib   # add --break-system-packages in sandboxes
tesseract --version        # the Tesseract binary is required
```
Windows: install Tesseract (UB Mannheim build) and set `TESSERACT_CMD` to `tesseract.exe` if
it is not on PATH. Commands below run from the toolkit root.

## 1. Single sheet

```bash
python scripts/inspect_sheet.py SHEET.pdf --out runs/<model> [--page N] [--template templates/<layout>.json]
```
Prints the plot frame, OCR'd axis candidates (id, side, value range, residual, grid snap),
text labels inside the plot, and candidate curve colours. Writes `sheet.png`, `review.png`,
`inspect.json`, `sheet.draft.json`.

**Look at `review.png` and `sheet.png` (Read tool)**, then write `runs/<model>/sheet.json`
from the draft:
- `x_axis`, `y_axes.head|power|eff`: copy `ticks` from the right `_candidates` entry and set
  `unit` from the axis title: x `bpd`|`m3d`; head `ft`|`m`; power `hp`|`kw`; eff `%`|`fraction`.
  Axes marked `ALT-UNITS-OF` are the second label set of a dual-unit scale (m|ft, kW|hp):
  prefer the primary (imperial) one. Omit a y-axis/curve that the sheet doesn't have.
- `curves.<role>.color`: hex from `_colors` (or sample `sheet.png` pixels). Black curves:
  `"#000000"`. Dashed curves (or solid-in-ROR/dashed-outside): `"style": "dashed"`.
- `exclude_boxes`: pre-filled from in-plot text labels; add `[x0,y0,x1,y1]` boxes for legend
  boxes, annotations or markers drawn in a curve's colour.
- `frame`: fix only if review.png shows the magenta box wrong.
- `meta` — read from the sheet text, never guess: `manufacturer` (null if not printed — the
  catalogue step then refuses the row, ask the user), `model`, `base_frequency_hz`, `rpm`,
  `series`, `pump_od_in` (convert mm/cm), `ror_min_bpd`/`ror_max_bpd` (convert m³/d ×6.289811),
  `bep_bpd` if marked, `shaft_limit_hp` (standard shaft), `housing_burst_psi` (buttress;
  kPa ×0.1450377), `catalog_source`; put alternates (HS shaft, V-thread, min casing) in `notes`.
- `fit.degree` stays 5 unless the user agrees otherwise; `fit.sg` = the sheet's SG (usually 1.00).

```bash
python scripts/digitize.py runs/<model>/sheet.json
```
Then **look at `overlay.png`** (fitted curves over the faded sheet — every curve must sit on
its line, no jumps onto labels/other curves) and **`fit_plot.png`** (points, fit, residuals,
red dashed = efficiency recomputed from Q·H/(135770·BHP)).

## 2. Reading the QA result (`results.json`, console)

`status` is `ok` only if every check passes. Flags and what to do:

| Flag | Meaning | Action |
|---|---|---|
| `curve not found` / `only N points` / `gap of N%` | colour mask missed the curve | check colour hex, raise `tol` (35→50), `style: dashed`, remove an exclude box covering it |
| `max residual` / `rmse` + "degree N would meet the limit" | extraction clean, 5th-degree form too stiff (typ. efficiency runout) | report to user with the degree scan; don't change degree silently (catalogue has c0..c9 columns but the consuming program may assume 5) |
| `max residual` without a degree suggestion | extraction picked up junk | inspect overlay; add exclude boxes; tighten `tol` |
| `hydraulic check ... differs by N pts` | head, power and efficiency don't agree physically | first suspect axis roles/units/SG; if those are right and overlay is clean, the **sheet itself** is inconsistent — tell the user |
| `axis calibration residual > 3 px` | a tick value/position is wrong | fix `ticks` (read px from a zoomed crop of `sheet.png`) |
| `raw-coefficient round-trip error` | precision lost converting to raw-unit coefficients | report; consider lower degree |

The hydraulic check is the strongest single test: it catches wrong axis roles, unit
mistakes and small calibration offsets that R² never shows.

## 3. Write to the catalogue

```bash
python scripts/catalogue.py path/ESPPump_Coefficients.csv runs/*/results.json [--accept-review] [--dry-run]
```
Upserts on (manufacturer, model, base_frequency_hz), keeps the file's column order, writes a
timestamped `.bak` first, and appends QA metrics + work folder to `digitize_log.csv`.
`status: review` rows are skipped unless `--accept-review`, which you pass **only after the
user has accepted those specific flags** (the flag text is copied into the row's notes).
Coefficients are C0..C5 in raw catalogue units (bpd → ft/stage, hp/stage, eff %), ascending
powers, same convention as `Polynomial.fit(...).convert()` in the user's fitter.

## 4. Batches (many sheets, same layout)

A template (`templates/*.json`) fixes axis roles by position (`left:0` = first primary axis left
of the plot, `right:1` = second on the right, `bottom:0` = x), units, curve colours/styles and
meta defaults. Make one from the first good `sheet.json` of a new layout.
```bash
python scripts/batch.py "sheets/*.pdf" --template templates/reda_imperial.json --out runs/
```
Each PDF page is a sheet. `runs/batch_summary.csv` lists status/flags per sheet. Then for
each sheet: fill the per-sheet `meta` (model, ROR, limits) in its `sheet.json` from the sheet
text, re-run `digitize.py` on it, review overlays of every `review`/`needs-axes`/`error`
sheet (spot-check a few `ok` ones), and write the accepted results to the catalogue in one
`catalogue.py` call. `needs-axes` = the template's axis positions weren't found — handle as
a single sheet.

## 5. Recurring gotchas (learned across REDA and Baker Hughes batches)

These don't show up as clean errors — each one produces a plausible-looking sheet.json
or an `ok`/`review` status with a good R², and only gets caught if you specifically check
for it. Budget time for all of them on any new batch.

| Symptom | What's actually wrong | Fix |
|---|---|---|
| `template roles not found: [eff (right:1)]` but `y_axes.eff` in the draft is already filled | Some sheets print power and efficiency sharing **one** right-hand numeric column (only one right axis, e.g. `R2`, shows up in inspect's console output) instead of two. `draft_config`'s fallback already copies the single axis into both roles — the "missing" annotation is just a warning, not a blocker. Trust the fallback (verify against the sheet: do Power and Eff visually share one printed number column?), strip `_template_missing` and run digitize.py directly. If the fallback key is absent entirely (`'eff' not in y_axes`), copy `y_axes.power`'s ticks in by hand with `"unit": "%"`. |
| `template roles not found: [x (bottom:0)]`, or frame height is much shorter than other sheets of the same pixel size | Auto `detect_frame()` collapsed to a too-short box (check `inspect.json`'s `"frame"` — a suspiciously small y-range, e.g. under half the height of sibling sheets, is the tell, not a real short chart). | Recompute the frame from border/gridline detection instead of trusting `detect_frame()`: `common.py`'s `detect_lines()` + a column/row density scan for the long solid axis lines. Often the SAME corrected frame works for every sheet from that PDF/export batch — compute it once, reuse it. |
| Fitted curve looks right in `overlay.png` but `hydraulic_check_max_diff_pts` is tens of points, not single digits, **and every individual axis looks fine when you eyeball it** | OCR misread one or more **tick values** (not just missed them) — this is more dangerous than a missing axis because nothing errors, you just get wrong-but-plausible coefficients. Two concrete patterns seen: (1) a "repair" heuristic corrected an OCR digit-run wrong, e.g. read `100,150,300` as `1,150,300`, when the true sequence (checked against the image) was `100,150,200`; (2) trailing zeros dropped from every tick on one axis, e.g. `2,4,6,8,10` printed for gridlines actually labelled `20,40,60,80,100`. | Never trust OCR'd tick *values* just because the pixel spacing is linear/consistent — that only proves internal consistency, not correctness. Re-read the axis labels directly off a zoomed crop of the sheet and compare. The hydraulic check is what will catch this — treat any hydraulic-check flag as "go re-read the axis labels by eye" before assuming it's a genuine sheet inconsistency. |
| `exclude_boxes` from `text_labels()` swallow a real chunk of a curve (shows as a `gap of N%` flag, or the fitted curve goes suspiciously straight through a region where it should bend) | OCR hallucinates a text label from a dense grid-line + curve crossing and draws an exclude box there — there is no real text badge in that region. Seen more than once on the same chart template. | Crop and look (Read tool) at the pixel region of every auto-generated exclude box before trusting it. If there's no visible text badge — just gridlines/curves crossing — delete that box and re-run. Only keep boxes that visibly correspond to a printed label (legend, "OPERATING RANGE" badge, BEP data box). |
| `hydraulic_check` is wildly wrong (e.g. 40-400+ pts) specifically on a small/low-flow pump, and reversing the printed tick order on power and/or efficiency (keep the same pixel rows, swap which value goes with which) makes the derived BEP match a printed BEP box closely | Some small pumps print their Power/Efficiency axis with values **increasing downward** (opposite of the Head axis and every other sheet in the batch) — a real, if unusual, vendor charting choice, not an OCR error. | If a BEP box is printed on the sheet, use it as ground truth: try the calibration both ways and keep whichever reproduces the printed Q/H/P/E. Note the finding in `meta.notes`; don't assume every sheet in a batch uses the same axis direction. |
| Two files in the source folder produce near-identical model/series/BEP values | Duplicate scans of the same physical sheet (different export, different crop, sometimes a screenshot of the other file) | Check before spending digitizing effort on both — `catalogue.py`'s upsert on (manufacturer, model, base_frequency_hz) will just collapse them anyway. Pick the cleaner source and skip the duplicate; note it in your final report. |
| A comparison chart overlays 2-3 pumps and the normal colour-mask extraction only finds 3 curves total instead of 6-9 | One (or more) of the overlaid pumps draws its head, power AND efficiency all in the same colour — impossible to separate by per-colour masking. Common on Baker Hughes "PumpA vs PumpB [vs PumpC]" sheets; the *first-listed* pump usually gets distinct colours, later ones share one. | Use `scripts/same_color_tracer.py` (`trace_two` for head+eff through their crossing, `trace_one` restricted to a low-value y-band for power) + `scripts/manual_fit.py` (`build_results`, reuses `digitize.py`'s exact fitting/QA math) to bypass the colour-mask pipeline for that pump. See the module docstring for tuning notes (velocity inertia, crossing-point robustness, excluding legend callout lines, avoiding power/head tail contamination). Verify with the hydraulic check same as any other curve. |

## 6. Rules

- Keep each `runs/<model>/` folder (sheet.png, sheet.json, points CSVs, overlay) — it's the
  audit trail for every catalogue row.
- Don't hand-edit coefficients; change `sheet.json` and re-run.
- After changing any script, run `python tests/regression.py` (synthetic vector PDF with
  known curves at 100–300 dpi; must print PASS).
- Report to the user: per sheet status, the flags you accepted or couldn't resolve, and
  anything read from the sheet you were unsure about (manufacturer naming, which limit).
