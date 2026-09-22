# Pump curve digitizer

Automates: curve sheet (PDF/PNG) → axis calibration → curve points → 5th-degree fits →
`ESPPump_Coefficients.csv`. It replaces the manual WebPlotDigitizer steps.

```
inspect_sheet.py  sheet -> frame, OCR axis calibration (grid-snapped), colours, draft sheet.json, review.png
digitize.py       sheet.json -> points_*.csv, fits, QA (R², residuals, hydraulic check), overlay.png, fit_plot.png
catalogue.py      results.json(s) -> upsert rows in the catalogue (+ .bak, + digitize_log.csv)
batch.py          folder of sheets + layout template -> all of the above + batch_summary.csv
tests/regression.py   synthetic vector PDF with known curves at 100-300 dpi -> PASS/FAIL
```

## Install
`pip install -r requirements.txt`, plus the Tesseract binary. On Windows, if `tesseract.exe` isn't on
PATH: `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

## How the automation works (and where it can go wrong)
1. **Frame.** The plot is where gridlines are dense in both directions. Check the magenta box in review.png.
2. **Axis calibration.** Tesseract reads the tick labels (after erasing gridlines and tick blocks). A
   robust line fit per axis rejects misreads and repairs dropped decimal points ("04"→0.4). Label
   positions are then **snapped to the gridlines** they label. REDA prints labels about 4 px above their
   lines, and without the snap that offset caused a 7-point efficiency error at the low end of the power axis.
   Dual-unit scales (m|ft, kW|hp, m³/d|bbl/d) are recognised by their unit ratio.
3. **Curves.** A colour mask in Lab space removes straight lines of the same colour (BEP markers,
   grid), in-plot text labels and specks. Each column's candidates are tracked with an iterative
   robust fit, so crossings, legend boxes and dashes don't pull the curve.
4. **Fit.** `Polynomial.fit(x, y, 5).convert()` is the same call as in PumpCoefficientGenerator.py,
   so the coefficients are raw-unit C0..C5 in bpd, ft/stage, hp/stage and eff %.
5. **QA.** Checks are R², RMSE and max residual as a percentage of axis span, the largest point gap,
   a degree scan (3–9), a raw-coefficient round-trip check, and the **hydraulic check**:
   η = Q·H·SG/(135770·BHP) compared with the digitized efficiency. Anything outside limits gives
   `status: review`, and the catalogue writer skips those unless told otherwise.

## Tested on
Five real layouts (Weatherford/Centrilift DC550, Baker Hughes FLEX6, REDA SN2600 and AN1200
(metric, 50 Hz), and NBV dashed dual-unit) and a synthetic vector PDF with known curves
(errors ≤0.41% of axis span at 100–300 dpi). See `examples/`.
