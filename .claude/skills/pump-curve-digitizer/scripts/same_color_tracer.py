"""Extract curves that share one printed colour on a multi-curve chart.

Some vendor comparison charts (e.g. Baker Hughes "PumpA vs PumpB" sheets) draw a
second/third pump's head, power AND efficiency curves in a single colour instead of
three distinct ones. digitize.py's curve_mask (one colour -> one curve) can't
separate them. This module does it by column-wise pixel clustering plus a
velocity-predicted nearest-neighbour tracker per curve, seeded at each curve's known
starting point (head starts at its printed shutoff-adjacent value; efficiency and
often power start at/near 0 at zero flow).

Usage pattern (see the "vs" sheets in a REDA/Baker-Hughes batch for a worked example):
    from same_color_tracer import color_mask, clusters_in_column, Tracker, trace_two
    from manual_fit import build_results   # turns traced px points into a results.json

    img = cv2.imread(sheet_png)
    head_pts, eff_pts = trace_two(img, "#aac54c", tol=45, x0=98, x1=1316,
                                   y_top=104, y_bottom=721,
                                   seed_head_y=368, seed_eff_y=719,
                                   max_jump=8, vel_smooth=0.95)
    # power is usually a thin dotted line confined to a low-value y-band; trace it
    # separately with trace_one() restricted to that band (see below).

Tuning notes learned the hard way:
- vel_smooth=0.95 (heavy inertia) is needed to survive the point where two curves
  cross pixel-for-pixel -- low inertia (0.7-0.85) lets the tracker jump onto the
  wrong curve right after a crossing. Only drop inertia for a curve with no
  crossings to track (e.g. an isolated power line).
- max_jump should be tight (6-10 px) for curves with real crossings, looser
  (15-20 px) only when re-acquiring after a text-label gap and there is no
  competing nearby curve.
- Dashed/dotted vertical marker lines (operating-range boundaries) are often
  rendered in one of the tracked colours and appear as a single column with many
  (5+) clusters -- skip any column with more than ~6 clusters, it is never real
  curve data.
- A curve's OWN legend/label box and its callout line are frequently drawn in the
  same tracked colour -- pass their pixel rectangle(s) via `exclude` to zero them
  out of the mask before tracing, the same way sheet.json's exclude_boxes work for
  the normal per-colour pipeline.
- Power is often confined to a narrow low-value y-band for the whole chart; trace
  it with a SEPARATE call restricted to that band (y_top/y_bottom much tighter)
  rather than folding it into trace_two, and seed a bit past x0 once the other
  curve(s) have visibly cleared that band, to avoid an ambiguous multi-cluster
  start. Truncate the power trace before the x-range where a declining head/eff
  curve's tail re-enters that same low band near shutoff, or it will get stolen.
"""
import numpy as np


def color_mask(img, color_hex, tol):
    """Boolean mask of pixels within `tol` (sum of abs RGB diff) of color_hex."""
    hexc = color_hex.lstrip('#')
    r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
    diff = (np.abs(img[:, :, 2].astype(int) - r) + np.abs(img[:, :, 1].astype(int) - g)
            + np.abs(img[:, :, 0].astype(int) - b))
    return diff < tol


def clusters_in_column(mask_col, gap=2):
    """Centres of contiguous True-runs in a 1-D boolean column (one per curve crossing)."""
    idx = np.flatnonzero(mask_col)
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


class Tracker:
    """Single-curve column tracker: predicts next y from smoothed velocity, then
    the caller snaps it to the nearest real cluster within max_jump."""
    def __init__(self, y0, vel_smooth=0.7):
        self.y = y0
        self.vel = 0.0
        self.vel_smooth = vel_smooth
        self.pts = []

    def predict(self):
        return self.y + self.vel

    def update(self, y_new, x):
        new_vel = y_new - self.y
        self.vel = self.vel_smooth * self.vel + (1 - self.vel_smooth) * new_vel
        self.y = y_new
        self.pts.append((x, y_new))


def apply_exclude(mask, boxes):
    """Zero out mask pixels inside each (x0,y0,x1,y1) box -- same convention as
    sheet.json's exclude_boxes. Mutates and returns mask."""
    for bx0, by0, bx1, by1 in boxes:
        mask[by0:by1, bx0:bx1] = False
    return mask


def trace_two(img, color_hex, tol, x0, x1, y_top, y_bottom, seed_head_y, seed_eff_y,
              max_jump=15, vel_smooth=0.7, exclude=()):
    """Trace two same-coloured curves (typically head + efficiency) through their
    crossing(s). Returns (head_pts, eff_pts), each a list of (x_px, y_px)."""
    mask = color_mask(img, color_hex, tol)
    if exclude:
        mask = apply_exclude(mask, exclude)
    head = Tracker(seed_head_y, vel_smooth)
    eff = Tracker(seed_eff_y, vel_smooth)
    for x in range(x0, x1 + 1):
        cs = [c + y_top for c in clusters_in_column(mask[y_top:y_bottom, x])]
        if not cs or len(cs) > 6:   # >6 clusters in one column = dashed marker line, not curve data
            continue
        ph, pe = head.predict(), eff.predict()
        best_head = min(cs, key=lambda c: abs(c - ph))
        best_eff = min(cs, key=lambda c: abs(c - pe))
        if abs(best_head - ph) <= max_jump:
            head.update(best_head, x)
        if abs(best_eff - pe) <= max_jump:
            eff.update(best_eff, x)
    return head.pts, eff.pts


def trace_one(img, color_hex, tol, x0, x1, y_top, y_bottom, seed_y,
              max_jump=8, vel_smooth=0.85, exclude=(), max_clusters=6):
    """Trace a single curve (typically power, restricted to its own low-value
    y-band to avoid the other same-coloured curves). Returns list of (x_px, y_px)."""
    mask = color_mask(img, color_hex, tol)
    if exclude:
        mask = apply_exclude(mask, exclude)
    tr = Tracker(seed_y, vel_smooth)
    for x in range(x0, x1 + 1):
        cs = [c + y_top for c in clusters_in_column(mask[y_top:y_bottom, x])]
        if not cs or len(cs) > max_clusters:
            continue
        pred = tr.predict()
        best = min(cs, key=lambda c: abs(c - pred))
        if abs(best - pred) <= max_jump:
            tr.update(best, x)
    return tr.pts
