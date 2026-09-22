"""Shared helpers for the pump-curve digitizer pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2


def load_sheet(path: str | Path, page: int = 0, dpi: int = 200) -> np.ndarray:
    """Load a PNG/JPG, or rasterize one page of a PDF, as a BGR uint8 array."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        pg = doc[page]
        # raster page (scan / pasted screenshot): use the embedded image at native resolution
        # -- re-rendering at another DPI resamples it and blurs small tick labels
        imgs = pg.get_images(full=True)
        if len(imgs) == 1 and not pg.get_text().strip():
            info = doc.extract_image(imgs[0][0])
            arr = cv2.imdecode(np.frombuffer(info["image"], np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                return arr
        pix = pg.get_pixmap(dpi=dpi, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def line_mask(img: np.ndarray, contrast: int = 25, max_sat: int = 90) -> np.ndarray:
    """Pixels that are thin, darker than their local background and near-gray (grid/frame lines)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    return ((bg.astype(int) - gray > contrast) & (sat < max_sat)).astype(np.uint8)


def _runs(mask_1d: np.ndarray, max_gap: int = 25):
    """Densest gappy run (start, end, count) of True values; gaps <= max_gap are bridged
    so lines interrupted by curves, labels or shaded bands still register."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return 0, -1, 0
    splits = np.flatnonzero(np.diff(idx) > max_gap) + 1
    best = max(np.split(idx, splits), key=len)
    return int(best[0]), int(best[-1]), int(best.size)


def detect_lines(img: np.ndarray, min_frac: float = 0.25):
    """Find long straight vertical and horizontal lines (grid + frame).

    Returns dict with 'v' and 'h' lists of {pos, start, end}. Dashed lines are
    bridged with a small closing so they register as continuous.
    """
    m = line_mask(img)
    H, W = m.shape
    out = {}
    for axis in ("v", "h"):
        k = np.ones((9, 1), np.uint8) if axis == "v" else np.ones((1, 9), np.uint8)
        mm = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n = W if axis == "v" else H
        length = H if axis == "v" else W
        cand = []
        for i in range(n):
            s, e, cnt = _runs(mm[:, i] if axis == "v" else mm[i, :])
            if cnt >= min_frac * length and cnt >= 0.5 * (e - s + 1):
                cand.append((i, s, e))
        # merge adjacent indices into single lines
        lines = []
        for i, s, e in cand:
            if lines and i - lines[-1]["_last"] <= 1:
                L = lines[-1]
                L["_idx"].append(i); L["_last"] = i
                L["start"] = min(L["start"], s); L["end"] = max(L["end"], e)
            else:
                lines.append({"_idx": [i], "_last": i, "start": s, "end": e})
        out[axis] = [{"pos": float(np.mean(L["_idx"])), "width": len(L["_idx"]),
                      "start": int(L["start"]), "end": int(L["end"])} for L in lines]
    return out


def _dominant_cluster(lines, tol=12):
    """Lines sharing the most common (start, end) extent -- the plot's grid lines."""
    best = []
    for L in lines:
        grp = [M for M in lines if abs(M["start"] - L["start"]) <= tol and abs(M["end"] - L["end"]) <= tol]
        if len(grp) > len(best):
            best = grp
    return best


def _dense_span(profile: np.ndarray, frac: float = 0.15, max_gap: int = 12):
    """Longest stretch (gaps <= max_gap bridged) where a density profile exceeds frac*max."""
    if profile.max() <= 0:
        return 0, len(profile) - 1
    idx = np.flatnonzero(profile >= frac * profile.max())
    segs = np.split(idx, np.flatnonzero(np.diff(idx) > max_gap) + 1)
    best = max(segs, key=lambda a: a[-1] - a[0])
    return int(best[0]), int(best[-1])


def detect_frame(img: np.ndarray):
    """Plot frame [x0, y0, x1, y1] proposal.

    1. Vertical grid lines share a common extent -> y-range.
    2. Within that y-range, columns crossed by many horizontal line pixels -> x-range.
    3. Within that x-range, rows crossed by many vertical line pixels -> refined y-range.
    Page separators, table borders and outboard axis spines carry few grid crossings and
    drop out. Always verify on the review image; override "frame" in sheet.json if wrong.
    """
    lines = detect_lines(img)
    H, W = img.shape[:2]
    m = line_mask(img)
    hs = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8)),
                          cv2.MORPH_OPEN, np.ones((1, 41), np.uint8))
    vs = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)),
                          cv2.MORPH_OPEN, np.ones((41, 1), np.uint8))
    V = _dominant_cluster(lines["v"])
    if V:
        y0, y1 = int(np.median([l["start"] for l in V])), int(np.median([l["end"] for l in V]))
    else:
        y0, y1 = 0, H - 1
    x0, x1 = _dense_span(hs[y0:y1 + 1].sum(axis=0))
    y0, y1 = _dense_span(vs[:, x0:x1 + 1].sum(axis=1))
    x0, x1 = _dense_span(hs[y0:y1 + 1].sum(axis=0))
    return [x0, y0, x1, y1], lines


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    return json.loads(Path(path).read_text())
