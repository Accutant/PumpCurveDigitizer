"""Step 1 of the pump-curve pipeline: inspect a curve sheet and propose a calibration.

Usage:
    python inspect_sheet.py SHEET.(png|jpg|pdf) --out WORKDIR [--page N] [--dpi 200]

Writes to WORKDIR:
    sheet.png          the rasterized sheet (all later pixel coordinates refer to it)
    inspect.json       frame proposal, OCR'd axis candidates with linear fits, curve colours
    review.png         annotated image: frame, axis ids, OCR ticks, curve-colour swatches
    sheet.draft.json   a draft config for digitize.py -- Claude reviews and completes it

Nothing here is final: the draft must be checked against review.png (axis roles,
units, curve colours) before running digitize.py.
"""
from __future__ import annotations
import argparse
import itertools
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from common import load_sheet, detect_frame, line_mask, save_json

import os
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

NUM_RE = re.compile(r"^[-+]?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?$")
OCR_SCALE = 4


# ----------------------------------------------------------------------------- OCR
def _clean_for_ocr(img):
    """Grayscale copy with long grid/axis lines and solid tick blocks painted white."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    lm = line_mask(img, contrast=15, max_sat=255)
    long_h = cv2.morphologyEx(lm, cv2.MORPH_OPEN, np.ones((1, 25), np.uint8))
    long_v = cv2.morphologyEx(lm, cv2.MORPH_OPEN, np.ones((25, 1), np.uint8))
    dark = (g < 140).astype(np.uint8)
    solid = cv2.dilate(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8)),
                       np.ones((3, 3), np.uint8))
    g[(long_h | long_v | solid) > 0] = 255
    return g


def ocr_numbers(img, strips=()):
    """Numeric tokens with pixel boxes.

    Tesseract runs twice (plain, and digits-only whitelist). Words are split on their
    numeric runs so a label pair like '6.1-20' (m and ft beside one tick) yields two
    tokens; x positions of the parts are apportioned by character count (y is shared).
    """
    g = _clean_for_ocr(img)
    H, W = g.shape
    passes = [((0, 0, W, H), c) for c in ("--psm 11", "--psm 11 -c tessedit_char_whitelist=0123456789.,")]
    # margin strips OCR'd on their own (block mode) -- sparse mode skips small bold labels
    for (sx0, sy0, sx1, sy1) in strips:
        box = (max(0, int(sx0)), max(0, int(sy0)), min(W, int(sx1)), min(H, int(sy1)))
        if box[2] - box[0] > 10 and box[3] - box[1] > 10:
            passes += [(box, "--psm 6"), (box, "--psm 11")]
    toks = []
    for (ox, oy, ex, ey), cfg in passes:
        big = cv2.resize(g[oy:ey, ox:ex], None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        d = pytesseract.image_to_data(big, config=cfg, output_type=pytesseract.Output.DICT)
        for i, word in enumerate(d["text"]):
            word = word.strip()
            if not word:
                continue
            word = re.sub(r"(?<=\d)[oO]|[oO](?=[\d.])", "0", word)
            L, T = (d["left"][i] - 40) / OCR_SCALE + ox, (d["top"][i] - 40) / OCR_SCALE + oy
            w, h = d["width"][i] / OCR_SCALE, d["height"][i] / OCR_SCALE
            for m in re.finditer(r"\d[\d,]*(\.\d+)?|\.\d+", word):
                text = m.group().strip(",")
                if not NUM_RE.match(text):
                    continue
                l = L + w * m.start() / len(word)
                r = L + w * m.end() / len(word)
                toks.append({"text": text, "value": float(text.replace(",", "")),
                             "left": l, "right": r, "top": T, "bottom": T + h,
                             "cx": (l + r) / 2, "cy": T + h / 2,
                             "conf": float(d["conf"][i]), "split": len(text) < len(word)})
    out = []   # de-duplicate the two passes, prefer the higher-confidence reading
    for t in sorted(toks, key=lambda t: -t["conf"]):
        if not any(abs(t["cy"] - u["cy"]) < 4 and t["left"] < u["right"] and u["left"] < t["right"]
                   for u in out):
            out.append(t)
    return out


def _alternatives(text):
    """Candidate values for a token; OCR often drops decimal points ('04' for 0.4)."""
    base = float(text.replace(",", ""))
    alts = [base]
    t = text.replace(",", "")
    if "." not in t and len(t) >= 2:
        alts += [float(t[:i] + "." + t[i:]) for i in range(1, len(t))]
    return alts


def _group(tokens, orient):
    """Group tokens that line up as one axis' tick labels."""
    n = len(tokens)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(n), 2):
        a, b = tokens[i], tokens[j]
        if orient == "v":
            tol = 6 if (a["split"] or b["split"]) else 3
            linked = (abs(a["left"] - b["left"]) <= tol or abs(a["right"] - b["right"]) <= tol) \
                and abs(a["cy"] - b["cy"]) > 6
        else:
            linked = abs(a["cy"] - b["cy"]) <= 3 and abs(a["cx"] - b["cx"]) > 6
        if linked:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(tokens[i])
    return [g for g in groups.values() if len(g) >= 3]


def _fit_axis(group, orient, tol_px=2.5):
    """RANSAC line value = a + b*pixel through tick labels (with decimal-point repair)."""
    px = np.array([t["cy"] if orient == "v" else t["cx"] for t in group])
    alts = [_alternatives(t["text"]) for t in group]
    best = None
    for i, j in itertools.combinations(range(len(group)), 2):
        if abs(px[i] - px[j]) < 10:
            continue
        for vi in alts[i]:
            for vj in alts[j]:
                if vi == vj:
                    continue
                b = (vj - vi) / (px[j] - px[i])
                if (orient == "v" and b >= 0) or (orient == "h" and b <= 0):
                    continue
                a = vi - b * px[i]
                chosen = []
                for k in range(len(group)):
                    errs = [abs((v - a) / b - px[k]) for v in alts[k]]
                    m = int(np.argmin(errs))
                    chosen.append(alts[k][m] if errs[m] <= tol_px else None)
                n_in = sum(c is not None for c in chosen)
                repaired = sum(c is not None and c != alts[k][0] for k, c in enumerate(chosen))
                score = n_in - 0.5 * repaired  # a repair must buy more than one inlier
                if best is None or score > best[0]:
                    best = (score, chosen)
    if best is None or sum(c is not None for c in best[1]) < 3:
        return None
    chosen = best[1]
    keep = [k for k, c in enumerate(chosen) if c is not None]
    val = np.array([chosen[k] for k in keep])
    p = px[keep]
    # tick labels sit on a lattice v0 + k*step: drop values off the lattice (stray tokens
    # such as a curve annotation), and reject coincidental alignments of random digits
    sv = np.sort(np.unique(val))
    if len(sv) < 3:
        return None
    diffs = np.round(np.diff(sv), 9)
    best_lat = None
    for step in np.unique(diffs):
        if step <= 0:
            continue
        for v0 in sv:
            k = (val - v0) / step
            on = np.abs(k - np.round(k)) < 0.05
            if on.sum() < 3:
                continue
            slots = (val[on].max() - val[on].min()) / step + 1
            if slots > 2.5 * on.sum():          # lattice far denser than the labels: not ticks
                continue
            if best_lat is None or on.sum() > best_lat[0].sum() or \
                    (on.sum() == best_lat[0].sum() and step > best_lat[1]):
                best_lat = (on, step)
    if best_lat is None:
        return None
    on = best_lat[0]
    for k_ in [keep[i] for i in np.flatnonzero(~on)]:
        chosen[k_] = None
    keep = [k for k, c in enumerate(chosen) if c is not None]
    val = np.array([chosen[k] for k in keep])
    p = px[keep]
    b, a = np.polyfit(p, val, 1)
    resid = np.abs((val - a) / b - p)
    return {"a": float(a), "b": float(b),
            "ticks": sorted([{"value": float(v), "px": float(q)} for v, q in zip(val, p)], key=lambda t: t["value"]),
            "rejected": [group[k]["text"] for k, c in enumerate(chosen) if c is None],
            "repaired": [f"{group[k]['text']}->{c:g}" for k, c in enumerate(chosen)
                         if c is not None and c != alts[k][0]],
            "max_resid_px": float(resid.max()),
            "value_range": [float(val.min()), float(val.max())],
            "pos": float(np.median([group[k]["cx"] if orient == "v" else group[k]["cy"] for k in keep])),
            "_keep_ids": {id(group[k]) for k in keep}}


def snap_to_grid(fit, lines, frame, max_off=5.0, min_share=0.7, max_spread=3.0):
    """Move OCR tick positions onto the grid lines they label.

    Label centres can sit a few px off their tick (fonts, baseline placement), which becomes a
    bias when the axis is extrapolated. If most ticks have a grid line within a few px and the
    offsets agree with each other, the grid positions replace the label positions.
    """
    x0, y0, x1, y1 = frame
    if fit["orient"] == "v":
        pos = np.array([l["pos"] for l in lines["h"]], float)
    else:
        pos = np.array([l["pos"] for l in lines["v"]], float)
    if pos.size == 0:
        fit["snapped"] = 0
        return fit
    pos = np.unique(pos)
    spacing = np.median(np.diff(pos)) if pos.size > 1 else 1e9
    lim = min(max_off, 0.45 * spacing)
    offs = []
    for t in fit["ticks"]:
        j = int(np.argmin(np.abs(pos - t["px"])))
        offs.append((pos[j] - t["px"], pos[j]))
    good = [(o, p) for o, p in offs if abs(o) <= lim]
    if len(good) >= max(2, min_share * len(offs)) and \
            np.ptp([o for o, _ in good]) <= max_spread:
        for t, (o, p) in zip(fit["ticks"], offs):
            if abs(o) <= lim:
                t["ocr_px"], t["px"] = t["px"], float(p)
        px = np.array([t["px"] for t in fit["ticks"]])
        val = np.array([t["value"] for t in fit["ticks"]])
        b, a = np.polyfit(px, val, 1)
        fit.update(a=float(a), b=float(b), max_resid_px=float(np.abs((val - a) / b - px).max()))
        fit["snapped"] = len(good)
        fit["snap_offset_px"] = float(np.median([o for o, _ in good]))
    else:
        fit["snapped"] = 0
    return fit


def text_labels(img, frame, pad=4):
    """Word labels inside the frame (curve names, 'BEP', 'Operating range' ...) as boxes.
    These are pre-filled as exclude_boxes so label text in a curve's colour isn't digitized."""
    x0, y0, x1, y1 = frame
    g = _clean_for_ocr(img)
    big = cv2.resize(g, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(big, config="--psm 11", output_type=pytesseract.Output.DICT)
    out = []
    for i, w in enumerate(d["text"]):
        w = w.strip()
        if len(re.sub(r"[^A-Za-z]", "", w)) < 2 or float(d["conf"][i]) < 60:
            continue
        L, T = d["left"][i] / OCR_SCALE, d["top"][i] / OCR_SCALE
        R, B = L + d["width"][i] / OCR_SCALE, T + d["height"][i] / OCR_SCALE
        if x0 <= (L + R) / 2 <= x1 and y0 <= (T + B) / 2 <= y1:
            out.append({"text": w, "box": [int(L - pad), int(T - pad), int(R + pad), int(B + pad)]})
    return out


def find_axes(img, frame, lines=None):
    """OCR the whole sheet, keep tokens beside/below the frame, fit candidate axes."""
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0
    W = img.shape[1]
    toks = ocr_numbers(img, strips=[
        (x0 - 0.05 * fw, y1 - 8, x1 + 0.05 * fw, y1 + 0.12 * fh),          # x labels
        (0, y0 - 0.08 * fh, x0 + 0.08 * fw, y1 + 0.08 * fh),                # left y labels
        (x1 - 0.08 * fw, y0 - 0.08 * fh, W, y1 + 0.08 * fh)])               # right y labels
    in_y = [t for t in toks if y0 - 0.08 * fh <= t["cy"] <= y1 + 0.08 * fh]
    sides = {
        "left": ([t for t in in_y if t["cx"] < x0 + 0.08 * fw], "v"),
        "right": ([t for t in in_y if t["cx"] > x1 - 0.08 * fw], "v"),
        "bottom": ([t for t in toks if y1 - 6 <= t["cy"] <= y1 + 0.25 * fh
                    and x0 - 0.05 * fw <= t["cx"] <= x1 + 0.05 * fw], "h"),
    }
    axes = []
    for side, (tk, orient) in sides.items():
        for g in _group(tk, orient):
            # one alignment group can hold several axes (e.g. HP and Eff labels sharing a
            # left edge): peel off the best line, refit on what it rejected, repeat
            while len(g) >= 3:
                fit = _fit_axis(g, orient)
                if not fit:
                    break
                fit.update(side=side, orient=orient)
                if lines is not None:
                    snap_to_grid(fit, lines, frame)
                axes.append(fit)
                used = {(round(t["value"], 6), round(t["px"], 1)) for t in fit["ticks"]}
                g = [t for t in g if fit["_keep_ids"] is None or id(t) not in fit["_keep_ids"]]
                fit.pop("_keep_ids")
    # the same labels can be found twice (two OCR passes, slightly different boxes): merge
    uniq = []
    for a in sorted(axes, key=lambda a: -len(a["ticks"])):
        dup = False
        for u in uniq:
            if u["side"] != a["side"]:
                continue
            same = sum(any(abs(t["value"] - v["value"]) < 1e-9 and abs(t["px"] - v["px"]) < 4
                           for v in u["ticks"]) for t in a["ticks"])
            if same >= 2 or (abs(a["pos"] - u["pos"]) < 4 and
                             abs((a["a"] + a["b"] * u["ticks"][0]["px"]) - u["ticks"][0]["value"])
                             < 0.02 * (u["value_range"][1] - u["value_range"][0] + 1e-9)):
                dup = True
                break
        if not dup:
            uniq.append(a)
    axes = uniq
    # dual-unit labelling (m|ft, hp|kW, m3/d|bbl/d) puts two label sets on one scale: their
    # value-per-pixel slopes differ by a known unit factor. The imperial one (larger values,
    # = catalogue units) stays primary; the other is marked alt_of and skipped by templates.
    UNIT_RATIOS = (0.3048, 0.7457, 0.158987)
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            if a["side"] != b["side"] or a.get("alt_of") or b.get("alt_of"):
                continue
            ratio = abs(a["b"] / b["b"])
            ratio = min(ratio, 1 / ratio)
            zero_a, zero_b = -a["a"] / a["b"], -b["a"] / b["b"]      # pixel where value = 0
            if any(abs(ratio / r - 1) < 0.04 for r in UNIT_RATIOS) and abs(zero_a - zero_b) < 3:
                prim, alt = (a, b) if abs(a["b"]) > abs(b["b"]) else (b, a)
                alt["alt_of"] = prim
    order = {"bottom": 0, "left": 1, "right": 2}
    axes.sort(key=lambda a: (order[a["side"]], a["pos"] if a["side"] != "left" else -a["pos"]))
    for k, a in enumerate(axes):
        a["id"] = f"{a['side'][0].upper()}{k}"
    for a in axes:
        if a.get("alt_of"):
            a["alt_of"] = a["alt_of"]["id"]
    return axes


# ----------------------------------------------------------------------------- colours
def curve_colors(img, frame, k=6):
    """Candidate curve colours inside the frame.

    Only thin foreground pixels are considered (flat fills such as operating-range bands are
    removed by a median background). Saturated pixels are clustered; near-black pixels that
    are NOT part of straight grid lines are reported separately as a possible black curve.
    """
    x0, y0, x1, y1 = frame
    sub = img[y0:y1, x0:x1]
    bg = cv2.medianBlur(sub, 15)
    fg = np.linalg.norm(sub.astype(int) - bg.astype(int), axis=2) > 60
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    out = []
    sat = fg & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)
    px = sub[sat].reshape(-1, 3).astype(np.float32)
    if len(px) >= 50:
        lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        k = min(k, max(1, len(px) // 50))
        _, labels, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
                                  3, cv2.KMEANS_PP_CENTERS)
        for c in range(k):
            sel = px[labels.ravel() == c]
            if len(sel) < 0.03 * len(px):
                continue
            bgr = np.median(sel, axis=0).astype(int)
            out.append({"kind": "color", "hex": "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]),
                        "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])], "pixels": int(len(sel))})
    # dark, non-grid pixels
    from common import line_mask
    lm = line_mask(sub, contrast=15, max_sat=255)
    grid = cv2.morphologyEx(lm, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8)) | \
        cv2.morphologyEx(lm, cv2.MORPH_OPEN, np.ones((41, 1), np.uint8))
    dark = fg & (hsv[:, :, 2] < 70) & (cv2.dilate(grid, np.ones((3, 3), np.uint8)) == 0)
    n_dark = int(dark.sum())
    if n_dark > 0.002 * dark.size:
        out.append({"kind": "dark", "hex": "#000000", "rgb": [0, 0, 0], "pixels": n_dark})
    # merge near-duplicate clusters
    merged = []
    for c in sorted(out, key=lambda c: -c["pixels"]):
        if not any(np.linalg.norm(np.array(c["rgb"]) - np.array(m["rgb"])) < 45 for m in merged):
            merged.append(c)
    return merged


# ----------------------------------------------------------------------------- review image
def review_image(img, frame, axes, colors, path, labels=()):
    o = img.copy()
    for l in labels:
        bx = l["box"]
        cv2.rectangle(o, (bx[0], bx[1]), (bx[2], bx[3]), (0, 165, 255), 1)
    x0, y0, x1, y1 = frame
    cv2.rectangle(o, (x0, y0), (x1, y1), (255, 0, 255), 1)
    palette = [(0, 0, 255), (255, 0, 0), (0, 150, 0), (200, 0, 200), (0, 140, 255), (140, 70, 0)]
    for n, a in enumerate(axes):
        col = (150, 150, 150) if a.get("alt_of") else palette[n % len(palette)]
        pos = a["pos"]
        for t in a["ticks"]:
            p = t["px"]
            c = (int(round(pos)), int(round(p))) if a["orient"] == "v" else (int(round(p)), int(round(pos)))
            cv2.drawMarker(o, c, col, cv2.MARKER_CROSS, 22, 1)
            # the calibrated tick position on the frame edge, so snapping can be judged
            if a["orient"] == "v":
                xa = x0 if a["side"] == "left" else x1
                cv2.line(o, (xa - 5, c[1]), (xa + 5, c[1]), col, 1)
            else:
                cv2.line(o, (c[0], y1 - 5), (c[0], y1 + 5), col, 1)
        top = min(a["ticks"], key=lambda t: t["px"]) if a["orient"] == "v" else max(a["ticks"], key=lambda t: t["px"])
        if a["orient"] == "v":
            org = (int(pos) - 10, int(top["px"]) - 14)
        else:
            org = (int(top["px"]) + 14, int(pos) + 5)
        tag = a["id"] + (f"={a['alt_of']}" if a.get("alt_of") else "")
        cv2.putText(o, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(o, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    # colour swatches bottom-left
    H = o.shape[0]
    pad = np.full((34, o.shape[1], 3), 255, np.uint8)
    for i, c in enumerate(colors):
        bx = 5 + i * 115
        cv2.rectangle(pad, (bx, 5), (bx + 20, 25), tuple(int(v) for v in c["rgb"][::-1]), -1)
        cv2.putText(pad, c["hex"], (bx + 24, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([o, pad]))


# ----------------------------------------------------------------------------- draft config
def pick_axis(axes, role_spec):
    """Resolve 'left:0' / 'right:1' / 'bottom:0' (side, ordinal outward from the frame)."""
    side, k = role_spec.split(":")
    cand = [a for a in axes if a["side"] == side and not a.get("alt_of")]
    return cand[int(k)] if int(k) < len(cand) else None


def apply_template(cfg, axes, tpl):
    """Fill roles, units, colours and meta defaults from a layout template."""
    ticks = lambda a: [[round(t["px"], 2), t["value"]] for t in a["ticks"]]
    missing = []
    for role, spec in tpl["roles"].items():
        a = pick_axis(axes, spec["axis"])
        if a is None:
            missing.append(f"{role} ({spec['axis']})")
            continue
        entry = {"ocr_id": a["id"], "ticks": ticks(a), "unit": spec["unit"]}
        if role == "x":
            cfg["x_axis"] = entry
        else:
            cfg["y_axes"][role] = entry
    cfg["y_axes"] = {k: v for k, v in cfg["y_axes"].items() if k in tpl["roles"]}
    cfg["curves"] = {k: dict(v) for k, v in tpl["curves"].items()}
    cfg["fit"] = dict(tpl.get("fit", cfg["fit"]))
    cfg["meta"].update(tpl.get("meta", {}))
    cfg["exclude_boxes"] += tpl.get("exclude_boxes", [])
    cfg["_template"] = tpl.get("name")
    cfg["_template_missing"] = missing
    return cfg


def draft_config(sheet_png, frame, axes, colors):
    """Draft sheet.json. Axis *roles* and units are guesses that must be confirmed."""
    ticks = lambda a: [[round(t["px"], 2), t["value"]] for t in a["ticks"]]
    horiz = sorted([a for a in axes if a["orient"] == "h"], key=lambda a: -len(a["ticks"]))
    vert = [a for a in axes if a["orient"] == "v"]
    cfg = {
        "image": "sheet.png",
        "frame": frame,
        "exclude_boxes": [],
        "x_axis": {"ocr_id": horiz[0]["id"], "ticks": ticks(horiz[0]), "unit": "bpd"} if horiz else
                  {"ticks": [], "unit": "bpd"},
        "y_axes": {},
        "curves": {},
        "fit": {"degree": 5, "sg": 1.0},
        "meta": {"manufacturer": None, "model": None, "base_frequency_hz": None, "catalog_source": None,
                 "catalog_version": None, "series": None, "pump_od_in": None, "stage_type": None,
                 "construction": None, "rpm": None, "stages_basis": 1,
                 "ror_min_bpd": None, "bep_bpd": None, "ror_max_bpd": None,
                 "shaft_limit_hp": None, "housing_burst_psi": None,
                 "freq_min_hz": 30, "freq_max_hz": 80, "freq_step_hz": 10, "notes": ""},
        "_candidates": {a["id"]: {"side": a["side"], "range": a["value_range"], "ticks": ticks(a)}
                        for a in axes},
        "_colors": [c["hex"] for c in colors],
        "_todo": "Confirm roles/units against review.png: copy the right candidate ticks into "
                 "y_axes.head/power/eff, set curve colours, fill meta from the sheet text, "
                 "then delete the _ keys (optional).",
    }
    # heuristic role guess: left axis -> head; right axes: small max -> power, 0..100 -> eff
    left = [a for a in vert if a["side"] == "left"]
    right = [a for a in vert if a["side"] == "right"]
    if left:
        h = max(left, key=lambda a: len(a["ticks"]))
        cfg["y_axes"]["head"] = {"ocr_id": h["id"], "ticks": ticks(h), "unit": "ft"}
    for a in right + [l for l in left if cfg["y_axes"].get("head", {}).get("ocr_id") != l["id"]]:
        hi = a["value_range"][1]
        role = "eff" if 20 <= hi <= 100 and "eff" not in cfg["y_axes"] else \
            "power" if "power" not in cfg["y_axes"] and hi < 20 else None
        if role:
            cfg["y_axes"][role] = {"ocr_id": a["id"], "ticks": ticks(a), "unit": "%" if role == "eff" else "hp"}
    for role in cfg["y_axes"]:
        cfg["curves"][role] = {"color": None, "tol": 35, "style": "solid"}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--template", help="layout template JSON (roles, units, colours, meta defaults)")
    args = ap.parse_args()
    run_inspect(args.sheet, args.out, args.page, args.dpi, args.template, verbose=True)


def label_height(img, preview_w=1000):
    """Median numeric-label height in px, measured on a <=1000 px wide preview (fast, and
    keeps Tesseract within its image-size limit on large renders)."""
    f = min(1.0, preview_w / img.shape[1])
    small = img if f == 1.0 else cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    toks = [t for t in ocr_numbers(small) if any(ch.isdigit() for ch in t["text"])]
    if not toks:
        return 10.0
    return float(np.median([t["bottom"] - t["top"] for t in toks])) / f


def is_raster_pdf_page(path, page):
    import pymupdf
    with pymupdf.open(path) as d:
        pg = d[page]
        return len(pg.get_images()) == 1 and not pg.get_text().strip()


def run_inspect(sheet, out, page=0, dpi=200, template=None, verbose=False):
    """Inspect one sheet; returns (draft_config, axes). Used by main() and batch.py."""
    from common import load_json
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img = load_sheet(sheet, page, dpi)
    # normalise scale: tolerances are tuned for tick labels ~10 px tall. The first estimate
    # can be poor on very large renders, so measure -> rescale -> re-measure (max 3 rounds).
    scale, h, raster = 1.0, 10.0, Path(sheet).suffix.lower() != ".pdf" or is_raster_pdf_page(sheet, page)
    for _ in range(3):
        h = label_height(img)
        if 7.0 <= h <= 13.0:
            break
        f = 10.5 / h
        scale *= f
        if not raster:                       # vector PDF: re-render crisp at the new DPI
            dpi = max(50, int(round(dpi * f)))
            img = load_sheet(sheet, page, dpi)
        else:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    sheet_png = out / "sheet.png"
    cv2.imwrite(str(sheet_png), img)
    frame, lines = detect_frame(img)
    axes = find_axes(img, frame, lines)
    colors = curve_colors(img, frame)
    labels = text_labels(img, frame)
    save_json({"source": str(sheet), "page": page, "scale_applied": scale, "label_height_px": h, "dpi": dpi, "size": [img.shape[1], img.shape[0]],
               "frame": frame, "axes": axes, "colors": colors, "labels": labels}, out / "inspect.json")
    review_image(img, frame, axes, colors, out / "review.png", labels)
    draft = draft_config(sheet_png, frame, axes, colors)
    draft["exclude_boxes"] = [l["box"] for l in labels]
    draft["_labels"] = [l["text"] for l in labels]
    if template:
        apply_template(draft, axes, load_json(template))
    save_json(draft, out / "sheet.draft.json")
    if not verbose:
        return draft, axes
    # console summary for the agent
    print(f"frame {frame}")
    for a in axes:
        print(f"axis {a['id']:>3} {a['side']:<6} range {a['value_range'][0]:g}..{a['value_range'][1]:g}"
              f"  n={len(a['ticks'])} max_resid={a['max_resid_px']:.1f}px snapped={a.get('snapped', 0)}"
              + (f"(offset {a['snap_offset_px']:+.1f}px)" if a.get('snapped') else "")
              + f" rejected={a['rejected']}" + (f" ALT-UNITS-OF {a['alt_of']}" if a.get('alt_of') else "")
              + (f" repaired={a['repaired']}" if a['repaired'] else ""))
    print("labels", [l["text"] for l in labels])
    print("colors", ", ".join(f"{c['hex']}({c['kind']}, n={c['pixels']})" for c in colors))
    if template:
        print("template", draft.get("_template"), "missing roles:", draft.get("_template_missing") or "none")
    return draft, axes


if __name__ == "__main__":
    main()
