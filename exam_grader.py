# -*- coding: utf-8 -*-
"""ExamVision V5 - robust OMR grading engine.

Design goals:
- Never silently shift question numbers when a row is missing.
- Support 2-5 choices and 1-4 physical question columns.
- Detect blank, single-mark, multi-mark, ambiguous and not-detected rows.
- Use page rectification, Hough + contour candidates, robust grid geometry,
  per-row local alignment and multi-feature bubble scoring.
- Be conservative: uncertain marks are sent to review instead of guessed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LETTERS = "ABCDE"
PERSIAN_LETTER_MAP = {
    "الف": "A", "ا": "A", "ب": "B", "ج": "C", "د": "D",
    "ه": "E", "هـ": "E",
}


@dataclass
class OptionScore:
    label: str
    score: float
    present: bool
    dark_ratio: float = 0.0
    very_dark_ratio: float = 0.0
    contrast: float = 0.0
    fill_ratio: float = 0.0

    def to_dict(self):
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "present": self.present,
            "dark_ratio": round(self.dark_ratio, 4),
            "very_dark_ratio": round(self.very_dark_ratio, 4),
            "contrast": round(self.contrast, 4),
            "fill_ratio": round(self.fill_ratio, 4),
        }


@dataclass
class QuestionDetection:
    question: int
    answer: str
    confidence: float
    detected: bool
    status_code: str = "blank"
    options: List[OptionScore] = field(default_factory=list)
    multi_marks: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 1),
            "detected": self.detected,
            "status_code": self.status_code,
            "multi_marks": self.multi_marks,
            "options": [o.to_dict() for o in self.options],
        }


class GradingError(ValueError):
    pass


class ExamGrader:
    """Conservative OMR grader for photographed/scanned answer sheets."""

    MARK_FLOOR = 0.145
    STRONG_MARK_FLOOR = 0.205
    MULTI_MARGIN = 0.045
    MAX_MARKED = 2

    def __init__(
        self,
        image_path: str,
        answer_key_text: str,
        option_count: int = 4,
        columns: int = 1,
        accept_threshold: float = 0.35,
        review_threshold: float = 65.0,
    ):
        self.image_path = str(image_path)
        self.option_count = int(option_count)
        if not 2 <= self.option_count <= 5:
            raise GradingError("تعداد گزینه باید بین ۲ تا ۵ باشد.")
        self.columns = max(1, int(columns))
        self.accept_threshold = float(accept_threshold)
        self.review_threshold = float(review_threshold)
        self.answer_key = self.parse_answer_key(answer_key_text)
        if not self.answer_key:
            raise GradingError("پاسخ‌نامه معتبر نیست. نمونه: 1:A یا 2:ب")
        self.original = None
        self.image = None
        self.gray = None
        self.binary = None
        self.bg_level = 235.0
        self.candidates: List[dict] = []
        self.warnings: List[str] = []
        self.detections: Dict[int, QuestionDetection] = {}
        self.results: Dict[int, dict] = {}
        self.summary: dict = {}
        self.annotated = None
        self.grid_debug = []
        self.image_quality = {}

    # ------------------------------------------------------------------
    # Answer key
    # ------------------------------------------------------------------
    @staticmethod
    def parse_answer_key(text: str) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if not text:
            return out
        for raw in text.splitlines():
            line = raw.strip().upper().replace("ي", "ی")
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[:=\.\-]?\s*([A-E])$", line)
            if m:
                out[int(m.group(1))] = m.group(2)
                continue
            m = re.match(r"^(\d+)\s*[:=\.\-]?\s*(الف|ا|ب|ج|د|ه|هـ)$", line, re.I)
            if m:
                out[int(m.group(1))] = PERSIAN_LETTER_MAP[m.group(2)]
        return dict(sorted(out.items()))

    # ------------------------------------------------------------------
    # Image loading / rectification / quality
    # ------------------------------------------------------------------
    def load(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise GradingError("تصویر قابل خواندن نیست.")
        if min(img.shape[:2]) < 300:
            raise GradingError("رزولوشن تصویر خیلی پایین است.")
        self.original = img.copy()
        self.image = img.copy()
        return img

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float32)
        s = p.sum(axis=1)
        d = np.diff(p, axis=1).reshape(-1)
        return np.array([
            p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]
        ], dtype=np.float32)

    def rectify(self) -> bool:
        img = self.image
        h, w = img.shape[:2]
        scale = min(1.0, 1400.0 / max(h, w))
        small = cv2.resize(img, None, fx=scale, fy=scale)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(g, 35, 150)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page = None
        image_area = small.shape[0] * small.shape[1]
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
            area = cv2.contourArea(c)
            if area < image_area * 0.30:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            if len(approx) == 4:
                page = approx.reshape(4, 2).astype(np.float32)
                break
        if page is None:
            # Fallback for slightly rotated sheets whose outer contour is broken.
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(80, int(min(small.shape[:2])*.22)),
                                    minLineLength=max(120, int(max(small.shape[:2])*.45)), maxLineGap=35)
            angles=[]
            if lines is not None:
                for ln in lines[:,0]:
                    x1,y1,x2,y2=map(float,ln)
                    length=math.hypot(x2-x1,y2-y1)
                    ang=math.degrees(math.atan2(y2-y1,x2-x1))
                    if length >= max(120, max(small.shape[:2])*.45):
                        if abs(ang) <= 20 or abs(abs(ang)-180) <= 20:
                            angles.append(ang if abs(ang)<=20 else ang-180 if ang>0 else ang+180)
            if angles:
                angle=float(np.median(angles))
                if abs(angle)>0.35:
                    M=cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
                    self.image=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderValue=(248,248,248))
                    self.warnings.append(f"ℹ️ چرخش جزئی برگه ({angle:.1f}°) اصلاح شد.")
                    return True
            self.warnings.append("مرز برگه پیدا نشد؛ تصویر بدون اصلاح پرسپکتیو استفاده شد.")
            return False

        page = self._order_points(page / scale)
        tl, tr, br, bl = page
        W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if W < 500 or H < 500:
            self.warnings.append("مرز برگه خیلی کوچک بود؛ اصلاح پرسپکتیو انجام نشد.")
            return False
        W = int(np.clip(W, 800, 3200))
        H = int(np.clip(H, 1000, 4200))
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        M = cv2.getPerspectiveTransform(page, dst)
        warped = cv2.warpPerspective(img, M, (W, H))
        if warped is None or warped.size == 0:
            self.warnings.append("اصلاح پرسپکتیو ناموفق بود؛ تصویر اصلی استفاده شد.")
            return False
        self.image = warped
        return True

    def _quality_check(self):
        g = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(g))
        std = float(np.std(g))
        lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
        h, w = g.shape
        # These are warnings rather than hard failures because some valid scans
        # are intentionally soft or low-contrast.
        resolution_score = min(100.0, 100.0 * min(h, w) / 1000.0)
        contrast_score = min(100.0, max(0.0, std * 3.0))
        blur_score = min(100.0, max(0.0, math.log1p(lap_var) * 9.0))
        brightness_score = max(0.0, 100.0 - abs(mean - 190.0) * 0.75)
        overall = 0.30 * resolution_score + 0.30 * contrast_score + 0.25 * blur_score + 0.15 * brightness_score
        self.image_quality = {
            "width": w, "height": h, "mean_gray": round(mean, 1),
            "contrast": round(contrast_score, 1), "sharpness": round(blur_score, 1),
            "brightness": round(brightness_score, 1), "overall": round(overall, 1),
        }
        if min(h, w) < 700:
            self.warnings.append("⚠️ رزولوشن تصویر پایین است؛ اطمینان تشخیص ممکن است کاهش یابد.")
        if blur_score < 30:
            self.warnings.append("⚠️ تصویر کمی تار است؛ پاسخ‌های کم‌رنگ ممکن است نیاز به بازبینی داشته باشند.")
        if contrast_score < 20:
            self.warnings.append("⚠️ کنتراست تصویر پایین است.")

    def preprocess(self):
        g = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        g = clahe.apply(g)
        self.gray = cv2.GaussianBlur(g, (3, 3), 0)
        self.binary = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 7
        )
        # 80th/85th percentile works better than assuming a pure white page.
        self.bg_level = float(np.percentile(self.gray, 85))
        return self.gray, self.binary

    # ------------------------------------------------------------------
    # Candidate detection
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate(cx, cy, r, source, circularity=0.8):
        return {
            "cx": float(cx), "cy": float(cy), "r": float(r),
            "source": source, "circularity": float(circularity),
        }

    def detect_candidates(self):
        g, b = self.preprocess()
        H, W = g.shape
        min_dim = min(H, W)
        base_min = max(4, int(min_dim * 0.0035))
        base_max = max(base_min + 5, int(min_dim * 0.026))
        # First pass: conservative Hough to estimate the real bubble radius.
        seed = cv2.HoughCircles(
            g, cv2.HOUGH_GRADIENT, dp=1.15,
            minDist=max(10, int(min_dim * 0.010)),
            param1=90, param2=24,
            minRadius=base_min, maxRadius=base_max,
        )
        seed_arr = np.round(seed[0]).astype(int) if seed is not None else np.empty((0, 3), int)
        if len(seed_arr):
            med_r = float(np.median(seed_arr[:, 2]))
            rlo = max(3.0, med_r * 0.68)
            rhi = min(base_max, med_r * 1.42)
        else:
            med_r = max(4.0, min_dim * 0.010)
            rlo, rhi = max(3.0, med_r * .65), min(base_max, med_r * 1.5)

        cands = []
        for p2 in (24, 20, 17, 14):
            circles = cv2.HoughCircles(
                g, cv2.HOUGH_GRADIENT, dp=1.15,
                minDist=max(10, int(med_r * 1.35)),
                param1=90, param2=p2,
                minRadius=max(3, int(rlo)), maxRadius=max(int(rlo)+2, int(rhi)),
            )
            if circles is not None:
                for x, y, r in np.round(circles[0]).astype(int):
                    if 0 <= x < W and 0 <= y < H:
                        cands.append(self._candidate(x, y, r, 'hough', .85))

        # Contours are restricted to the learned bubble size, preventing text
        # and decorative shapes from dominating the geometry stage.
        contours, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_img = H * W
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_img * 0.00001 or area > area_img * 0.0015:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) < max(5, int(rlo * 1.1)) or max(w, h) > int(rhi * 2.2):
                continue
            ar = w / max(h, 1)
            if not .72 <= ar <= 1.38:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * math.pi * area / (peri * peri)
            if circ < .48:
                continue
            M = cv2.moments(c)
            if abs(M['m00']) < 1e-6:
                continue
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            r = min(w, h) * .45
            if rlo * .55 <= r <= rhi * 1.15:
                cands.append(self._candidate(cx, cy, r, 'contour', circ))

        self.candidates = self._dedup(cands)
        self._filter_size_consistency()
        return self.candidates

    @staticmethod
    def _dedup(cands):
        if not cands:
            return []
        out = []
        order = sorted(cands, key=lambda c: (
            0 if c["source"] == "hough" else 1,
            -c["circularity"], -c["r"]
        ))
        for c in order:
            if not any(
                math.hypot(c["cx"] - x["cx"], c["cy"] - x["cy"])
                < max(5.0, min(c["r"], x["r"]) * 0.72)
                for x in out
            ):
                out.append(c)
        return out

    def _filter_size_consistency(self):
        if len(self.candidates) < 8:
            return
        rs = np.array([c["r"] for c in self.candidates], dtype=float)
        med = float(np.median(rs))
        mad = float(np.median(np.abs(rs - med))) + 1e-6
        lo = max(med * 0.52, med - 4.5 * mad)
        hi = min(med * 1.90, med + 4.5 * mad)
        self.candidates = [c for c in self.candidates if lo <= c["r"] <= hi]

    # ------------------------------------------------------------------
    # Rows / geometry
    # ------------------------------------------------------------------
    def _cluster_rows(self, candidates=None):
        c = sorted(candidates if candidates is not None else self.candidates,
                   key=lambda z: (z["cy"], z["cx"]))
        if not c:
            return []
        med_r = float(np.median([x["r"] for x in c]))
        tol = max(7.0, med_r * 1.35)
        rows = []
        for item in c:
            best = None
            bd = 1e9
            for i, row in enumerate(rows):
                d = abs(item["cy"] - row["y"])
                if d <= tol and d < bd:
                    best, bd = i, d
            if best is None:
                rows.append({"y": item["cy"], "items": [item]})
            else:
                rows[best]["items"].append(item)
                rows[best]["y"] = float(np.mean([z["cy"] for z in rows[best]["items"]]))
        rows.sort(key=lambda r: r["y"])
        merged = []
        for row in rows:
            if merged and abs(row["y"] - merged[-1]["y"]) < tol * 0.85:
                merged[-1]["items"].extend(row["items"])
                merged[-1]["y"] = float(np.mean([z["cy"] for z in merged[-1]["items"]]))
            else:
                merged.append(row)
        return [self._merge_close_x(sorted(r["items"], key=lambda z: z["cx"]))
                for r in merged if r["items"]]

    @staticmethod
    def _merge_close_x(items):
        if len(items) < 2:
            return items
        med_r = float(np.median([x["r"] for x in items]))
        min_gap = max(9.0, med_r * 1.55)
        out = [items[0]]
        for it in items[1:]:
            prev = out[-1]
            if it["cx"] - prev["cx"] < min_gap:
                # Hough is normally more centered than a contour fragment.
                if it["source"] == "hough" and prev["source"] != "hough":
                    out[-1] = it
                elif it["source"] == prev["source"] and it["r"] > prev["r"]:
                    out[-1] = it
            else:
                out.append(it)
        return out

    @staticmethod
    def _regularity(xs):
        if len(xs) < 2:
            return 999.0
        gaps = np.diff(np.sort(xs))
        mean = float(np.mean(gaps))
        return float(np.std(gaps) / mean) if mean > 1e-6 else 999.0

    def _best_n_subset(self, xs, n):
        xs = np.array(sorted(xs), dtype=float)
        if len(xs) < n:
            return None
        if len(xs) == n:
            return xs
        if len(xs) <= 14:
            best, score = None, 1e9
            for ids in combinations(range(len(xs)), n):
                a = xs[list(ids)]
                sc = self._regularity(a)
                if sc < score:
                    best, score = a, sc
            return best

        # Fast lattice approximation for very noisy rows.
        gaps = np.diff(xs)
        pitch = float(np.median(gaps))
        if pitch <= 0:
            return None
        best = None
        best_score = 1e9
        for start in range(min(len(xs), 8)):
            target = xs[start] + np.arange(n) * pitch
            chosen = []
            used = set()
            err = 0.0
            for t in target:
                j = min((j for j in range(len(xs)) if j not in used),
                        key=lambda j: abs(xs[j] - t))
                used.add(j)
                chosen.append(xs[j])
                err += abs(xs[j] - t) / max(pitch, 1.0)
            if self._regularity(chosen) < 0.30 and err < best_score:
                best, best_score = np.array(sorted(chosen)), err
        return best

    def _find_anchor_rows(self, rows):
        n = self.option_count
        anchors = []
        for r in rows:
            items = r
            if len(items) > n:
                med_r = float(np.median([x["r"] for x in items]))
                items = [x for x in items if 0.55 * med_r <= x["r"] <= 1.75 * med_r]
            xs = [x["cx"] for x in items]
            if len(xs) < n:
                continue
            subset = self._best_n_subset(xs, n)
            if subset is None:
                continue
            residual = self._regularity(subset)
            if residual <= 0.24:
                anchors.append({
                    "y": float(np.mean([x["cy"] for x in r])),
                    "template": subset,
                    "items": r,
                    "regularity": residual,
                })
        return anchors

    def _estimate_pitch(self, anchors):
        if len(anchors) < 2:
            return None
        ys = np.array(sorted(a["y"] for a in anchors), dtype=float)
        med_r = float(np.median([it["r"] for a in anchors for it in a["items"]]))
        min_pitch = max(2.8 * med_r, 24.0)
        candidates = []
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                gap = ys[j] - ys[i]
                if gap < min_pitch:
                    continue
                for d in range(1, min(10, int(gap / min_pitch)) + 1):
                    p = gap / d
                    if p >= min_pitch:
                        candidates.append(p)
        if not candidates:
            return None
        best_p, best_score = None, -1e9
        for p in candidates:
            residual = np.abs((ys[:, None] - ys[None, :]) / p)
            frac = np.abs(residual - np.round(residual))
            closeness = np.mean(np.min(frac + np.eye(len(ys)) * 10, axis=1))
            count = np.sum(np.min(frac + np.eye(len(ys)) * 10, axis=1) < 0.10)
            score = count * 10.0 - closeness
            if score > best_score:
                best_score, best_p = score, p
        return float(best_p)

    def _assign_anchor_indices(self, anchors, n_questions, pitch):
        """Find the integer row offset without assuming the last anchor is Qn.

        We test every plausible intercept generated by every anchor and every
        possible question index. The winner maximizes the number of anchors
        that land on the same integer grid while penalizing large residuals.
        """
        if not anchors or pitch is None:
            return None
        ys = np.array([a["y"] for a in anchors], dtype=float)
        best = None
        tol = max(5.0, pitch * 0.16)
        for y0 in ys:
            for q0 in range(n_questions):
                intercept = y0 - q0 * pitch
                q_float = (ys - intercept) / pitch
                q_int = np.rint(q_float).astype(int)
                valid = (q_int >= 0) & (q_int < n_questions)
                residual = np.abs(ys - (intercept + q_int * pitch))
                inlier = valid & (residual <= tol)
                count = int(np.sum(inlier))
                if count == 0:
                    continue
                # Prefer more matched rows, then smaller residual, then wider span.
                mean_res = float(np.mean(residual[inlier]))
                span = int(q_int[inlier].max() - q_int[inlier].min()) if count > 1 else 0
                score = count * 100.0 + span * 0.8 - mean_res * 2.0
                if best is None or score > best["score"]:
                    best = {
                        "intercept": float(intercept),
                        "indices": q_int,
                        "inlier": inlier,
                        "score": score,
                    }
        return best

    def _estimate_bubble_pitch_x(self):
        xs = np.array([c['cx'] for c in self.candidates], dtype=float)
        if len(xs) < self.option_count * 2:
            return None
        lo = max(3.0, float(np.median([c['r'] for c in self.candidates])) * 2.2)
        hi = max(lo + 5.0, min(self.gray.shape[1] * .16, lo * 7.0))
        diffs = []
        for i in range(len(xs)):
            d = np.abs(xs[i+1:] - xs[i])
            diffs.extend(d[(d >= lo) & (d <= hi)])
        if not diffs:
            return None
        hist, edges = np.histogram(diffs, bins=np.arange(lo, hi + 1.0, 1.0))
        if not len(hist):
            return None
        return float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist))+1]) / 2.0)

    def _find_column_templates(self, n_columns):
        """Find physical option lattices directly from X support.
        This replaces global K-means, which is easily hijacked by text/decoys.
        """
        if not self.candidates:
            return []
        xs = np.array([c['cx'] for c in self.candidates], dtype=float)
        rs = np.array([c['r'] for c in self.candidates], dtype=float)
        pitch0 = self._estimate_bubble_pitch_x()
        if pitch0 is None:
            pitch0 = max(8.0, float(np.median(rs) * 4.2))
        pitches = np.linspace(max(pitch0*.88, pitch0-8), pitch0*1.12, 19)
        tol = max(5.0, float(np.median(rs) * .95))
        candidates = []
        for p in pitches:
            for x0 in xs:
                counts=[]; med_err=0.0
                for k in range(self.option_count):
                    ds=np.abs(xs-(x0+k*p))
                    near=ds[ds<=tol]
                    counts.append(len(near))
                    if len(near): med_err += float(np.median(near))
                support=sum(min(c, 30) for c in counts)
                balance=np.std(counts)
                if support >= self.option_count*4:
                    # Require every option slot to have evidence.
                    score=support - .30*balance - .20*med_err
                    candidates.append((score,x0,p,counts))
        candidates.sort(reverse=True, key=lambda z:z[0])
        selected=[]
        for score,x0,p,counts in candidates:
            raw=np.array([x0+k*p for k in range(self.option_count)],float)
            # refine slot centers using nearby candidates
            refined=[]
            for tx in raw:
                near=xs[np.abs(xs-tx)<=tol]
                refined.append(float(np.median(near)) if len(near) else float(tx))
            refined=np.array(refined)
            center=float(np.mean(refined))
            if any(abs(center-other['center']) < p*self.option_count*.55 for other in selected):
                continue
            selected.append({'template':refined,'pitch':p,'center':center,'score':score})
            if len(selected)>=n_columns:
                break
        selected.sort(key=lambda z:z['center'])
        return selected

    def _solve_rows_from_template(self, template, n_questions):
        """Fit Y lattice from per-option evidence, allowing missing rows."""
        if not self.candidates or n_questions <= 0:
            return [None]*n_questions, ['❗ هیچ candidateای برای شبکه پیدا نشد.']
        xs=np.asarray(template, dtype=float)
        rs=np.array([c['r'] for c in self.candidates],float)
        tol_x=max(5.0,float(np.median(rs)*1.05))
        by_slot=[[] for _ in range(self.option_count)]
        for c in self.candidates:
            k=int(np.argmin(np.abs(xs-c['cx'])))
            if abs(xs[k]-c['cx'])<=tol_x:
                by_slot[k].append(c)
        ys=np.array([c['cy'] for slot in by_slot for c in slot],float)
        if len(ys)<self.option_count*2:
            return [None]*n_questions, ['❗ شواهد کافی برای شبکه سؤال‌ها پیدا نشد.']
        qspan=max(n_questions-1,1)
        p10=float(np.percentile(ys,5)); p90=float(np.percentile(ys,95))
        p_est=(p90-p10)/qspan
        p_est=max(float(np.median(rs)*3.0),p_est)
        p_min=max(float(np.median(rs)*2.7),p_est*.70)
        p_max=min(self.gray.shape[0]/max(1,(n_questions-1)*.42),p_est*1.30)
        if p_max<=p_min: p_max=p_min+max(8,p_est*.2)
        pitches=np.linspace(p_min,p_max,31)
        tol_y=max(5.0,float(np.median(rs)*1.35))
        best=None
        for pitch in pitches:
            for y0 in ys:
                coverage=[]; residual=0.0
                for q in range(n_questions):
                    ey=y0+q*pitch; cov=0; err=0.0
                    for slot in by_slot:
                        if not slot: continue
                        ds=np.abs(np.array([c['cy'] for c in slot])-ey)
                        d=float(np.min(ds))
                        if d<=tol_y:
                            cov+=1; err+=d
                    coverage.append(cov); residual+=err
                strong=sum(1 for c in coverage if c>=max(3, int(math.ceil(self.option_count*.60))))
                total=sum(coverage)
                edge_penalty=abs(y0-p10) + abs((y0+(n_questions-1)*pitch)-p90)
                score=strong*100 + total*3 - residual*.25 - edge_penalty*1.25
                if best is None or score>best[0]:
                    best=(score,y0,pitch,coverage)
        _,y0,pitch,coverage=best
        # Refine the linear model from row centers with >=60% option support.
        qids=[]; centers=[]
        for q in range(n_questions):
            ey=y0+q*pitch; vals=[]
            for slot in by_slot:
                if not slot: continue
                ds=np.abs(np.array([c['cy'] for c in slot])-ey); j=int(np.argmin(ds))
                if ds[j]<=tol_y: vals.append(float(slot[j]['cy']))
            if len(vals)>=max(3,int(math.ceil(self.option_count*.60))):
                qids.append(q); centers.append(float(np.median(vals)))
        if len(qids)>=2:
            coef=np.polyfit(np.asarray(qids,float),np.asarray(centers,float),1)
            pitch=float(coef[0]); y0=float(coef[1])
        grid=[]; notes=[]; threshold=max(3,int(math.ceil(self.option_count*.60)))
        for q in range(n_questions):
            ey=y0+q*pitch; slots=[]; used=set()
            for slot in by_slot:
                if not slot: slots.append(None); continue
                best_i=None; best_d=1e9
                for i,c in enumerate(slot):
                    if i in used: continue
                    d=abs(c['cy']-ey)
                    if d<best_d and d<=tol_y*1.15:
                        best_i,best_d=i,d
                if best_i is None: slots.append(None)
                else: used.add(best_i); slots.append(slot[best_i])
            present=sum(s is not None for s in slots)
            if present<threshold:
                grid.append(None); notes.append(f'❗ سؤال {q+1} شناسایی نشد (شواهد کافی نبود).')
            else:
                # Once the row geometry is trusted, score the ideal lattice
                # positions as well. A missed Hough circle must not erase a
                # real student mark from consideration.
                r_ref=float(np.median(rs))
                ideal_slots=[{'cx':float(x),'cy':float(ey),'r':r_ref} for x in xs]
                actual_vals=[s['cy'] for s in slots if s is not None]
                actual_y=float(np.mean(actual_vals)) if actual_vals else float(ey)
                grid.append({'y':float(ey),'actual_y':actual_y,'slots':ideal_slots})
        return grid,notes

    @staticmethod
    def _column_counts(n_questions, n_columns):
        n_columns = min(max(1, n_columns), n_questions)
        base, rem = divmod(n_questions, n_columns)
        return [base + (1 if i < rem else 0) for i in range(n_columns)]

    def _split_into_channels(self, n_columns):
        if n_columns <= 1:
            return [self.candidates]
        n = self.option_count
        # Use X-clustering only to find physical bubble groups. Rows are later
        # solved independently, so a few outliers cannot directly shift Qs.
        xs = np.array([c["cx"] for c in self.candidates], dtype=np.float32)
        K = n_columns * n
        if len(xs) >= K and len(xs) >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.25)
            try:
                _, _, centers = cv2.kmeans(
                    xs.reshape(-1, 1), K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
                )
                centers = np.sort(centers.flatten())
                gaps = np.diff(centers)
                split_ids = sorted(np.argsort(gaps)[-(n_columns - 1):])
                boundaries = [(centers[i] + centers[i + 1]) / 2 for i in split_ids]
            except cv2.error:
                boundaries = []
        else:
            boundaries = []
        if not boundaries:
            W = self.gray.shape[1]
            boundaries = [W * i / n_columns for i in range(1, n_columns)]
        channels = [[] for _ in range(n_columns)]
        for c in self.candidates:
            idx = int(np.searchsorted(boundaries, c["cx"], side="right"))
            idx = max(0, min(n_columns - 1, idx))
            channels[idx].append(c)
        return channels

    def _resolve_row_slots(self, row_items, template):
        n = len(template)
        if not row_items:
            return [None] * n
        items = sorted(row_items, key=lambda z: z["cx"])
        if len(items) >= n:
            subset = self._best_n_subset([x["cx"] for x in items], n)
            if subset is not None:
                chosen = []
                remaining = list(items)
                for tx in subset:
                    j = int(np.argmin([abs(z["cx"] - tx) for z in remaining]))
                    chosen.append(remaining.pop(j))
                return sorted(chosen, key=lambda z: z["cx"])

        # Missing candidate(s): translate the median template to this row.
        offsets = []
        for item in items:
            nearest = template[int(np.argmin(np.abs(template - item["cx"]))) ]
            offsets.append(item["cx"] - nearest)
        local = template + float(np.median(offsets)) if offsets else template
        used = set()
        med_r = float(np.median([x["r"] for x in items]))
        slots = []
        max_dist = max(13.0, med_r * 2.35)
        for tx in local:
            best_i, best_d = None, 1e9
            for i, item in enumerate(items):
                if i in used:
                    continue
                d = abs(item["cx"] - tx)
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None and best_d <= max_dist:
                used.add(best_i)
                slots.append(items[best_i])
            else:
                slots.append(None)
        return slots

    def _build_grid_single_channel(self, rows, n_questions, label=""):
        anchors = self._find_anchor_rows(rows)
        notes = []
        template = None
        if anchors:
            template = np.median(np.vstack([a["template"] for a in anchors]), axis=0)
        elif rows:
            all_xs = [x["cx"] for r in rows for x in r]
            template = np.linspace(min(all_xs), max(all_xs), self.option_count)
        else:
            template = np.linspace(0, 1, self.option_count)

        pitch = self._estimate_pitch(anchors)
        assignment = self._assign_anchor_indices(anchors, n_questions, pitch) if pitch else None

        # Fallback only when geometry genuinely cannot be established.
        if assignment is None or pitch is None:
            if rows:
                notes.append(f"⚠️{label} شبکه‌ی منظم سؤال‌ها با اطمینان کافی پیدا نشد؛ فقط ردیف‌های مستقیم بررسی شدند.")
            grid = []
            for i in range(n_questions):
                r = rows[i] if i < len(rows) else None
                if r is None:
                    notes.append(f"❗{label} سؤال {i + 1} شناسایی نشد.")
                    grid.append(None)
                else:
                    grid.append({"y": float(np.mean([x["cy"] for x in r])),
                                 "slots": self._resolve_row_slots(r, template)})
            return grid, notes

        intercept = assignment["intercept"]
        grid: List[Optional[dict]] = []
        all_row_ys = np.array([float(np.mean([x["cy"] for x in r])) for r in rows])
        used_row_indices = set()
        row_tol = max(10.0, pitch * 0.38)
        for q in range(n_questions):
            predicted_y = intercept + pitch * q
            dists = np.abs(all_row_ys - predicted_y)
            if len(dists) == 0:
                grid.append(None)
                notes.append(f"❗{label} سؤال {q + 1} شناسایی نشد.")
                continue
            order = np.argsort(dists)
            best_row_idx = None
            for idx in order:
                if dists[idx] <= row_tol and idx not in used_row_indices:
                    best_row_idx = int(idx)
                    break
            if best_row_idx is None:
                grid.append(None)
                notes.append(f"❗{label} سؤال {q + 1} شناسایی نشد (ردیف مورد انتظار خالی است).")
                continue
            used_row_indices.add(best_row_idx)
            r = rows[best_row_idx]
            grid.append({"y": float(predicted_y), "actual_y": float(all_row_ys[best_row_idx]),
                         "slots": self._resolve_row_slots(r, template)})

        matched = sum(x is not None for x in grid)
        if matched < n_questions:
            notes.append(f"ℹ️{label} {n_questions - matched} ردیف به‌صورت صادقانه شناسایی‌نشده باقی ماند.")
        if len(anchors) > int(np.sum(assignment["inlier"])):
            notes.append(f"ℹ️{label} چند candidate/ردیف مشکوک از مدل هندسی حذف شد.")
        return grid, notes

    def build_grid(self, rows):
        n_questions=len(self.answer_key)
        templates=self._find_column_templates(self.columns)
        notes=[]
        if len(templates)<self.columns:
            notes.append(f"⚠️ فقط {len(templates)} از {self.columns} ستون گزینه با اطمینان پیدا شد.")
        if not templates:
            return [None]*n_questions, notes+["❗ شبکه گزینه‌ها پیدا نشد."]
        counts=self._column_counts(n_questions,len(templates))
        grid=[]
        for i,(info,count) in enumerate(zip(templates,counts),1):
            g,n=self._solve_rows_from_template(info['template'],count)
            grid.extend(g); notes.extend([f" [ستون {i}] {x}" for x in n])
        return grid,notes

    # ------------------------------------------------------------------
    # Bubble scoring
    # ------------------------------------------------------------------
    def bubble_score(self, cx, cy, r):
        H, W = self.gray.shape
        rr = max(4, int(r * 0.88))
        x1, x2 = max(0, int(cx - rr)), min(W, int(cx + rr + 1))
        y1, y2 = max(0, int(cy - rr)), min(H, int(cy + rr + 1))
        roi = self.gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        mx, my = roi.shape[1] / 2.0, roi.shape[0] / 2.0
        rad = min(roi.shape) * 0.47
        mask = (xx - mx) ** 2 + (yy - my) ** 2 <= rad ** 2
        vals = roi[mask]
        if vals.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        inner_mean = float(np.mean(vals))
        dark = float(np.mean(vals < 170))
        very_dark = float(np.mean(vals < 110))
        bg = getattr(self, "bg_level", 235.0)
        contrast = float(np.clip((bg - inner_mean) / 100.0, 0, 1))

        # Filled bubbles have broad dark coverage; a thin stray line does not.
        bw = (vals < 180).astype(np.uint8)
        fill_ratio = float(np.mean(bw))

        # Robust center-vs-ring signal. The printed outline exists on every
        # bubble, so only interior darkening should matter.
        center_mask = ((xx - mx) ** 2 + (yy - my) ** 2 <= (rad * 0.55) ** 2) & mask
        center_vals = roi[center_mask]
        center_dark = float(np.mean(center_vals < 170)) if center_vals.size else dark

        score = (
            0.34 * dark +
            0.22 * very_dark +
            0.22 * contrast +
            0.14 * fill_ratio +
            0.08 * center_dark
        )
        return float(np.clip(score, 0, 1)), dark, very_dark, contrast, fill_ratio

    def score_grid(self, grid):
        detections = {}
        for q_idx, row in enumerate(grid, 1):
            if row is None:
                detections[q_idx] = QuestionDetection(
                    q_idx, "", 0.0, False, status_code="not_detected"
                )
                continue

            scores, options = [], []
            for i, slot in enumerate(row["slots"]):
                label = LETTERS[i]
                if slot is None:
                    scores.append(0.0)
                    options.append(OptionScore(label, 0.0, False))
                    continue
                result = self.bubble_score(slot["cx"], slot["cy"], slot["r"])
                score, dark, very_dark, contrast, fill_ratio = result
                scores.append(score)
                options.append(OptionScore(
                    label, score, True, dark, very_dark, contrast, fill_ratio
                ))

            arr = np.array(scores, dtype=float)
            marked = np.where(arr >= self.MARK_FLOOR)[0].tolist()
            strong = np.where(arr >= self.STRONG_MARK_FLOOR)[0].tolist()

            if len(strong) >= 2:
                # Two genuinely dark choices -> invalid/multi.
                top2 = sorted(strong, key=lambda i: arr[i], reverse=True)[:2]
                if arr[top2[1]] >= arr[top2[0]] - self.MULTI_MARGIN:
                    marks = [LETTERS[i] for i in top2]
                    detections[q_idx] = QuestionDetection(
                        q_idx, "", 100.0, True, status_code="multi_mark",
                        options=options, multi_marks=marks
                    )
                    continue

            if not marked:
                detections[q_idx] = QuestionDetection(
                    q_idx, "", 0.0, True, status_code="blank", options=options
                )
                continue

            # A single weak mark is not automatically accepted.
            best_i = int(np.argmax(arr))
            best = float(arr[best_i])
            rest = np.delete(arr, best_i)
            second = float(rest.max()) if rest.size else 0.0
            separation = float(np.clip((best - second) / max(best, 0.18), 0, 1))
            absolute = float(np.clip((best - self.MARK_FLOOR) / (1.0 - self.MARK_FLOOR), 0, 1))
            confidence = 100.0 * (0.60 * absolute + 0.40 * separation)

            # Ambiguous if the runner-up is also reasonably dark.
            if second >= self.MARK_FLOOR and best - second < self.MULTI_MARGIN:
                marks = [LETTERS[i] for i, s in enumerate(arr) if s >= self.MARK_FLOOR]
                detections[q_idx] = QuestionDetection(
                    q_idx, "", max(0.0, confidence), True,
                    status_code="ambiguous", options=options, multi_marks=marks[:self.MAX_MARKED]
                )
                continue

            answer = LETTERS[best_i] if confidence / 100.0 >= self.accept_threshold else ""
            status_code = "answered" if answer else "ambiguous"
            detections[q_idx] = QuestionDetection(
                q_idx, answer, confidence, True,
                status_code=status_code, options=options
            )

        self.detections = detections
        return detections

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    def grade(self):
        results = {}
        correct = wrong = blank = review = not_detected = 0
        for q, key in self.answer_key.items():
            det = self.detections.get(q)
            student = det.answer if det else ""
            conf = det.confidence if det else 0.0
            detected = det.detected if det else False

            if not detected:
                not_detected += 1
                status = "❓ ردیف شناسایی نشد"
            elif det.status_code == "multi_mark":
                wrong += 1
                status = f"🟠 چند پاسخی/باطل ({'+'.join(det.multi_marks)})"
            elif det.status_code == "ambiguous":
                review += 1
                status = "🔎 مبهم / نیازمند بازبینی"
            elif not student:
                blank += 1
                status = "⚪ بدون پاسخ"
            elif conf < self.review_threshold:
                review += 1
                if student == key:
                    correct += 1
                    status = "✅ صحیح / نیاز به بازبینی"
                else:
                    wrong += 1
                    status = "❌ غلط / نیاز به بازبینی"
            elif student == key:
                correct += 1
                status = "✅ صحیح"
            else:
                wrong += 1
                status = "❌ غلط"

            results[q] = {
                "question": q,
                "student": "+".join(det.multi_marks) if det and det.multi_marks else (student or "—"),
                "correct": key,
                "status": status,
                "confidence": round(conf, 1),
                "detected": detected,
            }

        total = len(self.answer_key)
        graded_total = total - not_detected
        self.results = results
        self.summary = {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "blank": blank,
            "review": review,
            "not_detected": not_detected,
            "percentage": round(100 * correct / total, 2) if total else 0.0,
            "percentage_of_detected": round(100 * correct / graded_total, 2) if graded_total else 0.0,
        }
        return results, self.summary

    # ------------------------------------------------------------------
    # Annotation / outputs
    # ------------------------------------------------------------------
    def annotate(self, grid):
        out = self.image.copy()
        for q, row in enumerate(grid, 1):
            det = self.detections.get(q)
            if row is None:
                continue
            multi = det.multi_marks if det else []
            selected = det.answer if det else ""
            for i, slot in enumerate(row["slots"]):
                if slot is None:
                    continue
                cx, cy, r = slot["cx"], slot["cy"], slot["r"]
                score = det.options[i].score if det and i < len(det.options) else 0.0
                if LETTERS[i] in multi:
                    color = (0, 140, 255)
                elif LETTERS[i] == selected:
                    color = (40, 180, 70)
                elif score >= self.MARK_FLOOR:
                    color = (40, 200, 220)
                else:
                    color = (130, 130, 130)
                cv2.circle(out, (int(cx), int(cy)), max(3, int(r)), color, 2)
                cv2.putText(out, LETTERS[i], (int(cx-r), max(15, int(cy-r-4))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                cv2.putText(out, f"{score:.2f}", (int(cx-r), int(cy+r+15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
            xs = [s["cx"] for s in row["slots"] if s]
            label_x = max(5, int(min(xs, default=10) - 32))
            label_y = max(20, int(row["y"] - 24))
            txt = f"Q{q}: {'+'.join(multi) if multi else (selected or '?')} | {det.confidence:.0f}%"
            cv2.putText(out, txt, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,80,30), 2)
        return out

    def run(self):
        self.load()
        rectified = self.rectify()
        self._quality_check()
        self.detect_candidates()
        rows = self._cluster_rows()
        grid, notes = self.build_grid(rows)
        self.warnings.extend(notes)
        if all(g is None for g in grid):
            raise GradingError("هیچ ردیف گزینه‌ای با اطمینان قابل قبول پیدا نشد. وضوح و نور تصویر را بررسی کنید.")
        self.score_grid(grid)
        self.grade()
        self.annotated = self.annotate(grid)
        return {
            "summary": self.summary,
            "results": self.results,
            "detections": {q: d.to_dict() for q, d in self.detections.items()},
            "rectified": rectified,
            "candidate_count": len(self.candidates),
            "row_count": sum(1 for g in grid if g is not None),
            "warnings": self.warnings,
            "image_quality": self.image_quality,
        }

    def save_outputs(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        img_path = os.path.join(output_dir, "analysis.png")
        cv2.imwrite(img_path, self.annotated)
        report = {
            "app": "ExamVision V5",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": self.summary,
            "results": self.results,
            "warnings": self.warnings,
            "image_quality": self.image_quality,
        }
        json_path = os.path.join(output_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        csv_path = os.path.join(output_dir, "results.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "student", "correct", "status", "confidence", "detected"])
            for q, r in self.results.items():
                w.writerow([q, r["student"], r["correct"], r["status"], r["confidence"], r["detected"]])
        return img_path, json_path, csv_path
