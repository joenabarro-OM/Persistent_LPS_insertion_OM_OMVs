#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python_code_New_Old_LPS_DBSCAN

"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from shapely.affinity import rotate

from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# --------------------------- Defaults (Methods-matching) --------------------------
DEFAULT_EPS_NM = 30.0
DEFAULT_MIN_SAMPLES = 7
DEFAULT_OM_BUFFER_NM = 20.0
DEFAULT_OM_SIMPLIFY_NM = 2.0


@dataclass(frozen=True)
class OverlapOutputs:
    out_dir: str
    summary_csv: str
    overlap_png: str
    old_hist_csv: str
    new_hist_csv: str


def _require_columns(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_xy(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    _require_columns(df, ["Position X [nm]", "Position Y [nm]"], os.path.basename(csv_path))
    return df[["Position X [nm]", "Position Y [nm]"]].to_numpy(dtype=float)


def compute_om_shape(all_xy: np.ndarray,
                     buffer_nm: float = DEFAULT_OM_BUFFER_NM,
                     simplify_nm: float = DEFAULT_OM_SIMPLIFY_NM):
    hull = ConvexHull(all_xy)
    boundary = all_xy[hull.vertices]
    # Expand and smooth (buffer then simplify)
    return MultiPoint(boundary).convex_hull.buffer(buffer_nm).simplify(simplify_nm)


def pca_rotation_deg(all_xy: np.ndarray) -> float:
    centered = all_xy - all_xy.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2).fit(centered)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    return float(np.degrees(-angle))  # clockwise


def rotate_xy(xy: np.ndarray, angle_deg: float, origin_xy: Tuple[float, float]) -> np.ndarray:
    if xy.size == 0:
        return xy.reshape((0, 2))
    pts = [rotate(Point(x, y), angle_deg, origin=origin_xy) for x, y in xy]
    return np.array([(p.x, p.y) for p in pts], dtype=float)


def cluster_polygons(xy: np.ndarray, om_shape, eps_nm: float, min_samples: int):
    if xy.size == 0:
        return [], np.array([], dtype=int)
    model = DBSCAN(eps=eps_nm, min_samples=min_samples).fit(xy)
    labels = model.labels_
    polys = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        pts = xy[labels == lbl]
        hull = MultiPoint(pts).convex_hull
        clipped = hull.intersection(om_shape)
        if (not clipped.is_empty) and (clipped.area > 0):
            polys.append(clipped)
    return polys, labels


def write_histogram_csv(path: str, polys) -> None:
    # Minimal "histogram" output: per-cluster area list (nm^2); downstream histogramming is trivial.
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_area_nm2"])
        for p in polys:
            w.writerow([float(p.area)])


def write_summary_csv(path: str,
                      om_area: float,
                      old_union_area: float,
                      new_union_area: float,
                      overlap_area: float,
                      n_old_total: int,
                      n_new_total: int,
                      n_old_in_overlap: int,
                      n_new_in_overlap: int) -> None:
    combined_union_area = float(unary_union([
        MultiPoint([]).buffer(0)  # dummy, overwritten below if needed
    ]).area)
    # Compute combined union robustly (avoid empty list corner cases)
    combined_union_area = float(unary_union([
        unary_union([])
    ]).area)  # will be 0.0

    # Recompute properly using areas (safe without geometry objects):
    # combined_union_area = A + B - intersection
    combined_union_area = float(old_union_area + new_union_area - overlap_area)

    def _safe_div(a, b):
        return float(a / b) if b and b > 0 else float("nan")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["om_area_nm2", om_area])
        w.writerow(["old_cluster_area_nm2", old_union_area])
        w.writerow(["new_cluster_area_nm2", new_union_area])
        w.writerow(["overlap_area_nm2", overlap_area])
        w.writerow(["combined_cluster_area_nm2", combined_union_area])

        w.writerow(["old_cluster_coverage_frac", _safe_div(old_union_area, om_area)])
        w.writerow(["new_cluster_coverage_frac", _safe_div(new_union_area, om_area)])
        w.writerow(["overlap_coverage_frac", _safe_div(overlap_area, om_area)])
        w.writerow(["overlap_over_union_frac", _safe_div(overlap_area, combined_union_area)])

        w.writerow(["n_old_localisations_total", n_old_total])
        w.writerow(["n_new_localisations_total", n_new_total])
        w.writerow(["n_old_localisations_in_overlap", n_old_in_overlap])
        w.writerow(["n_new_localisations_in_overlap", n_new_in_overlap])
        w.writerow(["frac_old_in_overlap", _safe_div(n_old_in_overlap, n_old_total)])
        w.writerow(["frac_new_in_overlap", _safe_div(n_new_in_overlap, n_new_total)])


def plot_overlap_map(path: str,
                     old_xy: np.ndarray,
                     new_xy: np.ndarray,
                     old_labels: np.ndarray,
                     new_labels: np.ndarray,
                     overlap_geom,
                     title: str = "Aligned Combined Overlap Map",
                     dpi: int = 1200) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(False)

    def _scatter(xy, labels, label, alpha_noise=0.25, alpha_cluster=0.65):
        if xy.size == 0:
            return
        noise = labels == -1
        cluster = labels != -1
        # colours kept as in original script
        ax.scatter(xy[noise, 0], xy[noise, 1], s=25, alpha=alpha_noise,
                   edgecolor="black", linewidth=0.25, label=label)
        ax.scatter(xy[cluster, 0], xy[cluster, 1], s=25, alpha=alpha_cluster,
                   edgecolor="black", linewidth=0.25)

    _scatter(old_xy, old_labels, "Old LPS")
    _scatter(new_xy, new_labels, "New LPS")

    # highlight overlap points (geometric overlap; not molecular co-occupancy)
    if overlap_geom and (not overlap_geom.is_empty):
        old_in = np.array([(x, y) for x, y in old_xy if overlap_geom.contains(Point(x, y))], dtype=float)
        new_in = np.array([(x, y) for x, y in new_xy if overlap_geom.contains(Point(x, y))], dtype=float)
        if old_in.size:
            ax.scatter(old_in[:, 0], old_in[:, 1], s=25, alpha=0.9, edgecolor="black", linewidth=0.3, label="Overlap")
        if new_in.size:
            ax.scatter(new_in[:, 0], new_in[:, 1], s=25, alpha=0.9, edgecolor="black", linewidth=0.3)

    # scale bar (500 nm)
    bar_nm = 500.0
    dx = ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] != ax.get_xlim()[0] else 1.0
    bar_frac = bar_nm / dx
    start_frac = 0.05
    bar_y = 0.01
    ax.plot([start_frac, start_frac + bar_frac], [bar_y, bar_y],
            transform=ax.transAxes, color="black", lw=2, solid_capstyle="butt")
    ax.text(start_frac + bar_frac / 2, bar_y - 0.025, "500 nm",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, pad=12)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_output_paths(old_path: str, new_path: str) -> OverlapOutputs:
    base_name = os.path.splitext(os.path.basename(old_path))[0] + "__" + os.path.splitext(os.path.basename(new_path))[0]
    out_dir = os.path.join(os.path.dirname(old_path), base_name)
    os.makedirs(out_dir, exist_ok=True)
    return OverlapOutputs(
        out_dir=out_dir,
        summary_csv=os.path.join(out_dir, "LPS_output_summary.csv"),
        overlap_png=os.path.join(out_dir, "combined_overlap_map.png"),
        old_hist_csv=os.path.join(out_dir, "old_lps_cluster_histogram.csv"),
        new_hist_csv=os.path.join(out_dir, "new_lps_cluster_histogram.csv"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="CSV file for background/old LPS localisations")
    ap.add_argument("--new", required=True, help="CSV file for newly inserted/new LPS localisations")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS_NM)
    ap.add_argument("--min_samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--om_buffer_nm", type=float, default=DEFAULT_OM_BUFFER_NM)
    ap.add_argument("--om_simplify_nm", type=float, default=DEFAULT_OM_SIMPLIFY_NM)
    args = ap.parse_args()

    old_xy = load_xy(args.old)
    new_xy = load_xy(args.new)
    all_xy = np.vstack([old_xy, new_xy])

    om_shape = compute_om_shape(all_xy, buffer_nm=args.om_buffer_nm, simplify_nm=args.om_simplify_nm)
    angle_deg = pca_rotation_deg(all_xy)
    origin = tuple(om_shape.centroid.coords[0])

    # rotate everything into common frame
    om_shape_r = rotate(om_shape, angle_deg, origin=origin)
    old_xy_r = rotate_xy(old_xy, angle_deg, origin)
    new_xy_r = rotate_xy(new_xy, angle_deg, origin)

    old_polys, old_labels = cluster_polygons(old_xy_r, om_shape_r, eps_nm=args.eps, min_samples=args.min_samples)
    new_polys, new_labels = cluster_polygons(new_xy_r, om_shape_r, eps_nm=args.eps, min_samples=args.min_samples)

    old_union = unary_union(old_polys) if old_polys else None
    new_union = unary_union(new_polys) if new_polys else None
    overlap = None
    if old_union and new_union:
        overlap = old_union.intersection(new_union)

    old_union_area = float(old_union.area) if old_union else 0.0
    new_union_area = float(new_union.area) if new_union else 0.0
    overlap_area = float(overlap.area) if overlap and (not overlap.is_empty) else 0.0
    om_area = float(om_shape_r.area)

    n_old_total = int(old_xy_r.shape[0])
    n_new_total = int(new_xy_r.shape[0])
    if overlap and (not overlap.is_empty):
        n_old_in = int(sum(overlap.contains(Point(x, y)) for x, y in old_xy_r))
        n_new_in = int(sum(overlap.contains(Point(x, y)) for x, y in new_xy_r))
    else:
        n_old_in, n_new_in = 0, 0

    out = build_output_paths(args.old, args.new)
    write_summary_csv(out.summary_csv, om_area, old_union_area, new_union_area, overlap_area,
                      n_old_total, n_new_total, n_old_in, n_new_in)
    write_histogram_csv(out.old_hist_csv, old_polys)
    write_histogram_csv(out.new_hist_csv, new_polys)
    plot_overlap_map(out.overlap_png, old_xy_r, new_xy_r, old_labels, new_labels, overlap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
