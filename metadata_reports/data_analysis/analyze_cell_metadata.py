#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze per-slide metadata JSONs produced by the CellViT mask pipeline and
generate summary tables + figures (pie charts, bars, histograms) + a Markdown report.

Edit the two variables below to match your system:
    META_DIR = "path/to/metadata"
    OUT_DIR  = "path/to/output/reports"

Then just run:
    python analyze_cell_metadata.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


META_DIR = r"/EC500 AI guided whole slide imaging analysis/Output_Folder/metadata"  # Folder containing the *.json metadata files
OUT_DIR  = r"/EC500 AI guided whole slide imaging analysis/Output_Folder/metadata_reports"  # Output folder for charts + CSVs + report




# utility functions
def load_metadata(meta_dir: Path):
    files = sorted(meta_dir.glob("*.json"))
    slides = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
                slides.append(data)
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
    if not slides:
        print("[WARN] No JSON metadata files found.")
    return slides


def aggregate_by_label(slides):
    agg = {}
    label_slide_presence = defaultdict(int)
    for slide in slides:
        labels = slide.get("labels", {})
        for lbl in labels:
            label_slide_presence[lbl] += 1
            rc = int(labels[lbl].get("regions_count", 0))
            areas = [float(a) for a in labels[lbl].get("areas", [])]
            if lbl not in agg:
                agg[lbl] = {"total_regions": 0, "total_area": 0.0, "all_areas": []}
            agg[lbl]["total_regions"] += rc
            agg[lbl]["total_area"] += sum(areas)
            agg[lbl]["all_areas"].extend(areas)
    for lbl in agg:
        agg[lbl]["slides_count_with_label"] = label_slide_presence[lbl]
    return agg


def make_output_dirs(out_dir: Path):
    figs = out_dir / "figs"
    tables = out_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figs, tables


# -----------------------------------------------------------------------------#
# Plotting helpers
# -----------------------------------------------------------------------------#
def pie_chart(values, labels, title, out_path):
    fig = plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def bar_chart(x, y, title, ylabel, out_path, yerr=None):
    fig = plt.figure()
    plt.bar(x, y, yerr=yerr)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def histogram(data, title, xlabel, out_path, bins=30):
    fig = plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------#
# Main analysis
# -----------------------------------------------------------------------------#
def main():
    meta_dir = Path(META_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir, tables_dir = make_output_dirs(out_dir)

    slides = load_metadata(meta_dir)
    if not slides:
        return

    agg = aggregate_by_label(slides)

    # ---------- Save per-label summary ----------
    rows = []
    for lbl, d in agg.items():
        arr = np.array(d["all_areas"]) if d["all_areas"] else np.array([0])
        stats = {
            "label": lbl,
            "total_regions": d["total_regions"],
            "slides_count_with_label": d["slides_count_with_label"],
            "total_area_px2": d["total_area"],
            "mean_area_px2": float(np.mean(arr)),
            "median_area_px2": float(np.median(arr)),
            "min_area_px2": float(np.min(arr)),
            "max_area_px2": float(np.max(arr)),
            "std_area_px2": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }
        rows.append(stats)

    label_df = pd.DataFrame(rows).sort_values("total_regions", ascending=False)
    label_csv = tables_dir / "label_summary.csv"
    label_df.to_csv(label_csv, index=False)

    # ---------- Pie charts ----------
    if not label_df.empty:
        pie_chart(
            label_df["total_regions"],
            label_df["label"],
            "Proportion of Regions by Label",
            figs_dir / "pie_regions_by_label.png",
        )
        pie_chart(
            label_df["total_area_px2"],
            label_df["label"],
            "Proportion of Total Area by Label (px²)",
            figs_dir / "pie_area_by_label.png",
        )

        bar_chart(
            label_df["label"],
            label_df["mean_area_px2"],
            "Mean Area per Label (px²)",
            "Mean area (px²)",
            figs_dir / "bar_mean_area_per_label.png",
            yerr=label_df["std_area_px2"],
        )

        # Histograms
        # ---------- Combined Histograms ----------
        # ---------- Combined Log-Scaled Histogram Across Labels ----------
        all_labels = []
        all_data = []

        for lbl, d in agg.items():
            arr = np.array(d["all_areas"])
            if arr.size > 0:
                all_labels.append(lbl)
                all_data.append(arr)

        if all_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot all histograms on the same axes
            for lbl, arr in zip(all_labels, all_data):
                # Filter out nonpositive values (can't take log10 of zero)
                arr = arr[arr > 0]
                if len(arr) == 0:
                    continue
                ax.hist(np.log10(arr), bins=30, alpha=0.5, label=lbl)

            ax.set_xlabel("log₁₀(Area [px²])")
            ax.set_ylabel("Count")
            ax.set_title("Log-Scaled Area Distributions Across Labels")
            ax.legend(loc="upper right", ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(figs_dir / "hist_all_labels_log_scaled.png", dpi=200)
            plt.close(fig)

    # ---------- Markdown report ----------
    report_path = out_dir / "report.md"
    total_slides = len(slides)
    total_regions = int(label_df["total_regions"].sum())
    with open(report_path, "w") as f:
        f.write(f"# Cell Segmentation Metadata Report\n\n")
        f.write(f"- **Metadata directory:** `{meta_dir}`\n")
        f.write(f"- **Slides analyzed:** {total_slides}\n")
        f.write(f"- **Total regions:** {total_regions}\n\n")

        f.write(f"## Summary Table\nSaved at: `{label_csv}`\n\n")
        f.write("## Figures\n")
        f.write(f"- Pie (regions): `figs/pie_regions_by_label.png`\n")
        f.write(f"- Pie (area): `figs/pie_area_by_label.png`\n")
        f.write(f"- Bar (mean area): `figs/bar_mean_area_per_label.png`\n\n")

        f.write("## Top Labels by Region Count\n\n")
        top = label_df.head(10)
        if not top.empty:
            f.write("| Label | Total Regions | Mean Area (px²) |\n|---|---:|---:|\n")
            for _, r in top.iterrows():
                f.write(f"| {r['label']} | {r['total_regions']} | {r['mean_area_px2']:.1f} |\n")

        f.write("\n## Notes\n")
        f.write("- Areas are in pixel² (px²).\n")
        f.write("- Pie charts represent proportions across all slides.\n")
        f.write("- Histograms show per-label area distributions.\n")

    print(f"Analysis complete.\nResults saved in: {out_dir}")


if __name__ == "__main__":
    main()
