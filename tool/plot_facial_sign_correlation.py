"""Plot ground-truth vs prediction correlations per facial sign.

This script expects both GT and prediction text files where every line is
formatted as "angle, facial_sign, grade, name". The script merges the two
files on (angle, facial_sign, name), computes Pearson correlations per facial
sign, and saves a lightweight SVG scatter plot for each sign that makes it
easy to inspect how predictions align with the GT grades.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import random


Record = Tuple[str, str, float, str]
Key = Tuple[str, str, str]


def parse_file(path: Path) -> Dict[Key, float]:
    """Parse a txt file and return a mapping from (angle, sign, name) to grade."""

    data: Dict[Key, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(
                    f"{path} line {idx}: expected 4 comma-separated fields, got {len(parts)}"
                )
            angle, facial_sign, grade_str, name = parts
            key: Key = (angle, facial_sign, name)
            if key in data:
                raise ValueError(
                    f"{path} line {idx}: duplicate entry for angle={angle}, sign={facial_sign}, name={name}"
                )
            try:
                grade = float(grade_str)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"{path} line {idx}: grade '{grade_str}' is not numeric"
                ) from exc
            data[key] = grade
    return data


def merge_samples(gt: Dict[Key, float], pred: Dict[Key, float]) -> List[Tuple[str, float, float]]:
    """Return (facial_sign, gt_grade, pred_grade) for matching samples."""

    merged: List[Tuple[str, float, float]] = []
    missing_pred = 0
    for key, gt_grade in gt.items():
        if key not in pred:
            missing_pred += 1
            continue
        merged.append((key[1], gt_grade, pred[key]))

    if missing_pred:
        print(f"[warning] Skipped {missing_pred} GT entries with no matching prediction.")
    missing_gt = len(pred) - len(merged)
    if missing_gt:
        print(f"[warning] {missing_gt} predictions had no GT counterpart.")

    if not merged:
        raise ValueError("No overlapping samples between GT and predictions.")
    return merged


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    xs_list = list(xs)
    ys_list = list(ys)
    n = len(xs_list)
    if n == 0 or n != len(ys_list):
        return None

    mean_x = sum(xs_list) / n
    mean_y = sum(ys_list) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_list, ys_list))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs_list))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys_list))
    denom = den_x * den_y
    if math.isclose(denom, 0.0):
        return None
    return num / denom


def build_plot(
    by_sign: Dict[str, Dict[str, List[float]]],
    correlations: Dict[str, float | None],
    output_path: Path,
) -> None:
    signs = sorted(by_sign)
    cols = min(3, max(1, int(math.ceil(math.sqrt(len(signs))))))
    rows = math.ceil(len(signs) / cols)
    cell_w, cell_h = 320, 320
    padding = 50
    svg_width = cols * cell_w
    svg_height = rows * cell_h

    def scale(value: float, min_val: float, max_val: float, length: float) -> float:
        if math.isclose(max_val, min_val):
            return length / 2
        return (value - min_val) / (max_val - min_val) * length

    def expand_range(min_val: float, max_val: float) -> Tuple[float, float]:
        if math.isclose(min_val, max_val):
            delta = max(1.0, abs(min_val) * 0.1)
            return min_val - delta, max_val + delta
        return min_val, max_val

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<style>text{font-family:"Arial",sans-serif;} .label{font-size:11px;} .title{font-size:14px;font-weight:bold;}</style>',
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    for idx, sign in enumerate(signs):
        gt_vals = by_sign[sign]["gt"]
        pred_vals = by_sign[sign]["pred"]
        col = idx % cols
        row = idx // cols
        origin_x = col * cell_w
        origin_y = row * cell_h
        plot_left = origin_x + padding
        plot_top = origin_y + padding
        plot_width = cell_w - 2 * padding
        plot_height = cell_h - 2 * padding

        min_x = min(gt_vals)
        max_x = max(gt_vals)
        min_y = min(pred_vals)
        max_y = max(pred_vals)
        min_x, max_x = expand_range(min_x, max_x)
        min_y, max_y = expand_range(min_y, max_y)

        svg_parts.append(
            f'<rect x="{origin_x}" y="{origin_y}" width="{cell_w}" height="{cell_h}" fill="#fafafa" stroke="#dddddd"/>'
        )
        corr = correlations.get(sign)
        corr_txt = f"r={corr:.2f}" if corr is not None else "r=NA"
        svg_parts.append(
            f'<text class="title" x="{origin_x + cell_w / 2}" y="{origin_y + 24}" text-anchor="middle">{sign} ({corr_txt})</text>'
        )

        # Axes
        x_axis_y = plot_top + plot_height
        svg_parts.append(
            f'<line x1="{plot_left}" y1="{x_axis_y}" x2="{plot_left + plot_width}" y2="{x_axis_y}" stroke="#333" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333" stroke-width="1"/>'
        )

        # Ticks for x-axis
        tick_count = 4
        for i in range(tick_count + 1):
            value = min_x + (max_x - min_x) * i / tick_count
            x = plot_left + scale(value, min_x, max_x, plot_width)
            svg_parts.append(
                f'<line x1="{x}" y1="{x_axis_y}" x2="{x}" y2="{x_axis_y + 5}" stroke="#333" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text class="label" x="{x}" y="{x_axis_y + 18}" text-anchor="middle">{value:.1f}</text>'
            )

        # Ticks for y-axis
        for i in range(tick_count + 1):
            value = min_y + (max_y - min_y) * i / tick_count
            y = plot_top + plot_height - scale(value, min_y, max_y, plot_height)
            svg_parts.append(
                f'<line x1="{plot_left}" y1="{y}" x2="{plot_left - 5}" y2="{y}" stroke="#333" stroke-width="1"/>'
            )
            svg_parts.append(
                f'<text class="label" x="{plot_left - 8}" y="{y + 4}" text-anchor="end">{value:.1f}</text>'
            )

        # Diagonal (y=x)
        diag_start = max(min_x, min_y)
        diag_end = min(max_x, max_y)
        if diag_start < diag_end:
            x1 = plot_left + scale(diag_start, min_x, max_x, plot_width)
            y1 = plot_top + plot_height - scale(diag_start, min_y, max_y, plot_height)
            x2 = plot_left + scale(diag_end, min_x, max_x, plot_width)
            y2 = plot_top + plot_height - scale(diag_end, min_y, max_y, plot_height)
            svg_parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#999" stroke-width="1" stroke-dasharray="4 2"/>'
            )

        # Scatter points
        for gt_val, pred_val in zip(gt_vals, pred_vals):
            cx = plot_left + scale(gt_val, min_x, max_x, plot_width)
            cy = plot_top + plot_height - scale(pred_val, min_y, max_y, plot_height)
            svg_parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="4" fill="#1f77b4" fill-opacity="0.7"/>'
            )

        # Axis labels
        svg_parts.append(
            f'<text class="label" x="{origin_x + cell_w / 2}" y="{origin_y + cell_h - 10}" text-anchor="middle">GT grade</text>'
        )
        svg_parts.append(
            f'<text class="label" transform="rotate(-90 {origin_x + 14} {origin_y + cell_h / 2})" x="{origin_x + 14}" y="{origin_y + cell_h / 2}" text-anchor="middle">Pred grade</text>'
        )

    svg_parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Saved plot to {output_path}")


def build_overlay_plot(
    by_sign: Dict[str, Dict[str, List[float]]],
    correlations: Dict[str, float | None],
    output_path: Path,
) -> None:
    signs = sorted(by_sign)
    all_gt = [g for sign in signs for g in by_sign[sign]["gt"]]
    all_pred = [p for sign in signs for p in by_sign[sign]["pred"]]
    min_x, max_x = min(all_gt), max(all_gt)
    min_y, max_y = min(all_pred), max(all_pred)

    def expand_range(min_val: float, max_val: float) -> Tuple[float, float]:
        if math.isclose(min_val, max_val):
            delta = max(1.0, abs(min_val) * 0.1)
            return min_val - delta, max_val + delta
        return min_val, max_val

    min_x, max_x = expand_range(min_x, max_x)
    min_y, max_y = expand_range(min_y, max_y)

    width, height = 900, 600
    left_margin, right_margin = 80, 260
    top_margin, bottom_margin = 60, 80
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    def scale_x(value: float) -> float:
        if math.isclose(max_x, min_x):
            return left_margin + plot_width / 2
        return left_margin + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        if math.isclose(max_y, min_y):
            return top_margin + plot_height / 2
        return top_margin + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:"Arial",sans-serif;} .label{font-size:12px;} .title{font-size:16px;font-weight:bold;} .legend{font-size:12px;}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    svg_parts.append(
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">GT vs Prediction (All Facial Signs)</text>'
    )

    # Axes
    x_axis_y = top_margin + plot_height
    svg_parts.append(
        f'<line x1="{left_margin}" y1="{x_axis_y}" x2="{left_margin + plot_width}" y2="{x_axis_y}" stroke="#333" stroke-width="1.2"/>'
    )
    svg_parts.append(
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#333" stroke-width="1.2"/>'
    )

    tick_count = 5
    for i in range(tick_count + 1):
        val = min_x + (max_x - min_x) * i / tick_count
        x = scale_x(val)
        svg_parts.append(
            f'<line x1="{x}" y1="{x_axis_y}" x2="{x}" y2="{x_axis_y + 6}" stroke="#333" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text class="label" x="{x}" y="{x_axis_y + 22}" text-anchor="middle">{val:.1f}</text>'
        )

    for i in range(tick_count + 1):
        val = min_y + (max_y - min_y) * i / tick_count
        y = scale_y(val)
        svg_parts.append(
            f'<line x1="{left_margin}" y1="{y}" x2="{left_margin - 6}" y2="{y}" stroke="#333" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text class="label" x="{left_margin - 10}" y="{y + 4}" text-anchor="end">{val:.1f}</text>'
        )

    svg_parts.append(
        f'<text class="label" x="{left_margin + plot_width / 2}" y="{height - 30}" text-anchor="middle">GT grade</text>'
    )
    svg_parts.append(
        f'<text class="label" transform="rotate(-90 {left_margin - 45} {top_margin + plot_height / 2})" x="{left_margin - 45}" y="{top_margin + plot_height / 2}" text-anchor="middle">Pred grade</text>'
    )

    diag_min = max(min_x, min_y)
    diag_max = min(max_x, max_y)
    if diag_min < diag_max:
        svg_parts.append(
            f'<line x1="{scale_x(diag_min)}" y1="{scale_y(diag_min)}" x2="{scale_x(diag_max)}" y2="{scale_y(diag_max)}" stroke="#999" stroke-width="1" stroke-dasharray="4 2"/>'
        )

    # Points by sign
    legend_x = width - right_margin + 20
    legend_y_start = top_margin + 10
    legend_line_height = 20
    for idx, sign in enumerate(signs):
        color = palette[idx % len(palette)]
        for gt_val, pred_val in zip(by_sign[sign]["gt"], by_sign[sign]["pred"]):
            svg_parts.append(
                f'<circle cx="{scale_x(gt_val)}" cy="{scale_y(pred_val)}" r="4" fill="{color}" fill-opacity="0.75" stroke="black" stroke-width="0.4"/>'
            )

        legend_y = legend_y_start + idx * legend_line_height
        corr = correlations.get(sign)
        corr_txt = f"r={corr:.2f}" if corr is not None else "r=NA"
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}" stroke="#333" stroke-width="0.5"/>'
        )
        svg_parts.append(
            f'<text class="legend" x="{legend_x + 20}" y="{legend_y + 1}" alignment-baseline="middle">{sign} ({corr_txt})</text>'
        )

    svg_parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Saved overlay plot to {output_path}")


def build_category_scatter(
    by_sign: Dict[str, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    signs = sorted(by_sign)
    all_values = [val for sign in signs for val in (by_sign[sign]["gt"] + by_sign[sign]["pred"])]
    min_y, max_y = min(all_values), max(all_values)

    def expand_range(min_val: float, max_val: float) -> Tuple[float, float]:
        if math.isclose(min_val, max_val):
            delta = max(1.0, abs(min_val) * 0.1)
            return min_val - delta, max_val + delta
        return min_val, max_val

    min_y, max_y = expand_range(min_y, max_y)

    left_margin, right_margin = 90, 40
    top_margin, bottom_margin = 60, 90
    spacing = 140
    width = left_margin + right_margin + spacing * max(0, len(signs) - 1)
    height = top_margin + bottom_margin + 460
    plot_height = height - top_margin - bottom_margin

    def scale_y(value: float) -> float:
        if math.isclose(max_y, min_y):
            return top_margin + plot_height / 2
        return top_margin + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    rng = random.Random(42)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:"Arial",sans-serif;} .label{font-size:12px;} .title{font-size:16px;font-weight:bold;} .legend{font-size:13px;}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]

    svg_parts.append(
        f'<text class="title" x="{width / 2}" y="30" text-anchor="middle">All Values per Facial Sign</text>'
    )

    # Horizontal grid lines
    tick_count = 5
    for i in range(tick_count + 1):
        val = min_y + (max_y - min_y) * i / tick_count
        y = scale_y(val)
        svg_parts.append(
            f'<line x1="{left_margin - 10}" y1="{y}" x2="{width - right_margin + 10}" y2="{y}" stroke="#efefef" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text class="label" x="{left_margin - 20}" y="{y + 4}" text-anchor="end">{val:.1f}</text>'
        )

    svg_parts.append(
        f'<text class="label" transform="rotate(-90 {left_margin - 50} {top_margin + plot_height / 2})" x="{left_margin - 50}" y="{top_margin + plot_height / 2}" text-anchor="middle">Grade</text>'
    )

    legend_x = width - right_margin - 120
    legend_y = top_margin - 20
    svg_parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="#1f77b4" stroke="#333" stroke-width="0.6"/>'
    )
    svg_parts.append(
        f'<text class="legend" x="{legend_x + 20}" y="{legend_y + 11}">GT</text>'
    )
    svg_parts.append(
        f'<rect x="{legend_x + 70}" y="{legend_y}" width="14" height="14" fill="#ff7f0e" stroke="#333" stroke-width="0.6"/>'
    )
    svg_parts.append(
        f'<text class="legend" x="{legend_x + 90}" y="{legend_y + 11}">Pred</text>'
    )

    for idx, sign in enumerate(signs):
        base_x = left_margin + idx * spacing
        svg_parts.append(
            f'<line x1="{base_x}" y1="{top_margin}" x2="{base_x}" y2="{top_margin + plot_height}" stroke="#d0d0d0" stroke-width="1" stroke-dasharray="3 3"/>'
        )
        svg_parts.append(
            f'<text class="label" x="{base_x}" y="{height - bottom_margin + 30}" text-anchor="middle">{sign}</text>'
        )

        gt_vals = by_sign[sign]["gt"]
        pred_vals = by_sign[sign]["pred"]
        jitter = spacing * 0.2
        for i, value in enumerate(gt_vals):
            offset = (i % 5 - 2) / 4 * jitter
            x = base_x - jitter / 2 + offset
            svg_parts.append(
                f'<circle cx="{x}" cy="{scale_y(value)}" r="4" fill="#1f77b4" fill-opacity="0.75" stroke="#103c6d" stroke-width="0.4"/>'
            )

        for i, value in enumerate(pred_vals):
            offset = (i % 5 - 2) / 4 * jitter
            x = base_x + jitter / 2 + offset
            svg_parts.append(
                f'<circle cx="{x}" cy="{scale_y(value)}" r="4" fill="#ff7f0e" fill-opacity="0.75" stroke="#7a3606" stroke-width="0.4"/>'
            )

    svg_parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Saved category scatter plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path, help="Path to gt.txt")
    parser.add_argument("--pred", required=True, type=Path, help="Path to pred.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("facial_sign_correlation.svg"),
        help="Path to save the per-sign SVG plot (used in per-sign or both modes).",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=None,
        help="Optional path for the all-in-one overlay plot.",
    )
    parser.add_argument(
        "--category-output",
        type=Path,
        default=None,
        help="Optional path for the per-sign category scatter plot.",
    )
    parser.add_argument(
        "--mode",
        choices=["per-sign", "overlay", "both", "category", "all"],
        default="per-sign",
        help="Which visualization(s) to generate (all == per-sign + overlay + category).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print Pearson correlation values per facial sign.",
    )
    args = parser.parse_args()

    gt_data = parse_file(args.gt)
    pred_data = parse_file(args.pred)
    merged = merge_samples(gt_data, pred_data)

    by_sign: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"gt": [], "pred": []})
    for sign, gt_grade, pred_grade in merged:
        by_sign[sign]["gt"].append(gt_grade)
        by_sign[sign]["pred"].append(pred_grade)

    correlations = {
        sign: pearson(values["gt"], values["pred"])
        for sign, values in by_sign.items()
    }

    if args.print_summary:
        print("Facial sign correlation (Pearson r):")
        for sign in sorted(correlations):
            corr = correlations[sign]
            print(f"  - {sign}: {corr:.3f}" if corr is not None else f"  - {sign}: N/A")

    if args.mode in {"per-sign", "both", "all"}:
        build_plot(by_sign, correlations, args.output)

    if args.mode in {"overlay", "both", "all"}:
        overlay_path = (
            args.overlay_output
            if args.overlay_output is not None
            else args.output.with_name(f"{args.output.stem}_overlay.svg")
        )
        build_overlay_plot(by_sign, correlations, overlay_path)

    if args.mode in {"category", "all"}:
        category_path = (
            args.category_output
            if args.category_output is not None
            else args.output.with_name(f"{args.output.stem}_category.svg")
        )
        build_category_scatter(by_sign, category_path)


if __name__ == "__main__":
    main()
