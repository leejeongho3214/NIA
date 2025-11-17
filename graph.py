import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_data(filepath: Path) -> pd.DataFrame:
    """텍스트 파일에서 angle, metric, class, sample_id를 파싱한다."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            angle, metric, grade, sample_id = parts[:4]
            records.append(
                {
                    "angle": angle,
                    "metric": metric,
                    "class": int(grade),
                    "sample_id": sample_id,
                }
            )
    return pd.DataFrame(records)


def build_scatter_by_condition(pred_path: Path, gt_path: Path, output_path: Path) -> None:
    pred_df = parse_data(pred_path)
    gt_df = parse_data(gt_path)

    if pred_df.empty or gt_df.empty:
        raise ValueError("예측 또는 정답 데이터가 비어 있습니다.")

    merged = pd.merge(
        pred_df,
        gt_df,
        on=["sample_id", "angle", "metric"],
        suffixes=("_pred", "_gt"),
    )

    if merged.empty:
        raise ValueError("pred와 gt 파일에 겹치는 샘플이 없습니다.")

    grouped = merged.groupby(["angle", "metric"])
    n_groups = grouped.ngroups

    cols = min(3, n_groups)
    rows = math.ceil(n_groups / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    flat_axes = axes.flatten()

    for ax in flat_axes[n_groups:]:
        ax.set_visible(False)

    for idx, ((angle, metric), group) in enumerate(grouped):
        ax = flat_axes[idx]
        ax.scatter(
            group["class_pred"],
            group["class_gt"],
            s=60,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.5,
        )

        min_val = min(group["class_pred"].min(), group["class_gt"].min())
        max_val = max(group["class_pred"].max(), group["class_gt"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

        corr = group["class_pred"].corr(group["class_gt"])
        corr_text = f"r={corr:.2f}" if pd.notna(corr) else "r=N/A"

        ax.set_title(f"{angle} · {metric}\n{corr_text}, n={len(group)}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("GT")
        ax.set_xticks(sorted(group["class_pred"].unique()))
        ax.set_yticks(sorted(group["class_gt"].unique()))
        ax.grid(True, alpha=0.3)

    fig.suptitle("GT vs Prediction by Angle & Symptom", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"그래프가 '{output_path}' 파일로 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    base_dir = Path("checkpoint/coatnet-ori/class/1st_c4_10/prediction/1")
    pred_file = base_dir / "pred.txt"
    gt_file = base_dir / "gt.txt"
    output_file = Path("angle_metric_scatter.png")

    build_scatter_by_condition(pred_file, gt_file, output_file)
