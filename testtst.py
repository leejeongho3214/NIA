"""
Compare Sub IDs between two GT files grouped by angle and facial sign.
"""

from pathlib import Path
import re
from collections import defaultdict

# Paths to the two GT files
GT1 = Path("checkpoint/coatnet-ori/class/1st_c4_10/prediction/1/gt.txt")
GT2 = Path("checkpoint/coatnet-ori/class/1st_c4_10/prediction/2/gt.txt")

SUB_PATTERN = re.compile(r"(Sub_[0-9]+)")


def _normalize_angle(angle: str) -> str:
    """Normalize angle tokens (e.g., L30/R30 -> L/R)."""
    if angle.startswith("L"):
        return "L"
    if angle.startswith("R"):
        return "R"
    return angle


def parse_gt(path: Path):
    """Return mapping {(angle, sign): set(Sub_XXXX)} from a GT file."""
    mapping = defaultdict(set)
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        angle, sign = _normalize_angle(parts[0]), parts[1]
        match = SUB_PATTERN.search(line)
        if match:
            mapping[(angle, sign)].add(match.group(1))
    return mapping


def compare():
    m1, m2 = parse_gt(GT1), parse_gt(GT2)
    keys = sorted(set(m1.keys()) | set(m2.keys()))

    for key in keys:
        s1, s2 = m1.get(key, set()), m2.get(key, set())
        only_1 = s1 - s2
        only_2 = s2 - s1

        angle, sign = key
        print(f"[{angle}][{sign}]  file1={len(s1)}  file2={len(s2)}")
        if only_1:
            print("  - only in file1:", ", ".join(sorted(only_1)))
        if only_2:
            print("  - only in file2:", ", ".join(sorted(only_2)))
        if not only_1 and not only_2:
            print("  (identical)")


if __name__ == "__main__":
    compare()
