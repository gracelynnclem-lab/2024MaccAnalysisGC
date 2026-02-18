#!/usr/bin/env python3
"""Deterministic rank ordering for MAcc exit survey course/program ratings."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEYWORDS = [
    "course",
    "program",
    "acct",
    "accounting",
    "macc",
    "tax",
    "audit",
    "analytics",
    "core",
    "elective",
]

EXCLUDE_PATTERNS = [
    "startdate",
    "enddate",
    "status",
    "ipaddress",
    "progress",
    "duration",
    "finished",
    "recordeddate",
    "responseid",
    "recipient",
    "lastname",
    "firstname",
    "email",
    "externalreference",
    "locationlatitude",
    "locationlongitude",
    "distributionchannel",
    "userlanguage",
    "name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank order course/program survey items")
    parser.add_argument("--input", type=str, default=None, help="Path to input survey file")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--topn", type=int, default=10, help="Number of top items to plot")
    return parser.parse_args()


def detect_input_file(explicit_input: str | None) -> Path:
    if explicit_input:
        path = Path(explicit_input)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("data/ directory does not exist")

    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    if xlsx_files:
        print(
            "[INFO] No CSV found in data/. Falling back to first XLSX file for local execution.",
            file=sys.stderr,
        )
        return xlsx_files[0]

    raise FileNotFoundError("No survey file found in data/ (expected .csv)")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def id_score(values: list[str]) -> float:
    if not values:
        return 0.0
    pattern = re.compile(r"^[A-Za-z]{0,5}\d+[A-Za-z0-9_]*$|^[A-Za-z0-9_]+$")
    hits = 0
    for value in values:
        if value and " " not in value and pattern.match(value):
            hits += 1
    return hits / len(values)


def text_score(values: list[str]) -> float:
    if not values:
        return 0.0
    hits = 0
    for value in values:
        if value and (" " in value or len(value) > 20):
            hits += 1
    return hits / len(values)


def clean_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(r"^[\W_]+|[\W_]+$", "", label)
    return label or "Unnamed Item"


def load_with_header_detection(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None, dtype=str)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if raw.empty:
        raise ValueError("Input dataset is empty")

    row0 = [normalize_text(v) for v in raw.iloc[0].tolist()]
    row1 = [normalize_text(v) for v in raw.iloc[1].tolist()] if len(raw) > 1 else []

    row0_id, row1_id = id_score(row0), id_score(row1)
    row0_text, row1_text = text_score(row0), text_score(row1)

    header_mode = "single"
    if len(raw) > 1 and ((row0_id > 0.65 and row1_text > 0.35) or (row1_id > 0.65 and row0_text > 0.35)):
        header_mode = "dual"

    if header_mode == "dual":
        if row0_id >= row1_id:
            column_names = [normalize_text(v) or f"column_{i}" for i, v in enumerate(raw.iloc[0].tolist())]
            labels = [normalize_text(v) or column_names[i] for i, v in enumerate(raw.iloc[1].tolist())]
        else:
            column_names = [normalize_text(v) or f"column_{i}" for i, v in enumerate(raw.iloc[1].tolist())]
            labels = [normalize_text(v) or column_names[i] for i, v in enumerate(raw.iloc[0].tolist())]
        data = raw.iloc[2:].copy()
    else:
        column_names = [normalize_text(v) or f"column_{i}" for i, v in enumerate(raw.iloc[0].tolist())]
        labels = column_names.copy()
        data = raw.iloc[1:].copy()

    # De-duplicate column names deterministically.
    seen: dict[str, int] = {}
    deduped_names = []
    for col in column_names:
        count = seen.get(col, 0)
        if count == 0:
            deduped_names.append(col)
        else:
            deduped_names.append(f"{col}__{count}")
        seen[col] = count + 1

    data.columns = deduped_names
    label_map = {deduped_names[i]: clean_label(labels[i]) for i in range(len(deduped_names))}
    return data.reset_index(drop=True), label_map


def is_excluded(column_name: str, label: str) -> bool:
    text = f"{column_name} {label}".lower().replace(" ", "")
    return any(pattern in text for pattern in EXCLUDE_PATTERNS)


def keyword_hit(column_name: str, label: str) -> bool:
    text = f"{column_name} {label}".lower()
    return any(keyword in text for keyword in KEYWORDS)


def likert_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    in_range = numeric[(numeric >= 1) & (numeric <= 7)]
    if len(in_range) / len(numeric) < 0.8:
        return False
    integers = np.isclose(in_range, np.round(in_range))
    if integers.mean() < 0.9:
        return False
    unique_values = sorted(pd.unique(np.round(in_range).astype(int)))
    return set(unique_values).issubset(set(range(1, 8)))


def compute_rankings(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column in df.columns:
        label = label_map.get(column, column)
        if is_excluded(column, label):
            continue

        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        n_valid = int(numeric.notna().sum())
        if n_valid == 0:
            continue

        if not (keyword_hit(column, label) or likert_like(series)):
            continue

        mean_rating = float(numeric.mean())
        rows.append(
            {
                "item_label": clean_label(label),
                "column_name": column,
                "mean_rating": mean_rating,
                "n_valid": n_valid,
            }
        )

    if not rows:
        raise ValueError("No candidate rating/preference columns were detected.")

    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        by=["mean_rating", "n_valid", "item_label"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def infer_year(path: Path) -> str | None:
    match = re.search(r"(20\d{2})", path.name)
    return match.group(1) if match else None


def save_plot(ranked: pd.DataFrame, out_png: Path, out_svg: Path, topn: int, year: str | None) -> None:
    top = ranked.head(topn).copy()
    top = top.sort_values(by=["mean_rating", "n_valid", "item_label"], ascending=[True, True, False])

    fig_height = max(4, 0.45 * len(top) + 1.5)
    plt.figure(figsize=(12, fig_height))
    plt.barh(top["item_label"], top["mean_rating"], color="#2B6CB0")
    plt.xlabel("Mean Rating")
    title = f"Exit Survey Rank Ordering ({year})" if year else "Exit Survey Rank Ordering"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_svg, format="svg")
    plt.close()


def ensure_reflection_stub(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "## What changed from Project 1 to this workflow\n\n"
        "## Where is the control now\n\n"
        "## What would I do next if I had one more week\n\n"
        "## One accounting application of this workflow (specific)\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = detect_input_file(args.input)
    print(f"[INFO] Using input file: {input_path}")

    df, label_map = load_with_header_detection(input_path)
    print(f"[INFO] Loaded data shape (rows, cols): {df.shape}")

    ranked = compute_rankings(df, label_map)
    out_csv = outdir / "rank_order.csv"
    out_png = outdir / "rank_order.png"
    out_svg = outdir / "rank_order.svg"
    out_reflection = outdir / "reflection.md"

    ranked.to_csv(out_csv, index=False, float_format="%.6f")
    save_plot(ranked, out_png, out_svg, args.topn, infer_year(input_path))
    ensure_reflection_stub(out_reflection)

    print(f"[INFO] Wrote ranking CSV: {out_csv}")
    print(f"[INFO] Wrote ranking figure: {out_png}")
    print(f"[INFO] Wrote ranking vector figure: {out_svg}")
    print(f"[INFO] Reflection stub present: {out_reflection}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
