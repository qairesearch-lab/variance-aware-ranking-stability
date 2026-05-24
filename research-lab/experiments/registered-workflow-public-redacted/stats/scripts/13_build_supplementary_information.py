#!/usr/bin/env python3
"""Build a numbered Supplementary Information draft from table and figure inventories."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rel_link(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def table_preview(path: Path, max_rows: int = 8) -> tuple[list[str], list[list[str]], int]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows: list[list[str]] = []
        total_rows = 0
        for row in reader:
            total_rows += 1
            if len(rows) < max_rows:
                rows.append(row)
    return header, rows, total_rows


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    out = [
        "| " + " | ".join(md_escape(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        out.append("| " + " | ".join(md_escape(v) for v in padded[: len(header)]) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-inventory",
        default="research-lab/experiments/registered-workflow/stats/tables/manuscript_table_inventory.csv",
    )
    parser.add_argument(
        "--figure-inventory",
        default="research-lab/experiments/registered-workflow/stats/figures/supplementary_figure_inventory.csv",
    )
    parser.add_argument(
        "--tables-dir",
        default="research-lab/experiments/registered-workflow/stats/tables",
    )
    parser.add_argument(
        "--figures-dir",
        default="research-lab/experiments/registered-workflow/stats/figures",
    )
    parser.add_argument(
        "--output",
        default="research-lab/reports/CMPB_Workflow_Ranking_Stability_Supplementary_Information_Draft.md",
    )
    args = parser.parse_args()

    table_inventory = read_csv(Path(args.table_inventory))
    figure_inventory = read_csv(Path(args.figure_inventory))
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    output_path = Path(args.output)

    supplementary_tables = [
        r for r in table_inventory if r["manuscript_table"].startswith("Supplementary Table")
    ]

    lines: list[str] = []
    lines.extend(
        [
            "# Supplementary Information",
            "",
            "Manuscript: Evaluation design and model-selection stability in medical image classification benchmarks",
            "",
            "Supplementary draft",
            "",
            "This file consolidates the currently numbered supplementary tables and figures for the CMPB manuscript manuscript draft. Large source tables are kept as CSV files to preserve complete machine-readable records; the Markdown entries below provide captions, source files, manuscript locations, and compact previews where useful.",
            "",
            "## Supplementary Table Inventory",
            "",
            "| Supplementary table | Title | Source file | Manuscript location |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in supplementary_tables:
        source = item["source_file"]
        source_path = tables_dir / source
        source_link = rel_link(source_path, ROOT) if source_path.exists() else source
        lines.append(
            f"| {item['manuscript_table']} | {md_escape(item['title'])} | `{source_link}` | {md_escape(item['manuscript_location'])} |"
        )

    lines.extend(
        [
            "",
            "## Supplementary Figure Inventory",
            "",
            "| Supplementary figure | Title | Source file | Manuscript location |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in figure_inventory:
        source_path = figures_dir / item["source_file"]
        source_link = rel_link(source_path, ROOT) if source_path.exists() else item["source_file"]
        lines.append(
            f"| {item['supplementary_figure']} | {md_escape(item['title'])} | `{source_link}` | {md_escape(item['manuscript_location'])} |"
        )

    lines.extend(["", "## Supplementary Tables", ""])
    for item in supplementary_tables:
        source_path = tables_dir / item["source_file"]
        lines.extend(
            [
                f"### {item['manuscript_table']}. {item['title']}",
                "",
                f"Manuscript location: {item['manuscript_location']}.",
                "",
            ]
        )
        if source_path.exists():
            source_link = rel_link(source_path, output_path.parent)
            header, preview_rows, total_rows = table_preview(source_path)
            lines.extend(
                [
                    f"Complete source table: [{item['source_file']}]({source_link}).",
                    "",
                    f"Rows in complete CSV: {total_rows}.",
                    "",
                ]
            )
            if total_rows <= 30:
                lines.extend([md_table(header, preview_rows), ""])
            else:
                lines.extend(
                    [
                        "Preview of the first 8 rows:",
                        "",
                        md_table(header, preview_rows),
                        "",
                        "The full table is retained in the linked CSV because it is too large for readable inline rendering.",
                        "",
                    ]
                )
        else:
            lines.extend([f"Complete source table: `{item['source_file']}` (not found).", ""])

    lines.extend(["## Supplementary Figures", ""])
    for item in figure_inventory:
        source_path = figures_dir / item["source_file"]
        lines.extend(
            [
                f"### {item['supplementary_figure']}. {item['title']}",
                "",
                f"Manuscript location: {item['manuscript_location']}.",
                "",
                f"Caption note: {item.get('notes', '')}",
                "",
            ]
        )
        if source_path.exists():
            image_link = rel_link(source_path, output_path.parent)
            lines.extend([f"![{item['supplementary_figure']}. {item['title']}]({image_link})", ""])
        else:
            lines.extend([f"Figure source: `{item['source_file']}` (not found).", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")
    print(f"Supplementary tables: {len(supplementary_tables)}")
    print(f"Supplementary figures: {len(figure_inventory)}")


if __name__ == "__main__":
    main()
