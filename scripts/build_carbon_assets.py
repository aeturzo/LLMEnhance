#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


ROOT = Path(__file__).resolve().parents[1]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


RAW_MATERIAL_FACTOR_ROWS = [
    {
        "factor_key": "plastics_generic",
        "material_label": "Generic plastics",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Map canonical plastics bucket to a curated EF flow before runtime use",
    },
    {
        "factor_key": "steel_generic",
        "material_label": "Generic steel",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Pick a production route and geography before filling this factor",
    },
    {
        "factor_key": "electronics_generic",
        "material_label": "Generic electronics assembly",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "This likely requires a curated BOM composite instead of a single raw workbook row",
    },
    {
        "factor_key": "elastomers_generic",
        "material_label": "Generic elastomers",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Map to an approved elastomer flow before runtime use",
    },
    {
        "factor_key": "packaging_cardboard_generic",
        "material_label": "Generic corrugated cardboard packaging",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Used for packaging once packaging mass is extracted",
    },
]

TRANSPORT_FACTOR_ROWS = [
    {
        "factor_key": "transport_truck_generic",
        "mode": "truck",
        "unit": "ton-km",
        "value_kg_co2e_per_ton_km": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Curate a standard freight truck factor for inbound and outbound transport legs",
    },
    {
        "factor_key": "transport_ship_generic",
        "mode": "ship",
        "unit": "ton-km",
        "value_kg_co2e_per_ton_km": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Curate a standard ocean freight factor if the product distribution route uses shipping",
    },
    {
        "factor_key": "transport_rail_generic",
        "mode": "rail",
        "unit": "ton-km",
        "value_kg_co2e_per_ton_km": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Curate a standard rail freight factor if needed",
    },
    {
        "factor_key": "transport_air_generic",
        "mode": "air",
        "unit": "ton-km",
        "value_kg_co2e_per_ton_km": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Curate only if air transport is explicitly in scope",
    },
]

END_OF_LIFE_FACTOR_ROWS = [
    {
        "factor_key": "eol_recycling_mixed_electronics",
        "route": "recycling",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Use for recovered printer body fractions once recycling split is confirmed",
    },
    {
        "factor_key": "eol_incineration_mixed_electronics",
        "route": "incineration",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Use only if the declared scenario includes combustion",
    },
    {
        "factor_key": "eol_landfill_mixed_electronics",
        "route": "landfill",
        "unit": "kg",
        "value_kg_co2e_per_kg": "",
        "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "status": "needs_manual_curation",
        "notes": "Use for residual waste once the end-of-life split is known",
    },
]

EF_FLOW_MAP = {
    "schema_version": "0.1",
    "notes": [
        "This file maps canonical calculator categories to curated rows or flow names in the EF workbook.",
        "Do not use the raw workbook directly at runtime.",
        "Leave entries null until the factor selection is manually reviewed.",
    ],
    "raw_materials": {
        "plastics_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "steel_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "electronics_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "elastomers_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
    },
    "transport": {
        "transport_truck_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "transport_ship_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "transport_rail_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "transport_air_generic": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
    },
    "end_of_life": {
        "eol_recycling_mixed_electronics": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "eol_incineration_mixed_electronics": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
        "eol_landfill_mixed_electronics": {
            "target_flow_name": None,
            "target_location": None,
            "target_method": "Climate change",
            "status": "needs_manual_curation",
            "source_ref": "totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        },
    },
}


@dataclass
class PdfExtract:
    source_id: str
    input_path: str
    output_path: str | None
    status: str
    characters: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized carbon-footprint assets.")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "totalCarbonfootprintcalculation"),
        help="Directory containing raw PDFs and workbooks.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "backend" / "data" / "carbon"),
        help="Directory where normalized outputs should be written.",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "products": out_dir / "products",
        "factors": out_dir / "factors",
        "mappings": out_dir / "mappings",
        "extracted": out_dir / "extracted",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_pdf_text(path: Path) -> tuple[str, str | None]:
    if PdfReader is None:
        return "", "pypdf_not_available"
    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        return text, None
    except Exception as exc:  # pragma: no cover - depends on local cryptography/pdf state
        return "", f"{type(exc).__name__}: {exc}"


def sanitize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_") or "source"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def extract_pdf_sources(raw_dir: Path, extracted_dir: Path) -> list[PdfExtract]:
    pdf_specs = [
        ("raw_calc_process", "Calculation process .pdf"),
        ("raw_ontology_pdf", "Ontology.pdf"),
        ("raw_lexmark_epd_note", "env-epd_21_1683665824.pdf"),
        ("raw_lcd_monitor_example", "LCD monitor.pdf"),
        ("raw_product_pdf", "Lexmark MX431adn.pdf"),
    ]
    extracts: list[PdfExtract] = []
    for source_id, filename in pdf_specs:
        path = raw_dir / filename
        if not path.exists():
            extracts.append(
                PdfExtract(
                    source_id=source_id,
                    input_path=str(path.relative_to(ROOT)),
                    output_path=None,
                    status="missing",
                    characters=0,
                    error="source_file_missing",
                )
            )
            continue
        text, error = load_pdf_text(path)
        if error:
            extracts.append(
                PdfExtract(
                    source_id=source_id,
                    input_path=str(path.relative_to(ROOT)),
                    output_path=None,
                    status="unreadable",
                    characters=0,
                    error=error,
                )
            )
            continue
        out_path = extracted_dir / f"{sanitize_stem(filename)}.txt"
        write_text(out_path, text)
        extracts.append(
            PdfExtract(
                source_id=source_id,
                input_path=str(path.relative_to(ROOT)),
                output_path=str(out_path.relative_to(ROOT)),
                status="extracted",
                characters=len(text),
                error=None,
            )
        )
    return extracts


def workbook_sheet_targets(zf: ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    out: dict[str, str] = {}
    sheets = wb.find(f"{NS}sheets")
    if sheets is None:
        return out
    for sheet in sheets:
        name = sheet.attrib["name"]
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        out[name] = rel_map[rid]
    return out


def shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{NS}t")) for si in root.findall(f"{NS}si")]


def cell_value(cell: ET.Element, sst: list[str]) -> str | None:
    value_node = cell.find(f"{NS}v")
    if value_node is None:
        inline = cell.find(f"{NS}is")
        if inline is not None:
            return "".join(t.text or "" for t in inline.iter(f"{NS}t"))
        return None
    raw = value_node.text
    if raw is None:
        return None
    if cell.attrib.get("t") == "s":
        try:
            return sst[int(raw)]
        except Exception:
            return raw
    return raw


def sheet_rows(zf: ZipFile, target: str) -> list[list[str | None]]:
    sst = shared_strings(zf)
    root = ET.fromstring(zf.read(f"xl/{target}"))
    data = root.find(f"{NS}sheetData")
    if data is None:
        return []
    rows: list[list[str | None]] = []
    for row in data.findall(f"{NS}row"):
        rows.append([cell_value(cell, sst) for cell in row.findall(f"{NS}c")])
    return rows


def build_electricity_factors(raw_dir: Path, out_csv: Path) -> dict[str, object]:
    workbook_path = raw_dir / "CoM-Emission-factors-for-national-electricity-2024.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(workbook_path)
    rows_written = 0
    skipped_non_numeric = 0
    countries: set[str] = set()
    years: set[int] = set()
    with ZipFile(workbook_path) as zf, out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "factor_key",
                "country_code",
                "country_name",
                "year",
                "method",
                "unit",
                "value_kg_co2e_per_kwh",
                "source_ref",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        sheet_map = workbook_sheet_targets(zf)
        targets = [
            ("Table3_EU_LC", "CoM Table3 EU LC GHG", "kg CO2e/kWh"),
            ("Table6_Other_LC_GHG", "CoM Table6 Other LC GHG", "kg CO2e/kWh"),
        ]
        for sheet_name, method_label, unit in targets:
            target = sheet_map.get(sheet_name)
            if target is None:
                continue
            rows = sheet_rows(zf, target)
            if len(rows) < 3:
                continue
            header = rows[1]
            year_columns: list[tuple[int, int]] = []
            for idx, value in enumerate(header[2:], start=2):
                if value is None:
                    continue
                try:
                    year_columns.append((idx, int(value)))
                except Exception:
                    continue
            for row in rows[2:]:
                if len(row) < 2:
                    continue
                code = row[0]
                name = row[1]
                if not code or not name:
                    continue
                for idx, year in year_columns:
                    if idx >= len(row):
                        continue
                    value = row[idx]
                    if value in (None, ""):
                        continue
                    try:
                        numeric_value = float(str(value))
                    except Exception:
                        skipped_non_numeric += 1
                        continue
                    factor_key = f"electricity_{str(code).lower()}_{year}_lc"
                    writer.writerow(
                        {
                            "factor_key": factor_key,
                            "country_code": code,
                            "country_name": name,
                            "year": year,
                            "method": method_label,
                            "unit": unit,
                            "value_kg_co2e_per_kwh": numeric_value,
                            "source_ref": str(workbook_path.relative_to(ROOT)),
                            "status": "grounded",
                            "notes": f"Extracted from {sheet_name} year column {year}",
                        }
                    )
                    rows_written += 1
                    countries.add(str(code))
                    years.add(year)
    return {
        "source": str(workbook_path.relative_to(ROOT)),
        "rows_written": rows_written,
        "country_count": len(countries),
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "skipped_non_numeric": skipped_non_numeric,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_sample_bootstrap(sample_path: Path) -> dict[str, object]:
    text = sample_path.read_text(encoding="utf-8")
    function_match = re.search(r"^Function:\s*(.+)$", text, re.MULTILINE)
    weight_match = re.search(r"Unboxed weight:\s*~(\d+)[–-](\d+)\s*kg", text)
    mix_match = re.search(
        r"Plastics ~(\d+) %; Steel ~(\d+) %; Electronics ~(\d+) %; Elastomers ~(\d+) %; Others ~(\d+) %",
        text,
    )

    circularity_notes = []
    eol_section = re.search(
        r"10\) END[^\n]*\n-+\n(?P<body>.*?)(?:\n11\)|\Z)",
        text,
        re.DOTALL,
    )
    if eol_section:
        for line in eol_section.group("body").splitlines():
            line = line.strip()
            if line.startswith("•"):
                circularity_notes.append(line.lstrip("•").strip())

    bootstrap: dict[str, object] = {
        "status": "estimated",
        "warning": "Illustrative bootstrap values only. Do not use for final reporting.",
        "based_on_source_ref_id": "sample_mx622_profile",
        "notes": [
            "This is the nearest structured Lexmark monochrome MFP sample found in the repo.",
            "These values are kept separate from official life_cycle_inputs so later extraction can replace them cleanly.",
        ],
        "unboxed_weight_kg_range": None,
        "material_mix_pct": None,
        "circularity_notes": circularity_notes,
    }
    if function_match:
        bootstrap["declared_function"] = function_match.group(1).strip()
    if weight_match:
        bootstrap["unboxed_weight_kg_range"] = [float(weight_match.group(1)), float(weight_match.group(2))]
    if mix_match:
        bootstrap["material_mix_pct"] = {
            "plastics_generic": float(mix_match.group(1)),
            "steel_generic": float(mix_match.group(2)),
            "electronics_generic": float(mix_match.group(3)),
            "elastomers_generic": float(mix_match.group(4)),
            "other_generic": float(mix_match.group(5)),
        }
    return bootstrap


def build_product_profile(
    raw_dir: Path,
    pdf_extracts: list[PdfExtract],
    sample_path: Path,
    out_path: Path,
) -> dict[str, object]:
    source_status = {item.source_id: item for item in pdf_extracts}
    bootstrap = parse_sample_bootstrap(sample_path)
    declared_function = bootstrap.get("declared_function", "Monochrome laser MFP")

    product = {
        "schema_version": "0.1",
        "product_id": "lexmark_mx431adn",
        "display_name": "Lexmark MX431adn",
        "brand": "Lexmark",
        "model": "MX431adn",
        "category": "monochrome_laser_mfp",
        "calculation_scope": {
            "default_basis": "per_printer_lifetime_excluding_paper",
            "supported_basis": [
                "per_printer_lifetime_excluding_paper",
                "per_printer_lifetime_including_paper",
            ],
            "default_reporting_unit": "kg CO2e",
            "notes": [
                "The Lexmark EPD explainer describes printer lifetime reporting with and without paper.",
                "The exact MX431adn functional unit still needs official product or EPD confirmation.",
            ],
        },
        "defaults": {
            "report_year": 2021,
            "electricity_factor_key": None,
            "use_country_code": None,
            "include_paper": False,
            "lifetime_years": None,
            "annual_energy_kwh": None,
            "lifetime_pages": None,
        },
        "identity": {
            "manufacturer": "Lexmark",
            "product_family": "MX",
            "declared_function": declared_function,
            "official_spec_extracted": False,
        },
        "source_refs": [
            {
                "id": "raw_product_pdf",
                "path": str((raw_dir / "Lexmark MX431adn.pdf").relative_to(ROOT)),
                "kind": "official_product_pdf",
                "extract_status": source_status.get("raw_product_pdf").status if source_status.get("raw_product_pdf") else "unknown",
            },
            {
                "id": "raw_lexmark_epd_note",
                "path": str((raw_dir / "env-epd_21_1683665824.pdf").relative_to(ROOT)),
                "kind": "lexmark_epd_explainer",
                "extract_status": source_status.get("raw_lexmark_epd_note").status if source_status.get("raw_lexmark_epd_note") else "unknown",
            },
            {
                "id": "raw_calc_process",
                "path": str((raw_dir / "Calculation process .pdf").relative_to(ROOT)),
                "kind": "calculation_process_note",
                "extract_status": source_status.get("raw_calc_process").status if source_status.get("raw_calc_process") else "unknown",
            },
            {
                "id": "raw_ontology_pdf",
                "path": str((raw_dir / "Ontology.pdf").relative_to(ROOT)),
                "kind": "carbon_ontology_note",
                "extract_status": source_status.get("raw_ontology_pdf").status if source_status.get("raw_ontology_pdf") else "unknown",
            },
            {
                "id": "raw_electricity_workbook",
                "path": str((raw_dir / "CoM-Emission-factors-for-national-electricity-2024.xlsx").relative_to(ROOT)),
                "kind": "electricity_factor_workbook",
                "extract_status": "normalized_to_csv",
            },
            {
                "id": "raw_ef_workbook",
                "path": str((raw_dir / "EF-LCIAMethod_CF(EF-v3.1).xlsx").relative_to(ROOT)),
                "kind": "generic_lcia_flow_workbook",
                "extract_status": "sheet_structure_inspected",
            },
            {
                "id": "sample_mx622_profile",
                "path": str(sample_path.relative_to(ROOT)),
                "kind": "nearest_structured_bootstrap_reference",
                "extract_status": "parsed_text",
            },
        ],
        "life_cycle_inputs": {
            "raw_materials": [
                {
                    "material_key": "plastics_generic",
                    "mass_kg": None,
                    "share_mass_pct": None,
                    "status": "pending_official_extraction",
                    "source_ref_ids": ["raw_product_pdf"],
                },
                {
                    "material_key": "steel_generic",
                    "mass_kg": None,
                    "share_mass_pct": None,
                    "status": "pending_official_extraction",
                    "source_ref_ids": ["raw_product_pdf"],
                },
                {
                    "material_key": "electronics_generic",
                    "mass_kg": None,
                    "share_mass_pct": None,
                    "status": "pending_official_extraction",
                    "source_ref_ids": ["raw_product_pdf"],
                },
                {
                    "material_key": "elastomers_generic",
                    "mass_kg": None,
                    "share_mass_pct": None,
                    "status": "pending_official_extraction",
                    "source_ref_ids": ["raw_product_pdf"],
                },
            ],
            "transport_legs": [
                {
                    "leg_id": "factory_to_market",
                    "mode": None,
                    "distance_km": None,
                    "mass_kg": None,
                    "status": "pending_official_extraction",
                    "source_ref_ids": ["raw_product_pdf"],
                }
            ],
            "use_phase": {
                "annual_energy_kwh": None,
                "country_code": None,
                "lifetime_years": None,
                "lifetime_pages": None,
                "include_paper_default": False,
                "status": "pending_official_extraction",
                "source_ref_ids": ["raw_product_pdf", "raw_lexmark_epd_note"],
            },
            "end_of_life": {
                "recycling_rate_pct": None,
                "incineration_rate_pct": None,
                "landfill_rate_pct": None,
                "status": "pending_official_extraction",
                "source_ref_ids": ["raw_product_pdf", "raw_lexmark_epd_note"],
            },
        },
        "bootstrap_estimates": bootstrap,
    }

    out_path.write_text(json.dumps(product, indent=2), encoding="utf-8")
    return product


def inspect_workbook(raw_dir: Path, filename: str) -> dict[str, object]:
    path = raw_dir / filename
    if not path.exists():
        return {"source": str(path.relative_to(ROOT)), "status": "missing", "sheet_names": []}
    with ZipFile(path) as zf:
        names = workbook_sheet_targets(zf)
    return {
        "source": str(path.relative_to(ROOT)),
        "status": "inspected",
        "sheet_names": list(names.keys()),
    }


def write_source_extracts(path: Path, extracts: list[PdfExtract], workbook_info: dict[str, object]) -> None:
    payload = {
        "pdf_sources": [item.__dict__ for item in extracts],
        "workbooks": workbook_info,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(
    path: Path,
    out_dir: Path,
    pdf_extracts: list[PdfExtract],
    electricity_info: dict[str, object],
    ef_info: dict[str, object],
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "pdf_extracts": [item.__dict__ for item in pdf_extracts],
        "electricity_factor_build": electricity_info,
        "ef_workbook_inspection": ef_info,
        "notes": [
            "The encrypted Lexmark MX431adn PDF remains unreadable with the current local PDF stack.",
            "Electricity factors are grounded and normalized from the CoM workbook.",
            "Raw-material, transport, and end-of-life factors remain curated placeholders until specific EF flow mappings are approved.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    paths = ensure_dirs(out_dir)

    pdf_extracts = extract_pdf_sources(raw_dir, paths["extracted"])

    electricity_info = build_electricity_factors(
        raw_dir,
        paths["factors"] / "electricity_lc_factors.csv",
    )

    write_csv(
        paths["factors"] / "raw_material_factors.csv",
        [
            "factor_key",
            "material_label",
            "unit",
            "value_kg_co2e_per_kg",
            "source_ref",
            "status",
            "notes",
        ],
        RAW_MATERIAL_FACTOR_ROWS,
    )
    write_csv(
        paths["factors"] / "transport_factors.csv",
        [
            "factor_key",
            "mode",
            "unit",
            "value_kg_co2e_per_ton_km",
            "source_ref",
            "status",
            "notes",
        ],
        TRANSPORT_FACTOR_ROWS,
    )
    write_csv(
        paths["factors"] / "end_of_life_factors.csv",
        [
            "factor_key",
            "route",
            "unit",
            "value_kg_co2e_per_kg",
            "source_ref",
            "status",
            "notes",
        ],
        END_OF_LIFE_FACTOR_ROWS,
    )

    mapping_path = paths["mappings"] / "ef_flow_map.yml"
    mapping_lines = [
        'schema_version: "0.1"',
        "notes:",
        "  - This file maps canonical calculator categories to curated rows or flow names in the EF workbook.",
        "  - Do not use the raw workbook directly at runtime.",
        "  - Leave entries null until the factor selection is manually reviewed.",
        "",
        "raw_materials:",
        "  plastics_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  steel_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  electronics_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  elastomers_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "",
        "transport:",
        "  transport_truck_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  transport_ship_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  transport_rail_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  transport_air_generic:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "",
        "end_of_life:",
        "  eol_recycling_mixed_electronics:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  eol_incineration_mixed_electronics:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "  eol_landfill_mixed_electronics:",
        "    target_flow_name: null",
        "    target_location: null",
        '    target_method: "Climate change"',
        "    status: needs_manual_curation",
        "    source_ref: totalCarbonfootprintcalculation/EF-LCIAMethod_CF(EF-v3.1).xlsx",
        "",
    ]
    mapping_path.write_text("\n".join(mapping_lines), encoding="utf-8")

    build_product_profile(
        raw_dir=raw_dir,
        pdf_extracts=pdf_extracts,
        sample_path=ROOT / "tests" / "lexmark" / "docs" / "lexmark_mx622adhe_dpp_full_02.txt",
        out_path=paths["products"] / "lexmark_mx431adn.json",
    )

    ef_info = inspect_workbook(raw_dir, "EF-LCIAMethod_CF(EF-v3.1).xlsx")
    workbook_info = {
        "electricity_workbook": electricity_info,
        "ef_workbook": ef_info,
    }
    write_source_extracts(paths["extracted"] / "source_extracts.json", pdf_extracts, workbook_info)
    write_manifest(out_dir / "build_manifest.json", out_dir, pdf_extracts, electricity_info, ef_info)

    print(f"Wrote normalized carbon assets to {out_dir}")
    print(
        "Electricity factors:",
        electricity_info["rows_written"],
        "rows",
        f"({electricity_info['country_count']} countries, {electricity_info['min_year']}..{electricity_info['max_year']})",
    )
    unreadable = [item for item in pdf_extracts if item.status == "unreadable"]
    if unreadable:
        print("Unreadable PDFs:")
        for item in unreadable:
            print(f"- {item.input_path}: {item.error}")


if __name__ == "__main__":
    main()
