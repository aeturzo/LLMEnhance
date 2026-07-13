from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.carbon_calculation_service import (
    CarbonCalculationResult,
    CarbonStageResult,
    DEFAULT_DATA_ROOT,
    calculate_carbon_footprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARBON_CORPUS_DIR = Path(DEFAULT_DATA_ROOT) / "corpus"
DEFAULT_CARBON_CORPUS_PATH = DEFAULT_CARBON_CORPUS_DIR / "carbon_docs.jsonl"
DEFAULT_CARBON_CORPUS_MANIFEST = DEFAULT_CARBON_CORPUS_DIR / "carbon_docs_manifest.json"
ALL_CARBON_STAGES = ["raw_materials", "transportation", "use_phase", "end_of_life"]

STAGE_LABELS = {
    "raw_materials": "raw-material",
    "transportation": "transportation",
    "use_phase": "use-phase",
    "end_of_life": "end-of-life",
}


def format_number(value: Optional[float], decimals: int = 3) -> str:
    if value is None:
        return "unavailable"
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text or "0"


def friendly_stage_name(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.replace("_", " "))


def carbon_asset_id(kind: str, product_id: str, scenario_name: str = "runtime") -> str:
    return f"carbon_{scenario_name}_{kind}_{product_id}"


def _source_refs_text(source_refs: List[str], limit: int = 4) -> str:
    if not source_refs:
        return "No normalized source references were recorded."
    shown = source_refs[:limit]
    extra = len(source_refs) - len(shown)
    text = "; ".join(shown)
    if extra > 0:
        text = f"{text}; +{extra} more"
    return f"Source refs: {text}."


def _quality_text(result: CarbonCalculationResult) -> str:
    quality = result.quality_status.replace("_", " ")
    if result.uncertainty_pct is not None and result.uncertainty_kg_co2e is not None:
        return (
            f"Result quality is {quality}. Approximate uncertainty is "
            f"+/- {format_number(result.uncertainty_kg_co2e)} kg CO2e "
            f"({format_number(result.uncertainty_pct, decimals=1)}%)."
        )
    return f"Result quality is {quality}."


def _provenance_text(result: CarbonCalculationResult, limit: int = 4) -> str:
    if not result.provenance:
        return "No structured provenance summary is available."
    chunks: List[str] = []
    for item in result.provenance[:limit]:
        value_text = item.value if not isinstance(item.value, dict) else json.dumps(item.value, ensure_ascii=False, sort_keys=True)
        source_text = "; ".join(item.source_refs[:2]) if item.source_refs else "no recorded source"
        chunks.append(
            f"{item.label}: {value_text} {item.unit} ({item.status}; method {item.method}; source {source_text})"
        )
    extra = len(result.provenance) - min(len(result.provenance), limit)
    text = " ".join(chunks)
    if extra > 0:
        text += f" +{extra} more provenance items."
    return text


def _trace_text(stage_result: CarbonStageResult, max_items: int = 4) -> str:
    lines: List[str] = []
    for trace in stage_result.traces[:max_items]:
        if trace.emissions_kg_co2e is not None:
            lines.append(
                f"{trace.label}: {trace.formula} = {format_number(trace.emissions_kg_co2e)} kg CO2e."
            )
        else:
            lines.append(
                f"{trace.label}: status {trace.status}; notes: {'; '.join(trace.notes[:2]) or 'missing inputs'}."
            )
    return " ".join(lines)


def _base_doc(
    *,
    product_id: str,
    scenario_name: str,
    kind: str,
    title: str,
    text: str,
    status: str,
    stage: str = "",
    source_refs: Optional[List[str]] = None,
    score: float = 1.0,
) -> Dict[str, Any]:
    pid = carbon_asset_id(kind, product_id, scenario_name=scenario_name)
    return {
        "id": pid,
        "pid": pid,
        "doc_id": f"carbon_answer_{product_id}_{scenario_name}",
        "domain": "carbon",
        "title": title,
        "text": text,
        "product_id": product_id,
        "scenario_name": scenario_name,
        "answer_kind": kind,
        "stage": stage,
        "status": status,
        "source_refs": source_refs or [],
        "score": score,
        "source": "carbon_answer_assets",
    }


def build_carbon_answer_assets(
    result: CarbonCalculationResult,
    scenario_name: str = "runtime",
) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    product_id = result.product_id
    product_name = result.product_name
    source_refs = list(result.source_refs)

    stage_bits = []
    for stage_name in ALL_CARBON_STAGES:
        stage = result.stage_results.get(stage_name)
        if stage is None:
            continue
        stage_bits.append(
            f"{friendly_stage_name(stage_name)} status {stage.status} with total {format_number(stage.total_kg_co2e)} kg CO2e"
        )

    overview_parts = [
        f"Product {product_name}.",
        f"Calculation status is {result.status}.",
        f"Requested stages: {', '.join(result.requested_stages)}.",
    ]
    overview_parts.append(_quality_text(result))
    if result.total_kg_co2e is not None:
        overview_parts.append(f"Total carbon footprint is {format_number(result.total_kg_co2e)} kg CO2e.")
    else:
        overview_parts.append("A full total carbon footprint is not currently available.")
    if result.partial_total_kg_co2e:
        overview_parts.append(f"Partial computed total is {format_number(result.partial_total_kg_co2e)} kg CO2e.")
    if stage_bits:
        overview_parts.append("Stage overview: " + "; ".join(stage_bits) + ".")
    if result.used_bootstrap_estimates:
        overview_parts.append("Bootstrap estimates were enabled for this calculation.")
    if result.assumptions:
        overview_parts.append("Assumptions: " + "; ".join(result.assumptions[:4]) + ".")
    if result.estimated_fields:
        overview_parts.append("Estimated inputs: " + "; ".join(result.estimated_fields[:6]) + ".")
    overview_parts.append("Provenance summary: " + _provenance_text(result) + ".")
    overview_parts.append(_source_refs_text(source_refs))
    passages.append(
        _base_doc(
            product_id=product_id,
            scenario_name=scenario_name,
            kind="overview",
            title=f"Carbon overview for {product_name}",
            text=" ".join(overview_parts).strip(),
            status=result.status,
            source_refs=source_refs,
            score=1.0,
        )
    )

    total_parts = [f"Total carbon footprint answer asset for {product_name}."]
    total_parts.append(_quality_text(result))
    if result.total_kg_co2e is not None:
        total_parts.append(f"Total product carbon footprint is {format_number(result.total_kg_co2e)} kg CO2e.")
        if result.quality_status != "exact":
            total_parts.append("This is not an exact official total; estimated inputs contributed to the calculation.")
    else:
        total_parts.append("Total product carbon footprint cannot be stated as a complete figure yet.")
        total_parts.append(f"Current partial computed total is {format_number(result.partial_total_kg_co2e)} kg CO2e.")
        if result.missing_inputs:
            total_parts.append("Missing inputs: " + "; ".join(result.missing_inputs[:8]) + ".")
    if result.estimated_fields:
        total_parts.append("Estimated inputs: " + "; ".join(result.estimated_fields[:6]) + ".")
    total_parts.append("Provenance summary: " + _provenance_text(result) + ".")
    total_parts.append(_source_refs_text(source_refs))
    passages.append(
        _base_doc(
            product_id=product_id,
            scenario_name=scenario_name,
            kind="total",
            title=f"Total carbon footprint for {product_name}",
            text=" ".join(total_parts).strip(),
            status=result.status,
            source_refs=source_refs,
            score=0.99,
        )
    )

    for stage_name in ALL_CARBON_STAGES:
        stage = result.stage_results.get(stage_name)
        if stage is None:
            continue
        stage_parts = [
            f"{friendly_stage_name(stage_name).capitalize()} answer asset for {product_name}.",
            f"Stage status is {stage.status}.",
            f"Stage total is {format_number(stage.total_kg_co2e)} kg CO2e.",
        ]
        if stage.quality_status:
            stage_parts.append(f"Stage quality is {stage.quality_status.replace('_', ' ')}.")
        if stage.uncertainty_pct is not None:
            stage_parts.append(f"Approximate stage uncertainty is {format_number(stage.uncertainty_pct, decimals=1)} percent.")
        if stage.estimated_inputs:
            stage_parts.append("Estimated stage inputs: " + "; ".join(stage.estimated_inputs[:6]) + ".")
        trace_text = _trace_text(stage)
        if trace_text:
            stage_parts.append(trace_text)
        if stage.missing_inputs:
            stage_parts.append("Missing inputs: " + "; ".join(stage.missing_inputs[:6]) + ".")
        passages.append(
            _base_doc(
                product_id=product_id,
                scenario_name=scenario_name,
                kind=stage_name,
                title=f"{friendly_stage_name(stage_name).capitalize()} carbon answer asset for {product_name}",
                text=" ".join(stage_parts).strip(),
                status=stage.status,
                stage=stage_name,
                source_refs=source_refs,
                score=0.95,
            )
        )

    recyclability = result.recyclability
    recyclability_parts = [
        f"Recyclability answer asset for {product_name}.",
        f"Recyclability status is {recyclability.status}.",
    ]
    if recyclability.recyclability_pct is not None:
        recyclability_parts.append(
            f"Recycling rate is {format_number(recyclability.recyclability_pct, decimals=1)} percent."
        )
    if recyclability.recoverable_mass_kg is not None:
        recyclability_parts.append(f"Recoverable mass is {format_number(recyclability.recoverable_mass_kg)} kg.")
    if recyclability.incineration_mass_kg is not None:
        recyclability_parts.append(f"Incineration mass is {format_number(recyclability.incineration_mass_kg)} kg.")
    if recyclability.landfill_mass_kg is not None:
        recyclability_parts.append(f"Landfill mass is {format_number(recyclability.landfill_mass_kg)} kg.")
    if recyclability.notes:
        recyclability_parts.append("Notes: " + "; ".join(recyclability.notes[:4]) + ".")
    passages.append(
        _base_doc(
            product_id=product_id,
            scenario_name=scenario_name,
            kind="recyclability",
            title=f"Recyclability answer asset for {product_name}",
            text=" ".join(recyclability_parts).strip(),
            status=recyclability.status,
            stage="end_of_life",
            source_refs=source_refs,
            score=0.9,
        )
    )

    if result.missing_inputs:
        passages.append(
            _base_doc(
                product_id=product_id,
                scenario_name=scenario_name,
                kind="missing_inputs",
                title=f"Missing carbon inputs for {product_name}",
                text=(
                    f"Missing calculation inputs for {product_name}: "
                    + "; ".join(result.missing_inputs[:10])
                    + "."
                ),
                status="partial",
                source_refs=source_refs,
                score=0.85,
            )
        )

    return passages


def build_default_carbon_corpus(
    product_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    products_dir = Path(DEFAULT_DATA_ROOT) / "products"
    if product_ids is None:
        product_ids = sorted(path.stem for path in products_dir.glob("*.json"))

    rows: List[Dict[str, Any]] = []
    products_summary: List[Dict[str, Any]] = []
    for product_id in product_ids:
        result = calculate_carbon_footprint(product_id, {"requested_stages": ALL_CARBON_STAGES})
        docs = build_carbon_answer_assets(result, scenario_name="official_default")
        rows.extend(docs)
        products_summary.append(
            {
                "product_id": product_id,
                "status": result.status,
                "requested_stages": list(result.requested_stages),
                "doc_count": len(docs),
                "missing_input_count": len(result.missing_inputs),
                "stage_statuses": {stage: res.status for stage, res in result.stage_results.items()},
                "asset_kinds": [doc["answer_kind"] for doc in docs],
            }
        )

    manifest = {
        "schema_version": "0.1",
        "corpus_path": str(DEFAULT_CARBON_CORPUS_PATH),
        "product_count": len(products_summary),
        "doc_count": len(rows),
        "products": products_summary,
    }
    return {"rows": rows, "manifest": manifest}


def write_carbon_corpus(
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    corpus_path: Path | str = DEFAULT_CARBON_CORPUS_PATH,
    manifest_path: Path | str = DEFAULT_CARBON_CORPUS_MANIFEST,
) -> None:
    corpus_path = Path(corpus_path)
    manifest_path = Path(manifest_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
