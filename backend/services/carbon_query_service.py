from __future__ import annotations

import importlib.util
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from backend.services.carbon_answer_assets_service import (
    ALL_CARBON_STAGES,
    build_carbon_answer_assets,
    carbon_asset_id,
    format_number,
    friendly_stage_name,
)
from backend.services.carbon_calculation_service import (
    CarbonCalculationResult,
    calculate_carbon_footprint,
)

CARBON_TRIGGER_TERMS = [
    "carbon footprint",
    "co2",
    "co2e",
    "carbon emission",
    "carbon emissions",
    "emission",
    "emissions",
    "ghg",
    "greenhouse gas",
    "environmental factor",
    "environment factor",
]

RECYCLABILITY_TERMS = [
    "recyclability",
    "recyclable",
    "recycle",
    "recycling",
    "end of life",
    "end-of-life",
    "landfill",
    "incineration",
    "disposal",
]

BREAKDOWN_TERMS = [
    "breakdown",
    "by stage",
    "stagewise",
    "stage-wise",
    "raw materials and transportation",
]

RAW_MATERIAL_TERMS = [
    "raw material",
    "raw materials",
    "material emission",
    "material emissions",
    "materials footprint",
]

TRANSPORT_TERMS = [
    "transport",
    "transportation",
    "shipping",
    "shipment",
    "logistics",
    "freight",
]

USE_PHASE_TERMS = [
    "use phase",
    "use-phase",
    "electricity",
    "energy use",
    "operational",
    "operation emissions",
]

ESTIMATE_TERMS = [
    "estimate",
    "estimated",
    "approx",
    "approximate",
    "roughly",
    "bootstrap",
]

STRICT_EXACT_TERMS = [
    "exact only",
    "official only",
    "strict only",
    "no estimate",
    "without estimate",
]

CARBON_PRODUCT_ALIASES = {
    "lexmark_mx431adn": "lexmark_mx431adn",
    "lexmark mx431adn": "lexmark_mx431adn",
    "mx431adn": "lexmark_mx431adn",
    "mx431": "lexmark_mx431adn",
}

_ANSWERER_MODULE = None


def _normalize_text(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _get_answerer():
    global _ANSWERER_MODULE
    if _ANSWERER_MODULE is not None:
        return _ANSWERER_MODULE

    path = Path(__file__).resolve().parents[1] / "api" / "answerer_ctx.py"
    spec = importlib.util.spec_from_file_location("carbon_answerer_ctx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load answerer context module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ANSWERER_MODULE = module
    return module


def is_carbon_query(query: str, mode: Optional[str] = None) -> bool:
    if (mode or "").strip().upper() == "CARBON":
        return True

    text = _normalize_text(query)
    if not text:
        return False

    # DPP/passport field lookups often contain words such as "recyclable" or
    # "emissions" as ordinary document labels. Those must stay in the normal
    # retrieval path unless the caller explicitly selected CARBON mode.
    if text.startswith("according to ") or re.search(r"\b(?:battery|lexmark|viessmann) seed \d{4}\b", text):
        return False

    if any(term in text for term in RECYCLABILITY_TERMS):
        return True

    return any(term in text for term in CARBON_TRIGGER_TERMS)


def _resolve_product_id(product: Optional[str], query: str) -> Optional[str]:
    candidates = [_normalize_text(product), _normalize_text(query)]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in CARBON_PRODUCT_ALIASES:
            return CARBON_PRODUCT_ALIASES[candidate]
        for alias, product_id in CARBON_PRODUCT_ALIASES.items():
            if alias in candidate:
                return product_id
    return None


def _detect_requested_stages(query: str) -> List[str]:
    text = _normalize_text(query)
    if not text:
        return ALL_CARBON_STAGES.copy()

    if any(term in text for term in BREAKDOWN_TERMS):
        return ALL_CARBON_STAGES.copy()

    stages: List[str] = []
    if any(term in text for term in RAW_MATERIAL_TERMS):
        stages.append("raw_materials")
    if any(term in text for term in TRANSPORT_TERMS):
        stages.append("transportation")
    if any(term in text for term in USE_PHASE_TERMS):
        stages.append("use_phase")
    if any(term in text for term in RECYCLABILITY_TERMS):
        stages.append("end_of_life")

    if stages:
        return _dedupe(stages)
    return ALL_CARBON_STAGES.copy()


def _wants_bootstrap_estimates(query: str) -> bool:
    text = _normalize_text(query)
    return any(term in text for term in ESTIMATE_TERMS)


def _wants_strict_exact(query: str) -> bool:
    text = _normalize_text(query)
    return any(term in text for term in STRICT_EXACT_TERMS)

def _top_snippet_fallback(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return "Insufficient context."
    top = passages[0]
    snippet = (top.get("text") or "")[:220].strip()
    pid = str(top.get("id") or "doc")
    return f"{snippet} [{pid}]"


def _primary_stage(query: str) -> Optional[str]:
    requested = _detect_requested_stages(query)
    if len(requested) == 1:
        return requested[0]
    if any(term in _normalize_text(query) for term in RECYCLABILITY_TERMS):
        return "end_of_life"
    return None


def _fallback_answer(query: str, result: CarbonCalculationResult) -> str:
    product_id = result.product_id
    product_name = result.product_name
    scenario_name = "runtime"
    summary_id = carbon_asset_id("overview", product_id, scenario_name=scenario_name)
    missing_id = carbon_asset_id("missing_inputs", product_id, scenario_name=scenario_name)
    primary_stage = _primary_stage(query)
    text = _normalize_text(query)

    if any(term in text for term in RECYCLABILITY_TERMS):
        recyclability = result.recyclability
        cite = carbon_asset_id("recyclability", product_id, scenario_name=scenario_name)
        if recyclability.recyclability_pct is not None:
            prefix = "The estimated recyclability result" if result.quality_status in ("hybrid_estimate", "partial_estimate") else "The recyclability result"
            answer = (
                f"{prefix} for {product_name} is "
                f"{format_number(recyclability.recyclability_pct, decimals=1)}%."
            )
            if recyclability.recoverable_mass_kg is not None:
                answer += f" Recoverable mass is {format_number(recyclability.recoverable_mass_kg)} kg."
            if result.quality_status in ("hybrid_estimate", "partial_estimate"):
                answer += " The end-of-life split is estimated rather than officially declared."
            return f"{answer} [{cite}]"
        return (
            f"Recyclability for {product_name} cannot be reported yet because the end-of-life split is missing "
            f"from the normalized inputs [{cite}]."
        )

    if primary_stage and primary_stage in result.stage_results:
        stage = result.stage_results[primary_stage]
        cite = carbon_asset_id(primary_stage, product_id, scenario_name=scenario_name)
        if stage.status == "complete":
            prefix = "The estimated" if stage.quality_status == "estimated" else "The"
            answer = (
                f"{prefix} {friendly_stage_name(primary_stage)} emissions for {product_name} are "
                f"{format_number(stage.total_kg_co2e)} kg CO2e"
            )
            if stage.quality_status == "estimated" and stage.uncertainty_pct is not None:
                answer += f" with approximate uncertainty of {format_number(stage.uncertainty_pct, decimals=1)}%"
            return (
                f"{answer} [{cite}]."
            )
        missing_text = "; ".join(stage.missing_inputs[:3]) if stage.missing_inputs else "required inputs are still missing"
        return (
            f"The {friendly_stage_name(primary_stage)} emissions for {product_name} cannot be fully calculated yet "
            f"because {missing_text} [{cite}]."
        )

    if result.total_kg_co2e is not None:
        breakdown = []
        for stage_name in ALL_CARBON_STAGES:
            stage = result.stage_results.get(stage_name)
            if stage is None:
                continue
            breakdown.append(
                f"{friendly_stage_name(stage_name)} {format_number(stage.total_kg_co2e)} kg CO2e"
            )
        if result.quality_status == "exact":
            answer = f"The calculated carbon footprint for {product_name} is {format_number(result.total_kg_co2e)} kg CO2e [{summary_id}]."
        else:
            answer = f"The estimated carbon footprint for {product_name} is {format_number(result.total_kg_co2e)} kg CO2e [{summary_id}]."
            if result.uncertainty_kg_co2e is not None and result.uncertainty_pct is not None:
                answer += (
                    f" Approximate uncertainty is +/- {format_number(result.uncertainty_kg_co2e)} kg CO2e "
                    f"({format_number(result.uncertainty_pct, decimals=1)}%)."
                )
            if result.estimated_fields:
                answer += " Estimated inputs include " + "; ".join(result.estimated_fields[:4]) + "."
        if breakdown:
            answer += " Breakdown: " + "; ".join(breakdown) + "."
        return answer

    if result.missing_inputs:
        missing_text = "; ".join(result.missing_inputs[:4])
        return (
            f"A full carbon footprint for {product_name} cannot be calculated yet from the currently normalized "
            f"inputs. Missing items include {missing_text} [{missing_id}]."
        )

    return f"Carbon calculation for {product_name} is currently incomplete [{summary_id}]."


def _answer_from_passages(question: str, passages: List[Dict[str, Any]], fallback_answer: str) -> tuple[str, Dict[str, Any]]:
    answerer = _get_answerer()
    answer_trace = {
        **answerer.answerer_config(),
        "llm_attempted": False,
        "llm_used": False,
        "provider": None,
        "model": None,
        "api": None,
        "path": "carbon_rule_fallback",
        "passage_count": len(passages),
    }
    if not passages:
        answer_trace["reason"] = "No carbon passages were available."
        return fallback_answer, answer_trace
    try:
        out = answerer.answer_with_context_detailed(question, passages)
        answer = out["answer"]
        answer_trace = out["trace"]
    except Exception as exc:
        answer_trace["reason"] = str(exc)
        return fallback_answer, answer_trace

    if not answer or answer.strip().lower() == "insufficient context.":
        answer_trace["llm_used"] = False
        answer_trace["path"] = "carbon_rule_fallback"
        answer_trace["reason"] = "The answerer returned insufficient context."
        return fallback_answer, answer_trace
    if answer.strip() == _top_snippet_fallback(passages):
        answer_trace["llm_used"] = False
        answer_trace["path"] = "carbon_rule_fallback"
        answer_trace["reason"] = "The answerer fell back to the top snippet, so the deterministic carbon answer was kept."
        return fallback_answer, answer_trace
    return answer, answer_trace


def _confidence(result: CarbonCalculationResult) -> float:
    stage_results = list(result.stage_results.values())
    complete_count = sum(1 for stage in stage_results if stage.status == "complete")
    if result.quality_status in ("hybrid_estimate", "partial_estimate"):
        if result.status == "complete":
            return 0.62
        return 0.45
    if result.status == "complete":
        return 0.75 if result.used_bootstrap_estimates else 0.92
    if complete_count > 0:
        base = 0.65 if not result.used_bootstrap_estimates else 0.58
        return min(base + 0.05 * max(0, complete_count - 1), 0.8)
    if result.status == "partial":
        return 0.35
    return 0.2


def solve_carbon_query(query: str, product: Optional[str], session: str) -> Dict[str, Any]:
    product_id = _resolve_product_id(product, query)
    if product_id is None:
        passages = [
            {
                "id": "carbon_supported_products",
                "title": "Supported carbon products",
                "text": (
                    "Currently supported carbon-calculation product: Lexmark MX431adn "
                    "(product_id lexmark_mx431adn)."
                ),
                "score": 1.0,
            }
        ]
        answer = "Carbon calculation currently supports Lexmark MX431adn only [carbon_supported_products]."
        return {
            "answer": answer,
            "answer_trace": {
                **_get_answerer().answerer_config(),
                "llm_attempted": False,
                "llm_used": False,
                "provider": None,
                "model": None,
                "api": None,
                "path": "unsupported_product",
                "passage_count": len(passages),
                "reason": "The query did not map to a supported carbon product.",
            },
            "steps": [
                {
                    "source": "CARBON",
                    "status": "unsupported_product",
                    "requested_stages": [],
                    "missing_input_count": 0,
                    "used_bootstrap_estimates": False,
                }
            ],
            "sources": passages,
            "carbon": None,
            "mode": "CARBON",
            "session": session,
            "product": product,
            "confidence": 0.2,
        }

    requested_stages = _detect_requested_stages(query)
    exact_only = _wants_strict_exact(query)
    scenario: Dict[str, Any] = {
        "requested_stages": requested_stages,
        "use_bootstrap_estimates": False,
    }
    result = calculate_carbon_footprint(product_id, scenario)
    estimate_fallback_used = False
    if not exact_only and (result.total_kg_co2e is None or result.status != "complete" or _wants_bootstrap_estimates(query)):
        estimated_scenario = {
            "requested_stages": requested_stages,
            "use_bootstrap_estimates": True,
        }
        estimated_result = calculate_carbon_footprint(product_id, estimated_scenario)
        exact_complete_stages = sum(1 for stage in result.stage_results.values() if stage.status == "complete")
        estimated_complete_stages = sum(1 for stage in estimated_result.stage_results.values() if stage.status == "complete")
        if (
            estimated_result.total_kg_co2e is not None
            or estimated_complete_stages > exact_complete_stages
            or len(estimated_result.missing_inputs) < len(result.missing_inputs)
        ):
            result = estimated_result
            estimate_fallback_used = True
    passages = build_carbon_answer_assets(result, scenario_name="runtime")
    fallback_answer = _fallback_answer(query, result)
    answer, answer_trace = _answer_from_passages(query, passages, fallback_answer)

    return {
        "answer": answer,
        "answer_trace": answer_trace,
        "steps": [
            {
                "source": "CARBON",
                "status": result.status,
                "requested_stages": list(result.requested_stages),
                    "missing_input_count": len(result.missing_inputs),
                    "used_bootstrap_estimates": result.used_bootstrap_estimates,
                    "estimate_fallback_used": estimate_fallback_used,
                    "total_kg_co2e": result.total_kg_co2e,
                    "partial_total_kg_co2e": result.partial_total_kg_co2e,
                    "quality_status": result.quality_status,
                    "uncertainty_pct": result.uncertainty_pct,
                    "answer_trace": answer_trace,
                }
            ],
        "sources": [
            {
                "type": "carbon_calc",
                "id": passage["id"],
                "title": passage["title"],
                "score": passage.get("score"),
                "snippet": passage["text"][:220],
            }
            for passage in passages[:8]
        ],
        "carbon": result.as_dict(),
        "mode": "CARBON",
        "session": session,
        "product": product_id,
        "confidence": _confidence(result),
    }
