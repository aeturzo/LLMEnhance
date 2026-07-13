from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from backend.api.carbon_models import CarbonRequest, CarbonResult
from backend.services.carbon_calculation_service import DEFAULT_DATA_ROOT, calculate_carbon_footprint
from backend.services.carbon_ontology_service import build_carbon_ontology_sidecar


router = APIRouter(prefix="/carbon", tags=["carbon"])


def _available_product_ids() -> List[str]:
    products_dir = Path(DEFAULT_DATA_ROOT) / "products"
    if not products_dir.exists():
        return []
    return sorted(path.stem for path in products_dir.glob("*.json"))


@router.get("/products", response_model=List[str], summary="List supported carbon products")
def list_carbon_products() -> List[str]:
    return _available_product_ids()


@router.post("/calculate", response_model=CarbonResult, summary="Run a deterministic carbon calculation")
def calculate_carbon(req: CarbonRequest) -> CarbonResult:
    try:
        result = calculate_carbon_footprint(req.product_id, req.to_service_scenario())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    ontology_sidecar = None
    if req.include_ontology_sidecar:
        try:
            ontology_sidecar = build_carbon_ontology_sidecar(
                result=result,
                include_turtle=req.include_ontology_turtle,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OntologySidecarError: {type(exc).__name__}: {exc}") from exc

    return CarbonResult.from_service_result(
        result,
        include_trace=req.include_trace,
        ontology_sidecar=ontology_sidecar,
    )
