from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from pydantic import BaseModel
from typing import List, Optional

from ..services.symbolic_reasoning_service import SymbolicReasoner

router = APIRouter(prefix="/symbolic", tags=["symbolic"])

def get_reasoner(request: Request) -> SymbolicReasoner:
    # Uses the instance created in main.py startup
    return request.app.state.reasoner

class ProductInfo(BaseModel):
    product: str
    components: List[str]
    steps: List[str]
    requires_compliance: List[str]
    requires_steps: List[str]


@router.get("/products", response_model=List[str])
def list_products(reasoner: SymbolicReasoner = Depends(get_reasoner)):
    return [p.split("#")[-1] for p in reasoner.list_products()]


@router.get("/product", response_model=ProductInfo)
def product_info(
    product: str = Query(..., description="Product local name, e.g. ProductA"),
    reasoner: SymbolicReasoner = Depends(get_reasoner),
):
    prod_uri = f"{product}"
    products = [p.split("#")[-1] for p in reasoner.list_products()]
    if product not in products:
        raise HTTPException(status_code=404, detail=f"Unknown product '{product}'")

    components = [c.split("#")[-1] for c in reasoner.list_components(prod_uri)]
    steps = [s.split("#")[-1] for s in reasoner.list_process_steps(prod_uri)]
    req_comp = [r.split("#")[-1] for r in reasoner.check_compliance_requirements(prod_uri)]
    req_steps = [r.split("#")[-1] for r in reasoner.suggest_missing_steps(prod_uri)]

    return ProductInfo(
        product=product,
        components=components,
        steps=steps,
        requires_compliance=req_comp,
        requires_steps=req_steps,
    )


@router.get("/check_compliance", response_model=List[str])
def check_compliance(
    product: str = Query(..., description="Product local name, e.g. ProductA"),
    reasoner: SymbolicReasoner = Depends(get_reasoner),
):
    prod_uri = f"{product}"
    products = [p.split("#")[-1] for p in reasoner.list_products()]
    if product not in products:
        raise HTTPException(status_code=404, detail=f"Unknown product '{product}'")
    return [r.split("#")[-1] for r in reasoner.check_compliance_requirements(prod_uri)]


@router.get("/suggest_steps", response_model=List[str])
def suggest_steps(
    product: str = Query(..., description="Product local name, e.g. ProductA"),
    reasoner: SymbolicReasoner = Depends(get_reasoner),
):
    prod_uri = f"{product}"
    products = [p.split("#")[-1] for p in reasoner.list_products()]
    if product not in products:
        raise HTTPException(status_code=404, detail=f"Unknown product '{product}'")
    return [r.split("#")[-1] for r in reasoner.suggest_missing_steps(prod_uri)]
