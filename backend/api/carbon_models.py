from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from backend.services.carbon_calculation_service import (
    CarbonCalculationResult,
    CarbonProvenanceItem,
    CarbonStageResult,
    CarbonTraceItem,
    RecyclabilityResult,
)
from backend.services.carbon_ontology_service import (
    CarbonOntologySidecarResult,
    CarbonOntologyValidationIssue,
    CarbonOntologyValidationReport,
)


CarbonStageName = Literal["raw_materials", "transportation", "use_phase", "end_of_life"]

_STAGE_ALIASES = {
    "materials": "raw_materials",
    "raw_materials": "raw_materials",
    "raw-materials": "raw_materials",
    "transport": "transportation",
    "transportation": "transportation",
    "use": "use_phase",
    "use_phase": "use_phase",
    "use-phase": "use_phase",
    "eol": "end_of_life",
    "end_of_life": "end_of_life",
    "end-of-life": "end_of_life",
}


def _normalize_stage_name(value: str) -> str:
    return _STAGE_ALIASES.get((value or "").strip().lower(), value)


def _model_dump(instance: BaseModel) -> Dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(exclude_none=True)
    return instance.dict(exclude_none=True)


class RawMaterialInput(BaseModel):
    material_key: str
    mass_kg: Optional[float] = None
    share_mass_pct: Optional[float] = None
    factor_key: Optional[str] = None
    factor_value: Optional[float] = None
    factor_value_kg_co2e_per_kg: Optional[float] = None
    source_ref: Optional[str] = None
    source_ref_ids: List[str] = Field(default_factory=list)


class TransportLegInput(BaseModel):
    leg_id: Optional[str] = None
    mode: Optional[str] = None
    mode_key: Optional[str] = None
    distance_km: Optional[float] = None
    mass_kg: Optional[float] = None
    factor_key: Optional[str] = None
    factor_value: Optional[float] = None
    factor_value_kg_co2e_per_ton_km: Optional[float] = None
    source_ref: Optional[str] = None
    source_ref_ids: List[str] = Field(default_factory=list)


class UsePhaseInput(BaseModel):
    annual_energy_kwh: Optional[float] = None
    lifetime_years: Optional[float] = None
    lifetime_energy_kwh: Optional[float] = None
    country_code: Optional[str] = None
    electricity_country_code: Optional[str] = None
    report_year: Optional[int] = None
    electricity_year: Optional[int] = None
    electricity_factor_key: Optional[str] = None
    electricity_factor_value: Optional[float] = None
    electricity_factor_value_kg_co2e_per_kwh: Optional[float] = None
    include_paper_default: Optional[bool] = None
    source_ref: Optional[str] = None
    source_ref_ids: List[str] = Field(default_factory=list)


class EndOfLifeRouteFactorInput(BaseModel):
    factor_key: Optional[str] = None
    factor_value: Optional[float] = None
    factor_value_kg_co2e_per_kg: Optional[float] = None
    source_ref: Optional[str] = None


class EndOfLifeInput(BaseModel):
    mass_kg: Optional[float] = None
    recycling_rate_pct: Optional[float] = None
    recyclability_pct: Optional[float] = None
    incineration_rate_pct: Optional[float] = None
    incineration_pct: Optional[float] = None
    landfill_rate_pct: Optional[float] = None
    landfill_pct: Optional[float] = None
    route_factor_values: Dict[str, float] = Field(default_factory=dict)
    route_factor_keys: Dict[str, str] = Field(default_factory=dict)
    route_factors: Dict[str, EndOfLifeRouteFactorInput] = Field(default_factory=dict)
    source_ref: Optional[str] = None
    source_ref_ids: List[str] = Field(default_factory=list)


class CarbonCalculationRequest(BaseModel):
    product_id: str = Field(..., description="Canonical normalized product id.")
    requested_stages: List[CarbonStageName] = Field(default_factory=list)
    use_bootstrap_estimates: bool = False
    total_product_mass_kg: Optional[float] = None
    raw_materials: List[RawMaterialInput] = Field(default_factory=list)
    raw_material_factor_keys: Dict[str, str] = Field(default_factory=dict)
    raw_material_factor_values: Dict[str, float] = Field(default_factory=dict)
    transport_legs: List[TransportLegInput] = Field(default_factory=list)
    use_phase: Optional[UsePhaseInput] = None
    end_of_life: Optional[EndOfLifeInput] = None
    report_year: Optional[int] = None
    use_country_code: Optional[str] = None
    include_trace: bool = True
    include_ontology_sidecar: bool = False
    include_ontology_turtle: bool = False

    @validator("requested_stages", pre=True, each_item=True)
    def normalize_requested_stage(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _normalize_stage_name(value)
        return value

    @validator("requested_stages")
    def deduplicate_requested_stages(cls, value: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in value:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def to_service_scenario(self) -> Dict[str, Any]:
        payload = _model_dump(self)
        payload.pop("product_id", None)
        payload.pop("include_trace", None)
        payload.pop("include_ontology_sidecar", None)
        payload.pop("include_ontology_turtle", None)
        return payload


class CarbonTrace(BaseModel):
    item_id: str
    label: str
    stage: CarbonStageName
    activity_value: Optional[float] = None
    activity_unit: str
    factor_value: Optional[float] = None
    factor_unit: str
    emissions_kg_co2e: Optional[float] = None
    formula: str
    status: str
    source_refs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @classmethod
    def from_service_trace(cls, trace: CarbonTraceItem) -> "CarbonTrace":
        return cls(
            item_id=trace.item_id,
            label=trace.label,
            stage=trace.stage,
            activity_value=trace.activity_value,
            activity_unit=trace.activity_unit,
            factor_value=trace.factor_value,
            factor_unit=trace.factor_unit,
            emissions_kg_co2e=trace.emissions_kg_co2e,
            formula=trace.formula,
            status=trace.status,
            source_refs=list(trace.source_refs),
            notes=list(trace.notes),
        )


class CarbonStageResultModel(BaseModel):
    stage: CarbonStageName
    total_kg_co2e: float
    status: str
    traces: List[CarbonTrace] = Field(default_factory=list)
    missing_inputs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    quality_status: str = "unknown"
    uncertainty_pct: Optional[float] = None
    estimated_inputs: List[str] = Field(default_factory=list)

    @classmethod
    def from_service_stage(
        cls,
        stage_result: CarbonStageResult,
        include_trace: bool = True,
    ) -> "CarbonStageResultModel":
        return cls(
            stage=stage_result.stage,
            total_kg_co2e=stage_result.total_kg_co2e,
            status=stage_result.status,
            traces=[CarbonTrace.from_service_trace(trace) for trace in stage_result.traces] if include_trace else [],
            missing_inputs=list(stage_result.missing_inputs),
            notes=list(stage_result.notes),
            quality_status=stage_result.quality_status,
            uncertainty_pct=stage_result.uncertainty_pct,
            estimated_inputs=list(stage_result.estimated_inputs),
        )


class RecyclabilityResultModel(BaseModel):
    recyclability_pct: Optional[float] = None
    recoverable_mass_kg: Optional[float] = None
    incineration_mass_kg: Optional[float] = None
    landfill_mass_kg: Optional[float] = None
    status: str
    notes: List[str] = Field(default_factory=list)

    @classmethod
    def from_service_recyclability(cls, result: RecyclabilityResult) -> "RecyclabilityResultModel":
        return cls(
            recyclability_pct=result.recyclability_pct,
            recoverable_mass_kg=result.recoverable_mass_kg,
            incineration_mass_kg=result.incineration_mass_kg,
            landfill_mass_kg=result.landfill_mass_kg,
            status=result.status,
            notes=list(result.notes),
        )


class CarbonProvenanceModel(BaseModel):
    field_name: str
    label: str
    value: Optional[Any] = None
    unit: str
    status: str
    method: str
    source_refs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    uncertainty_pct: Optional[float] = None

    @classmethod
    def from_service_provenance(cls, item: CarbonProvenanceItem) -> "CarbonProvenanceModel":
        return cls(
            field_name=item.field_name,
            label=item.label,
            value=item.value,
            unit=item.unit,
            status=item.status,
            method=item.method,
            source_refs=list(item.source_refs),
            notes=list(item.notes),
            uncertainty_pct=item.uncertainty_pct,
        )


class CarbonOntologyValidationIssueModel(BaseModel):
    code: str
    severity: str
    message: str
    node_id: Optional[str] = None
    stage: Optional[str] = None

    @classmethod
    def from_service_issue(cls, issue: CarbonOntologyValidationIssue) -> "CarbonOntologyValidationIssueModel":
        return cls(
            code=issue.code,
            severity=issue.severity,
            message=issue.message,
            node_id=issue.node_id,
            stage=issue.stage,
        )


class CarbonOntologyValidationReportModel(BaseModel):
    status: str
    issues: List[CarbonOntologyValidationIssueModel] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    @classmethod
    def from_service_report(
        cls,
        report: CarbonOntologyValidationReport,
    ) -> "CarbonOntologyValidationReportModel":
        return cls(
            status=report.status,
            issues=[CarbonOntologyValidationIssueModel.from_service_issue(issue) for issue in report.issues],
            error_count=report.error_count,
            warning_count=report.warning_count,
        )


class CarbonOntologySidecarModel(BaseModel):
    ontology_path: str
    triple_count: int
    validation: CarbonOntologyValidationReportModel
    graph_turtle: Optional[str] = None

    @classmethod
    def from_service_sidecar(
        cls,
        sidecar: CarbonOntologySidecarResult,
    ) -> "CarbonOntologySidecarModel":
        return cls(
            ontology_path=sidecar.ontology_path,
            triple_count=sidecar.triple_count,
            validation=CarbonOntologyValidationReportModel.from_service_report(sidecar.validation),
            graph_turtle=sidecar.graph_turtle,
        )


class CarbonCalculationResponse(BaseModel):
    product_id: str
    product_name: str
    requested_stages: List[CarbonStageName]
    status: str
    total_kg_co2e: Optional[float] = None
    partial_total_kg_co2e: float
    stage_results: Dict[str, CarbonStageResultModel]
    recyclability: RecyclabilityResultModel
    missing_inputs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    used_bootstrap_estimates: bool = False
    quality_status: str = "unknown"
    estimated_fields: List[str] = Field(default_factory=list)
    provenance: List[CarbonProvenanceModel] = Field(default_factory=list)
    uncertainty_pct: Optional[float] = None
    uncertainty_kg_co2e: Optional[float] = None
    uncertainty_range_kg_co2e: Optional[Dict[str, float]] = None
    ontology_sidecar: Optional[CarbonOntologySidecarModel] = None

    @classmethod
    def from_service_result(
        cls,
        result: CarbonCalculationResult,
        include_trace: bool = True,
        ontology_sidecar: Optional[CarbonOntologySidecarResult] = None,
    ) -> "CarbonCalculationResponse":
        return cls(
            product_id=result.product_id,
            product_name=result.product_name,
            requested_stages=list(result.requested_stages),
            status=result.status,
            total_kg_co2e=result.total_kg_co2e,
            partial_total_kg_co2e=result.partial_total_kg_co2e,
            stage_results={
                stage_name: CarbonStageResultModel.from_service_stage(stage_result, include_trace=include_trace)
                for stage_name, stage_result in result.stage_results.items()
            },
            recyclability=RecyclabilityResultModel.from_service_recyclability(result.recyclability),
            missing_inputs=list(result.missing_inputs),
            warnings=list(result.warnings),
            assumptions=list(result.assumptions),
            source_refs=list(result.source_refs),
            used_bootstrap_estimates=result.used_bootstrap_estimates,
            quality_status=result.quality_status,
            estimated_fields=list(result.estimated_fields),
            provenance=[CarbonProvenanceModel.from_service_provenance(item) for item in result.provenance],
            uncertainty_pct=result.uncertainty_pct,
            uncertainty_kg_co2e=result.uncertainty_kg_co2e,
            uncertainty_range_kg_co2e=dict(result.uncertainty_range_kg_co2e) if result.uncertainty_range_kg_co2e else None,
            ontology_sidecar=CarbonOntologySidecarModel.from_service_sidecar(ontology_sidecar)
            if ontology_sidecar is not None
            else None,
        )


CarbonRequest = CarbonCalculationRequest
CarbonStageResultResponse = CarbonStageResultModel
CarbonResult = CarbonCalculationResponse
RecyclabilityResultResponse = RecyclabilityResultModel
CarbonOntologySidecarResponse = CarbonOntologySidecarModel
