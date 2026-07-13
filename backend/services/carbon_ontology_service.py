from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL, XSD

try:
    from owlrl import DeductiveClosure, OWLRL_Semantics  # type: ignore

    OWL_AVAILABLE = True
except Exception:
    OWL_AVAILABLE = False

from backend.services.carbon_calculation_service import CarbonCalculationResult, DEFAULT_DATA_ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "backend" / "ontologies" / "carbon_ontology.ttl"
CARB = Namespace("http://example.com/carbon#")
INST = Namespace("http://example.com/carbon/instance/")

STAGE_NODES = {
    "raw_materials": CARB.RawMaterials,
    "transportation": CARB.Transportation,
    "use_phase": CARB.UsePhase,
    "end_of_life": CARB.EndOfLife,
}

UNIT_NODES = {
    "kg": CARB.kg,
    "kWh": CARB.kWh,
    "ton-km": CARB.tonkm,
    "%": CARB.percent,
    "kg CO2e": CARB.kgCO2e,
    "kg CO2e/kg": CARB.kgCO2ePerkg,
    "kg CO2e/kWh": CARB.kgCO2ePerkWh,
    "kg CO2e/ton-km": CARB.kgCO2ePertonkm,
}

STATUS_NODES = {
    "computed": CARB.Known,
    "resolved": CARB.Known,
    "known": CARB.Known,
    "estimated": CARB.Estimated,
    "missing": CARB.Missing,
    "complete": CARB.Complete,
    "partial": CARB.Partial,
    "override": CARB.Override,
}

TRANSPORT_MODE_KEYWORDS = {
    "truck": "truck",
    "road": "truck",
    "ship": "ship",
    "sea": "ship",
    "rail": "rail",
    "train": "rail",
    "air": "air",
    "flight": "air",
}

END_OF_LIFE_KEYWORDS = {
    "recycling": "recycling",
    "incineration": "incineration",
    "landfill": "landfill",
}


@dataclass
class CarbonOntologyValidationIssue:
    code: str
    severity: str
    message: str
    node_id: Optional[str] = None
    stage: Optional[str] = None


@dataclass
class CarbonOntologyValidationReport:
    status: str
    issues: List[CarbonOntologyValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0


@dataclass
class CarbonOntologySidecarResult:
    ontology_path: str
    triple_count: int
    validation: CarbonOntologyValidationReport
    graph_turtle: Optional[str] = None


@dataclass
class _GraphArtifacts:
    product: URIRef
    scenario: URIRef
    functional_unit: URIRef
    total_result: Optional[URIRef] = None
    recyclability_result: Optional[URIRef] = None
    stage_results: Dict[str, URIRef] = field(default_factory=dict)
    trace_results: Dict[Tuple[str, str], URIRef] = field(default_factory=dict)
    activities: Dict[Tuple[str, str], URIRef] = field(default_factory=dict)
    factors: Dict[Tuple[str, str], URIRef] = field(default_factory=dict)
    calculations: Dict[Tuple[str, str], URIRef] = field(default_factory=dict)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value or "").strip("_").lower()
    return slug or "item"


def _literal_decimal(value: Optional[float]) -> Optional[Literal]:
    if value is None:
        return None
    return Literal(f"{float(value):.12g}", datatype=XSD.decimal)


def _literal_int(value: Optional[int]) -> Optional[Literal]:
    if value is None:
        return None
    return Literal(int(value), datatype=XSD.integer)


def _status_node(name: Optional[str]) -> URIRef:
    return STATUS_NODES.get((name or "").strip().lower(), CARB.Known)


def _unit_node(unit: Optional[str]) -> Optional[URIRef]:
    if unit is None:
        return None
    return UNIT_NODES.get(unit)


def _append_issue(
    issues: List[CarbonOntologyValidationIssue],
    code: str,
    severity: str,
    message: str,
    node_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    issues.append(
        CarbonOntologyValidationIssue(
            code=code,
            severity=severity,
            message=message,
            node_id=node_id,
            stage=stage,
        )
    )


class CarbonOntologyService:
    def __init__(self, ontology_path: Path | str = DEFAULT_ONTOLOGY_PATH, run_owl_rl: bool = True):
        self.ontology_path = Path(ontology_path)
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {self.ontology_path}")
        self.run_owl_rl = run_owl_rl

        graph = Graph()
        graph.parse(self.ontology_path, format="turtle")
        graph.bind("carb", CARB)
        graph.bind("inst", INST)
        graph.bind("rdfs", RDFS)
        graph.bind("owl", OWL)

        if self.run_owl_rl and OWL_AVAILABLE:
            DeductiveClosure(OWLRL_Semantics).expand(graph)

        self._base_graph = graph

    def build_sidecar(
        self,
        result: CarbonCalculationResult,
        include_turtle: bool = False,
    ) -> CarbonOntologySidecarResult:
        product_profile = self._load_product_profile(result.product_id)
        graph = self._fresh_graph()
        source_lookup = self._build_source_lookup(product_profile)
        artifacts = self._populate_graph(graph, product_profile, result, source_lookup)
        validation = self._validate_graph(graph, result, artifacts)
        graph_turtle = graph.serialize(format="turtle") if include_turtle else None
        return CarbonOntologySidecarResult(
            ontology_path=str(self.ontology_path),
            triple_count=len(graph),
            validation=validation,
            graph_turtle=graph_turtle,
        )

    def _fresh_graph(self) -> Graph:
        graph = Graph()
        for triple in self._base_graph:
            graph.add(triple)
        graph.bind("carb", CARB)
        graph.bind("inst", INST)
        graph.bind("rdfs", RDFS)
        graph.bind("owl", OWL)
        return graph

    def _load_product_profile(self, product_id: str) -> Dict[str, Any]:
        path = Path(DEFAULT_DATA_ROOT) / "products" / f"{product_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Product profile not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_source_lookup(self, product_profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for source in product_profile.get("source_refs", []) or []:
            source_id = source.get("id")
            source_path = source.get("path")
            if source_id:
                lookup[source_id] = source
            if source_path:
                lookup[source_path] = source
        return lookup

    def _ensure_source_node(
        self,
        graph: Graph,
        cache: Dict[str, URIRef],
        lookup: Dict[str, Dict[str, Any]],
        source_ref: str,
    ) -> URIRef:
        key = source_ref or "source"
        if key in cache:
            return cache[key]

        source_data = lookup.get(source_ref, {})
        node = INST[f"source/{_slug(source_ref)}"]
        graph.add((node, RDF.type, CARB.Source))
        graph.add((node, CARB.sourceReference, Literal(source_data.get("path") or source_ref)))

        source_name = source_data.get("id") or source_data.get("kind") or source_ref
        graph.add((node, CARB.sourceName, Literal(source_name)))
        if source_data.get("kind"):
            graph.add((node, CARB.sourceKind, Literal(source_data["kind"])))
        if source_data.get("extract_status"):
            graph.add((node, CARB.noteText, Literal(f"extract_status={source_data['extract_status']}")))

        cache[key] = node
        return node

    def _populate_graph(
        self,
        graph: Graph,
        product_profile: Dict[str, Any],
        result: CarbonCalculationResult,
        source_lookup: Dict[str, Dict[str, Any]],
    ) -> _GraphArtifacts:
        product_node = INST[f"product/{_slug(result.product_id)}"]
        scenario_node = INST[f"scenario/{_slug(result.product_id)}_{_slug(result.status)}"]
        functional_unit_node = INST[f"functional_unit/{_slug(result.product_id)}"]
        factor_set_node = INST[f"factor_set/{_slug(result.product_id)}"]

        artifacts = _GraphArtifacts(
            product=product_node,
            scenario=scenario_node,
            functional_unit=functional_unit_node,
        )
        source_nodes: Dict[str, URIRef] = {}

        graph.add((product_node, RDF.type, CARB.Product))
        graph.add((product_node, CARB.productId, Literal(result.product_id)))
        graph.add((product_node, CARB.productName, Literal(result.product_name)))
        graph.add((product_node, CARB.hasScenario, scenario_node))

        graph.add((scenario_node, RDF.type, CARB.Scenario))
        graph.add((scenario_node, CARB.scenarioName, Literal(f"{result.product_name} carbon calculation scenario")))
        graph.add((scenario_node, CARB.hasFunctionalUnit, functional_unit_node))
        graph.add((scenario_node, CARB.hasFactorSet, factor_set_node))
        graph.add((scenario_node, CARB.hasStatus, _status_node(result.status)))

        graph.add((functional_unit_node, RDF.type, CARB.FunctionalUnit))
        graph.add(
            (
                functional_unit_node,
                CARB.functionalUnitText,
                Literal(product_profile.get("calculation_scope", {}).get("default_basis") or "per product lifetime"),
            )
        )

        graph.add((factor_set_node, RDF.type, CARB.FactorSet))
        graph.add((factor_set_node, CARB.noteText, Literal("Normalized factor tables from backend/data/carbon/factors/.")))

        defaults = product_profile.get("defaults", {}) or {}
        report_year = defaults.get("report_year")
        if report_year is not None:
            graph.add((scenario_node, CARB.reportYear, _literal_int(int(report_year))))
        if defaults.get("use_country_code"):
            graph.add((scenario_node, CARB.countryCode, Literal(defaults["use_country_code"])))
        if defaults.get("include_paper") is not None:
            graph.add((scenario_node, CARB.includesPaper, Literal(bool(defaults["include_paper"]), datatype=XSD.boolean)))
        if defaults.get("annual_energy_kwh") is not None:
            graph.add((scenario_node, CARB.annualElectricityKWh, _literal_decimal(float(defaults["annual_energy_kwh"]))))
        if defaults.get("lifetime_years") is not None:
            graph.add((scenario_node, CARB.lifetimeYears, _literal_decimal(float(defaults["lifetime_years"]))))
        if defaults.get("lifetime_pages") is not None:
            graph.add((scenario_node, CARB.lifetimePages, _literal_decimal(float(defaults["lifetime_pages"]))))

        for assumption in result.assumptions:
            graph.add((scenario_node, CARB.assumptionText, Literal(assumption)))
        for missing in result.missing_inputs:
            graph.add((scenario_node, CARB.missingInputText, Literal(missing)))
        for warning in result.warnings:
            graph.add((scenario_node, CARB.noteText, Literal(warning)))

        for source in product_profile.get("source_refs", []) or []:
            if source.get("id"):
                self._ensure_source_node(graph, source_nodes, source_lookup, source["id"])
            elif source.get("path"):
                self._ensure_source_node(graph, source_nodes, source_lookup, source["path"])

        for stage_name in result.requested_stages:
            stage_node = STAGE_NODES.get(stage_name)
            if stage_node is not None:
                graph.add((scenario_node, CARB.hasStage, stage_node))

        for stage_name, stage_result in result.stage_results.items():
            stage_node = STAGE_NODES[stage_name]
            stage_result_node = INST[f"result/stage/{_slug(result.product_id)}_{stage_name}"]
            artifacts.stage_results[stage_name] = stage_result_node

            graph.add((stage_result_node, RDF.type, CARB.StageResult))
            graph.add((stage_result_node, CARB.belongsToStage, stage_node))
            graph.add((stage_result_node, CARB.resultType, Literal("StageResult")))
            graph.add((stage_result_node, CARB.resultUnit, CARB.kgCO2e))
            graph.add((stage_result_node, CARB.hasStatus, _status_node(stage_result.status)))
            graph.add((stage_result_node, CARB.resultValue, _literal_decimal(stage_result.total_kg_co2e)))
            graph.add((scenario_node, CARB.hasResult, stage_result_node))
            for note in stage_result.notes:
                graph.add((stage_result_node, CARB.noteText, Literal(note)))
            for missing in stage_result.missing_inputs:
                graph.add((stage_result_node, CARB.noteText, Literal(f"missing_input={missing}")))

            for trace in stage_result.traces:
                trace_key = (stage_name, trace.item_id)
                activity_node = INST[f"activity/{_slug(result.product_id)}_{stage_name}_{_slug(trace.item_id)}"]
                factor_node = INST[f"factor/{_slug(result.product_id)}_{stage_name}_{_slug(trace.item_id)}"]
                calc_node = INST[f"calc/{_slug(result.product_id)}_{stage_name}_{_slug(trace.item_id)}"]
                trace_result_node = INST[f"result/trace/{_slug(result.product_id)}_{stage_name}_{_slug(trace.item_id)}"]

                artifacts.activities[trace_key] = activity_node
                artifacts.factors[trace_key] = factor_node
                artifacts.calculations[trace_key] = calc_node
                artifacts.trace_results[trace_key] = trace_result_node

                graph.add((activity_node, RDF.type, CARB.ActivityData))
                graph.add((activity_node, CARB.flowName, Literal(trace.label)))
                graph.add((activity_node, CARB.hasStatus, _status_node(trace.status)))
                if trace.activity_value is not None:
                    graph.add((activity_node, CARB.activityValue, _literal_decimal(trace.activity_value)))
                unit_node = _unit_node(trace.activity_unit)
                if unit_node is not None:
                    graph.add((activity_node, CARB.hasUnit, unit_node))
                if stage_name == "raw_materials" and trace.activity_value is not None:
                    graph.add((activity_node, CARB.materialMassKg, _literal_decimal(trace.activity_value)))
                    material_node = INST[f"material_category/{_slug(trace.item_id)}"]
                    graph.add((material_node, RDF.type, CARB.MaterialCategory))
                    graph.add((material_node, RDFS.label, Literal(trace.item_id)))
                    graph.add((activity_node, CARB.hasMaterialCategory, material_node))
                if stage_name == "transportation":
                    mode = self._infer_transport_mode(trace.label, trace.item_id)
                    if mode:
                        mode_node = INST[f"transport_mode/{_slug(mode)}"]
                        graph.add((mode_node, RDF.type, CARB.TransportMode))
                        graph.add((mode_node, RDFS.label, Literal(mode)))
                        graph.add((activity_node, CARB.hasTransportMode, mode_node))
                if stage_name == "end_of_life":
                    route = self._infer_end_of_life_route(trace.label, trace.item_id)
                    if route:
                        route_node = INST[f"end_of_life_route/{_slug(route)}"]
                        graph.add((route_node, RDF.type, CARB.EndOfLifeRoute))
                        graph.add((route_node, RDFS.label, Literal(route)))
                        graph.add((activity_node, CARB.hasEndOfLifeRoute, route_node))

                graph.add((factor_node, RDF.type, CARB.EmissionFactor))
                graph.add((factor_node, CARB.hasStatus, _status_node(trace.status)))
                if trace.factor_value is not None:
                    graph.add((factor_node, CARB.factorValue, _literal_decimal(trace.factor_value)))
                factor_unit_node = _unit_node(trace.factor_unit)
                if factor_unit_node is not None:
                    graph.add((factor_node, CARB.factorUnit, factor_unit_node))
                graph.add((factor_node, CARB.factorKey, Literal(trace.item_id)))

                graph.add((trace_result_node, RDF.type, CARB.TraceResult))
                graph.add((trace_result_node, CARB.resultType, Literal("TraceResult")))
                graph.add((trace_result_node, CARB.belongsToStage, stage_node))
                graph.add((trace_result_node, CARB.resultUnit, CARB.kgCO2e))
                graph.add((trace_result_node, CARB.hasStatus, _status_node(trace.status)))
                if trace.emissions_kg_co2e is not None:
                    graph.add((trace_result_node, CARB.resultValue, _literal_decimal(trace.emissions_kg_co2e)))

                graph.add((calc_node, RDF.type, CARB.CalculationStep))
                graph.add((calc_node, CARB.inputActivity, activity_node))
                graph.add((calc_node, CARB.inputFactor, factor_node))
                graph.add((calc_node, CARB.outputResult, trace_result_node))
                graph.add((calc_node, CARB.formulaText, Literal(trace.formula)))
                graph.add((scenario_node, CARB.hasCalculationStep, calc_node))
                graph.add((scenario_node, CARB.hasResult, trace_result_node))
                graph.add((stage_node, CARB.hasActivity, activity_node))
                graph.add((activity_node, CARB.usesEmissionFactor, factor_node))

                for note in trace.notes:
                    graph.add((activity_node, CARB.noteText, Literal(note)))

                for source_ref in trace.source_refs:
                    source_node = self._ensure_source_node(graph, source_nodes, source_lookup, source_ref)
                    graph.add((activity_node, CARB.derivedFrom, source_node))
                    graph.add((factor_node, CARB.factorSource, source_node))

        if result.total_kg_co2e is not None:
            total_node = INST[f"result/total/{_slug(result.product_id)}"]
            artifacts.total_result = total_node
            graph.add((total_node, RDF.type, CARB.TotalResult))
            graph.add((total_node, CARB.resultType, Literal("TotalResult")))
            graph.add((total_node, CARB.resultUnit, CARB.kgCO2e))
            graph.add((total_node, CARB.hasStatus, _status_node(result.status)))
            graph.add((total_node, CARB.resultValue, _literal_decimal(result.total_kg_co2e)))
            graph.add((scenario_node, CARB.hasResult, total_node))
        else:
            partial_node = INST[f"result/partial_total/{_slug(result.product_id)}"]
            artifacts.total_result = partial_node
            graph.add((partial_node, RDF.type, CARB.TotalResult))
            graph.add((partial_node, CARB.resultType, Literal("PartialTotalResult")))
            graph.add((partial_node, CARB.resultUnit, CARB.kgCO2e))
            graph.add((partial_node, CARB.hasStatus, CARB.Partial))
            graph.add((partial_node, CARB.resultValue, _literal_decimal(result.partial_total_kg_co2e)))
            graph.add((scenario_node, CARB.hasResult, partial_node))

        recyclability = result.recyclability
        recyclability_node = INST[f"result/recyclability/{_slug(result.product_id)}"]
        artifacts.recyclability_result = recyclability_node
        graph.add((recyclability_node, RDF.type, CARB.RecyclabilityResult))
        graph.add((recyclability_node, CARB.resultType, Literal("RecyclabilityResult")))
        graph.add((recyclability_node, CARB.resultUnit, CARB.percent))
        graph.add((recyclability_node, CARB.hasStatus, _status_node(recyclability.status)))
        graph.add((scenario_node, CARB.hasResult, recyclability_node))
        if recyclability.recyclability_pct is not None:
            graph.add((recyclability_node, CARB.recyclabilityRatePct, _literal_decimal(recyclability.recyclability_pct)))
            graph.add((recyclability_node, CARB.resultValue, _literal_decimal(recyclability.recyclability_pct)))
        if recyclability.recoverable_mass_kg is not None:
            graph.add((recyclability_node, CARB.recoverableMassKg, _literal_decimal(recyclability.recoverable_mass_kg)))
        if recyclability.incineration_mass_kg is not None:
            graph.add((recyclability_node, CARB.incinerationMassKg, _literal_decimal(recyclability.incineration_mass_kg)))
        if recyclability.landfill_mass_kg is not None:
            graph.add((recyclability_node, CARB.landfillMassKg, _literal_decimal(recyclability.landfill_mass_kg)))
        for note in recyclability.notes:
            graph.add((recyclability_node, CARB.noteText, Literal(note)))

        return artifacts

    def _validate_graph(
        self,
        graph: Graph,
        result: CarbonCalculationResult,
        artifacts: _GraphArtifacts,
    ) -> CarbonOntologyValidationReport:
        issues: List[CarbonOntologyValidationIssue] = []

        if (artifacts.product, CARB.hasScenario, artifacts.scenario) not in graph:
            _append_issue(issues, "missing_product_scenario", "error", "Product is missing hasScenario link.")
        if (artifacts.scenario, CARB.hasFunctionalUnit, artifacts.functional_unit) not in graph:
            _append_issue(issues, "missing_functional_unit", "error", "Scenario is missing hasFunctionalUnit link.")

        for stage_name in result.requested_stages:
            stage_node = STAGE_NODES.get(stage_name)
            if stage_node is None or (artifacts.scenario, CARB.hasStage, stage_node) not in graph:
                _append_issue(
                    issues,
                    "missing_stage_link",
                    "error",
                    f"Scenario is missing requested stage {stage_name}.",
                    stage=stage_name,
                )
            stage_result = result.stage_results.get(stage_name)
            stage_result_node = artifacts.stage_results.get(stage_name)
            if stage_result_node is None:
                _append_issue(
                    issues,
                    "missing_stage_result",
                    "error",
                    f"Stage result node for {stage_name} was not created.",
                    stage=stage_name,
                )
                continue
            if (stage_result_node, CARB.belongsToStage, stage_node) not in graph:
                _append_issue(
                    issues,
                    "stage_result_missing_stage",
                    "error",
                    f"Stage result for {stage_name} is missing belongsToStage.",
                    node_id=str(stage_result_node),
                    stage=stage_name,
                )
            if not list(graph.objects(stage_result_node, CARB.resultUnit)):
                _append_issue(
                    issues,
                    "stage_result_missing_unit",
                    "error",
                    f"Stage result for {stage_name} is missing a result unit.",
                    node_id=str(stage_result_node),
                    stage=stage_name,
                )
            if stage_result is not None and stage_result.status == "complete" and not stage_result.traces:
                _append_issue(
                    issues,
                    "complete_stage_without_traces",
                    "warning",
                    f"Stage {stage_name} is complete but has no trace items.",
                    node_id=str(stage_result_node),
                    stage=stage_name,
                )

            if stage_result is None:
                continue
            for trace in stage_result.traces:
                key = (stage_name, trace.item_id)
                activity_node = artifacts.activities.get(key)
                factor_node = artifacts.factors.get(key)
                calc_node = artifacts.calculations.get(key)
                trace_result_node = artifacts.trace_results.get(key)

                if trace.status == "computed":
                    if trace.activity_value is None:
                        _append_issue(
                            issues,
                            "computed_trace_missing_activity_value",
                            "error",
                            f"Computed trace {trace.item_id} is missing activity value.",
                            stage=stage_name,
                        )
                    if trace.factor_value is None:
                        _append_issue(
                            issues,
                            "computed_trace_missing_factor_value",
                            "error",
                            f"Computed trace {trace.item_id} is missing factor value.",
                            stage=stage_name,
                        )
                    if trace.emissions_kg_co2e is None:
                        _append_issue(
                            issues,
                            "computed_trace_missing_result_value",
                            "error",
                            f"Computed trace {trace.item_id} is missing emissions value.",
                            stage=stage_name,
                        )
                    if not all([activity_node, factor_node, calc_node, trace_result_node]):
                        _append_issue(
                            issues,
                            "computed_trace_missing_nodes",
                            "error",
                            f"Computed trace {trace.item_id} is missing one or more ontology nodes.",
                            stage=stage_name,
                        )
                if not trace.source_refs:
                    _append_issue(
                        issues,
                        "trace_missing_provenance",
                        "warning",
                        f"Trace {trace.item_id} does not record any provenance source refs.",
                        stage=stage_name,
                    )

        if result.total_kg_co2e is not None:
            if artifacts.total_result is None or not list(graph.objects(artifacts.total_result, CARB.resultValue)):
                _append_issue(
                    issues,
                    "missing_total_result",
                    "error",
                    "Complete calculation is missing total result value.",
                    node_id=str(artifacts.total_result) if artifacts.total_result else None,
                )

        if result.recyclability.recyclability_pct is not None:
            if artifacts.recyclability_result is None or not list(
                graph.objects(artifacts.recyclability_result, CARB.recyclabilityRatePct)
            ):
                _append_issue(
                    issues,
                    "missing_recyclability_result",
                    "error",
                    "Recyclability value is missing from the ontology sidecar.",
                    node_id=str(artifacts.recyclability_result) if artifacts.recyclability_result else None,
                )

        if result.missing_inputs:
            _append_issue(
                issues,
                "calculation_has_missing_inputs",
                "warning",
                f"Calculation still has {len(result.missing_inputs)} missing inputs.",
            )
        if result.used_bootstrap_estimates:
            _append_issue(
                issues,
                "bootstrap_estimates_used",
                "warning",
                "Bootstrap estimates were used in this calculation scenario.",
            )

        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        if error_count:
            status = "invalid"
        elif warning_count:
            status = "warning"
        else:
            status = "valid"

        return CarbonOntologyValidationReport(
            status=status,
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
        )

    def _infer_transport_mode(self, label: str, item_id: str) -> Optional[str]:
        text = f"{label} {item_id}".lower()
        for keyword, mode in TRANSPORT_MODE_KEYWORDS.items():
            if keyword in text:
                return mode
        return None

    def _infer_end_of_life_route(self, label: str, item_id: str) -> Optional[str]:
        text = f"{label} {item_id}".lower()
        for keyword, route in END_OF_LIFE_KEYWORDS.items():
            if keyword in text:
                return route
        return None


_SERVICE_SINGLETON: Optional[CarbonOntologyService] = None


def get_carbon_ontology_service() -> CarbonOntologyService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = CarbonOntologyService()
    return _SERVICE_SINGLETON


def build_carbon_ontology_sidecar(
    result: CarbonCalculationResult,
    include_turtle: bool = False,
) -> CarbonOntologySidecarResult:
    return get_carbon_ontology_service().build_sidecar(result=result, include_turtle=include_turtle)
