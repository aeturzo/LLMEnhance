from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "carbon"
DEFAULT_STAGES = ["raw_materials", "transportation", "use_phase", "end_of_life"]


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _unique(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _stage_alias(name: str) -> str:
    mapping = {
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
    return mapping.get((name or "").strip().lower(), name)


def _transport_mode_alias(name: Optional[str]) -> Optional[str]:
    mapping = {
        "truck": "truck",
        "road": "truck",
        "road_freight": "truck",
        "lorry": "truck",
        "ship": "ship",
        "sea": "ship",
        "sea_freight": "ship",
        "ocean": "ship",
        "ocean_freight": "ship",
        "rail": "rail",
        "train": "rail",
        "air": "air",
        "air_freight": "air",
    }
    key = (name or "").strip().lower()
    return mapping.get(key) if key else None


def _collect_source_refs(*values: Any) -> List[str]:
    refs: List[str] = []
    for value in values:
        if isinstance(value, str) and value:
            refs.append(value)
        elif isinstance(value, list):
            refs.extend(item for item in value if isinstance(item, str) and item)
    return _unique(refs)


@dataclass
class FactorResolution:
    factor_key: Optional[str]
    factor_value: Optional[float]
    factor_unit: str
    source_ref: Optional[str]
    status: str
    notes: List[str] = field(default_factory=list)


@dataclass
class CarbonTraceItem:
    item_id: str
    label: str
    stage: str
    activity_value: Optional[float]
    activity_unit: str
    factor_value: Optional[float]
    factor_unit: str
    emissions_kg_co2e: Optional[float]
    formula: str
    status: str
    source_refs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CarbonProvenanceItem:
    field_name: str
    label: str
    value: Optional[Any]
    unit: str
    status: str
    method: str
    source_refs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    uncertainty_pct: Optional[float] = None


@dataclass
class CarbonStageResult:
    stage: str
    total_kg_co2e: float
    status: str
    traces: List[CarbonTraceItem] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    quality_status: str = "unknown"
    uncertainty_pct: Optional[float] = None
    estimated_inputs: List[str] = field(default_factory=list)


@dataclass
class RecyclabilityResult:
    recyclability_pct: Optional[float]
    recoverable_mass_kg: Optional[float]
    incineration_mass_kg: Optional[float]
    landfill_mass_kg: Optional[float]
    status: str
    notes: List[str] = field(default_factory=list)


@dataclass
class CarbonCalculationResult:
    product_id: str
    product_name: str
    requested_stages: List[str]
    status: str
    total_kg_co2e: Optional[float]
    partial_total_kg_co2e: float
    stage_results: Dict[str, CarbonStageResult]
    recyclability: RecyclabilityResult
    missing_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    used_bootstrap_estimates: bool = False
    quality_status: str = "unknown"
    estimated_fields: List[str] = field(default_factory=list)
    provenance: List[CarbonProvenanceItem] = field(default_factory=list)
    uncertainty_pct: Optional[float] = None
    uncertainty_kg_co2e: Optional[float] = None
    uncertainty_range_kg_co2e: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CarbonCalculationService:
    """
    Deterministic carbon-footprint calculator over the normalized carbon data
    layer. It prefers official normalized inputs but can also use explicit
    scenario overrides while those official inputs are still incomplete.
    """

    def __init__(self, data_root: Path | str = DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root)
        self.products_dir = self.data_root / "products"
        self.factors_dir = self.data_root / "factors"

        self._electricity_by_key, self._electricity_by_country = self._load_electricity_factors()
        self._raw_material_factors = self._load_generic_factor_table(
            self.factors_dir / "raw_material_factors.csv",
            value_column="value_kg_co2e_per_kg",
        )
        self._transport_factors = self._load_generic_factor_table(
            self.factors_dir / "transport_factors.csv",
            value_column="value_kg_co2e_per_ton_km",
        )
        self._end_of_life_factors = self._load_generic_factor_table(
            self.factors_dir / "end_of_life_factors.csv",
            value_column="value_kg_co2e_per_kg",
        )

    def product_path(self, product_id: str) -> Path:
        return self.products_dir / f"{product_id}.json"

    def load_product_profile(self, product_id: str) -> Dict[str, Any]:
        path = self.product_path(product_id)
        if not path.exists():
            raise FileNotFoundError(f"Product profile not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def calculate(self, product_id: str, scenario: Optional[Dict[str, Any]] = None) -> CarbonCalculationResult:
        scenario = scenario or {}
        product = self.load_product_profile(product_id)
        requested_stages = [_stage_alias(stage) for stage in (scenario.get("requested_stages") or DEFAULT_STAGES)]
        requested_stages = [stage for stage in requested_stages if stage in DEFAULT_STAGES]
        if not requested_stages:
            requested_stages = DEFAULT_STAGES.copy()

        warnings: List[str] = []
        assumptions: List[str] = []
        source_refs = [ref.get("path") for ref in product.get("source_refs", []) if ref.get("path")]
        used_bootstrap = bool(scenario.get("use_bootstrap_estimates"))

        total_mass_kg = self._resolve_total_mass_kg(product, scenario, used_bootstrap, assumptions)

        stage_results: Dict[str, CarbonStageResult] = {}
        if "raw_materials" in requested_stages:
            stage_results["raw_materials"] = self._calculate_raw_materials_stage(
                product=product,
                scenario=scenario,
                total_mass_kg=total_mass_kg,
                use_bootstrap=used_bootstrap,
                assumptions=assumptions,
            )
        if "transportation" in requested_stages:
            stage_results["transportation"] = self._calculate_transport_stage(
                product=product,
                scenario=scenario,
                total_mass_kg=total_mass_kg,
                use_bootstrap=used_bootstrap,
                assumptions=assumptions,
            )
        if "use_phase" in requested_stages:
            stage_results["use_phase"] = self._calculate_use_phase_stage(
                product=product,
                scenario=scenario,
                use_bootstrap=used_bootstrap,
                assumptions=assumptions,
                warnings=warnings,
            )
        if "end_of_life" in requested_stages:
            stage_results["end_of_life"] = self._calculate_end_of_life_stage(
                product=product,
                scenario=scenario,
                total_mass_kg=total_mass_kg,
                use_bootstrap=used_bootstrap,
                assumptions=assumptions,
                warnings=warnings,
            )

        stage_source_refs: List[str] = []
        missing_inputs: List[str] = []
        partial_total = 0.0
        complete = True
        any_trace = False
        for result in stage_results.values():
            partial_total += result.total_kg_co2e
            missing_inputs.extend(result.missing_inputs)
            any_trace = any_trace or bool(result.traces)
            for trace in result.traces:
                stage_source_refs.extend(trace.source_refs)
            if result.status != "complete":
                complete = False

        if complete and stage_results:
            status = "complete"
            total = partial_total
        elif any_trace:
            status = "partial"
            total = None
        else:
            status = "missing"
            total = None

        recyclability = self._calculate_recyclability(
            product=product,
            scenario=scenario,
            total_mass_kg=total_mass_kg,
            use_bootstrap=used_bootstrap,
        )

        quality_status, estimated_fields, uncertainty_pct, uncertainty_kg, uncertainty_range = self._summarize_quality(
            product=product,
            stage_results=stage_results,
            total_kg_co2e=total,
            partial_total_kg_co2e=partial_total,
        )
        provenance = self._build_provenance(
            product=product,
            scenario=scenario,
            result=stage_results,
            total_mass_kg=total_mass_kg,
            used_bootstrap=used_bootstrap,
        )
        provenance_source_refs = _unique(
            ref
            for item in provenance
            for ref in item.source_refs
            if ref
        )

        return CarbonCalculationResult(
            product_id=product_id,
            product_name=product.get("display_name", product_id),
            requested_stages=requested_stages,
            status=status,
            total_kg_co2e=total,
            partial_total_kg_co2e=partial_total,
            stage_results=stage_results,
            recyclability=recyclability,
            missing_inputs=_unique(missing_inputs),
            warnings=_unique(warnings),
            assumptions=_unique(assumptions),
            source_refs=_unique(source_refs + self._resolve_source_refs(product, stage_source_refs) + provenance_source_refs),
            used_bootstrap_estimates=used_bootstrap,
            quality_status=quality_status,
            estimated_fields=estimated_fields,
            provenance=provenance,
            uncertainty_pct=uncertainty_pct,
            uncertainty_kg_co2e=uncertainty_kg,
            uncertainty_range_kg_co2e=uncertainty_range,
        )

    def _load_electricity_factors(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        path = self.factors_dir / "electricity_lc_factors.csv"
        by_key: Dict[str, Dict[str, Any]] = {}
        by_country: Dict[str, List[Dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["year"] = int(row["year"])
                row["value_kg_co2e_per_kwh"] = float(row["value_kg_co2e_per_kwh"])
                by_key[row["factor_key"]] = row
                country_code = (row["country_code"] or "").upper()
                by_country.setdefault(country_code, []).append(row)
        for values in by_country.values():
            values.sort(key=lambda item: item["year"])
        return by_key, by_country

    def _load_generic_factor_table(self, path: Path, value_column: str) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_numeric_value"] = _to_float(row.get(value_column))
                rows[row["factor_key"]] = row
        return rows

    @staticmethod
    def _source_ref_map(product: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        refs = product.get("source_refs", []) or []
        return {
            ref.get("id"): ref
            for ref in refs
            if isinstance(ref, dict) and ref.get("id")
        }

    def _resolve_source_refs(self, product: Dict[str, Any], refs: List[str]) -> List[str]:
        ref_map = self._source_ref_map(product)
        resolved: List[str] = []
        for ref in refs:
            if not ref:
                continue
            if ref in ref_map:
                resolved.append(ref_map[ref].get("path") or ref)
                continue
            resolved.append(ref)
        return _unique(resolved)

    @staticmethod
    def _observed_fact(product: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
        facts = product.get("observed_facts", {}) or {}
        value = facts.get(key)
        return value if isinstance(value, dict) else None

    @staticmethod
    def _fact_numeric_value(fact: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(fact, dict):
            return None
        return (
            _to_float(fact.get("value"))
            or _to_float(fact.get("preferred_value"))
        )

    @staticmethod
    def _merge_missing_values(target: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(target)
        for key, value in defaults.items():
            current = merged.get(key)
            if current in (None, "", [], {}):
                merged[key] = value
        return merged

    @staticmethod
    def _status_looks_estimated(status: Optional[str]) -> bool:
        text = (status or "").strip().lower()
        return text in {
            "estimated",
            "hybrid_estimate",
            "scenario_override",
            "exact_conflict_resolved",
        } or "estimate" in text

    def _stage_uncertainty_default(
        self,
        product: Dict[str, Any],
        stage_name: str,
        quality_status: str,
    ) -> Optional[float]:
        defaults = product.get("estimation_profile", {}).get("uncertainty_defaults_pct", {}) or {}
        if quality_status == "estimated":
            return _to_float(defaults.get(stage_name)) or 30.0
        if quality_status == "scenario_override":
            return 10.0
        if quality_status == "exact":
            return 8.0
        return _to_float(defaults.get(stage_name)) or _to_float(defaults.get("total"))

    def _resolve_total_mass_kg(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        use_bootstrap: bool,
        assumptions: List[str],
    ) -> Optional[float]:
        direct = _to_float(scenario.get("total_product_mass_kg"))
        if direct is not None:
            return direct

        observed_mass = self._fact_numeric_value(self._observed_fact(product, "product_mass_kg"))
        if observed_mass is not None:
            return observed_mass

        if not use_bootstrap:
            return None

        weight_range = product.get("bootstrap_estimates", {}).get("unboxed_weight_kg_range")
        if isinstance(weight_range, list) and len(weight_range) == 2:
            lo = _to_float(weight_range[0])
            hi = _to_float(weight_range[1])
            if lo is not None and hi is not None:
                midpoint = (lo + hi) / 2.0
                assumptions.append(
                    f"Used bootstrap midpoint total mass {midpoint:.3f} kg from estimated range {lo:.3f}-{hi:.3f} kg."
                )
                return midpoint
        return None

    def _resolve_generic_factor(
        self,
        table: Dict[str, Dict[str, Any]],
        factor_key: Optional[str],
        explicit_value: Optional[float],
        unit: str,
        default_key: Optional[str] = None,
        assumption_prefix: Optional[str] = None,
        explicit_source_ref: Optional[str] = None,
    ) -> FactorResolution:
        if explicit_value is not None:
            return FactorResolution(
                factor_key=factor_key or default_key,
                factor_value=explicit_value,
                factor_unit=unit,
                source_ref=explicit_source_ref or "scenario_override",
                status="override",
            )

        chosen_key = factor_key or default_key
        if not chosen_key:
            return FactorResolution(
                factor_key=None,
                factor_value=None,
                factor_unit=unit,
                source_ref=None,
                status="missing",
                notes=["No factor key or explicit factor value was provided."],
            )

        row = table.get(chosen_key)
        if row is None:
            return FactorResolution(
                factor_key=chosen_key,
                factor_value=None,
                factor_unit=unit,
                source_ref=None,
                status="missing",
                notes=[f"Factor key {chosen_key} was not found in the normalized table."],
            )

        numeric_value = row.get("_numeric_value")
        if numeric_value is None:
            return FactorResolution(
                factor_key=chosen_key,
                factor_value=None,
                factor_unit=unit,
                source_ref=row.get("source_ref"),
                status="missing",
                notes=[f"Factor key {chosen_key} exists but still requires manual curation."],
            )

        notes: List[str] = []
        if factor_key is None and default_key and assumption_prefix:
            notes.append(f"{assumption_prefix} default factor key {default_key}.")
        return FactorResolution(
            factor_key=chosen_key,
            factor_value=numeric_value,
            factor_unit=unit,
            source_ref=row.get("source_ref"),
            status="resolved",
            notes=notes,
        )

    def _resolve_electricity_factor(
        self,
        country_code: Optional[str],
        report_year: Optional[int],
        factor_key: Optional[str],
        explicit_value: Optional[float],
        explicit_source_ref: Optional[str] = None,
    ) -> FactorResolution:
        if explicit_value is not None:
            return FactorResolution(
                factor_key=factor_key,
                factor_value=explicit_value,
                factor_unit="kg CO2e/kWh",
                source_ref=explicit_source_ref or "scenario_override",
                status="override",
            )

        if factor_key:
            row = self._electricity_by_key.get(factor_key)
            if row is not None:
                return FactorResolution(
                    factor_key=factor_key,
                    factor_value=row["value_kg_co2e_per_kwh"],
                    factor_unit=row["unit"],
                    source_ref=row["source_ref"],
                    status="resolved",
                )

        if not country_code:
            return FactorResolution(
                factor_key=factor_key,
                factor_value=None,
                factor_unit="kg CO2e/kWh",
                source_ref=None,
                status="missing",
                notes=["No electricity country code or factor override was provided."],
            )

        rows = self._electricity_by_country.get(country_code.upper(), [])
        if not rows:
            return FactorResolution(
                factor_key=factor_key,
                factor_value=None,
                factor_unit="kg CO2e/kWh",
                source_ref=None,
                status="missing",
                notes=[f"No normalized electricity factor exists for country {country_code.upper()}."],
            )

        chosen = None
        notes: List[str] = []
        if report_year is not None:
            for row in rows:
                if row["year"] == report_year:
                    chosen = row
                    break
            if chosen is None:
                eligible = [row for row in rows if row["year"] <= report_year]
                if eligible:
                    chosen = eligible[-1]
                    notes.append(
                        f"Used latest available electricity factor for {country_code.upper()} not exceeding {report_year}: {chosen['year']}."
                    )
                else:
                    chosen = rows[-1]
                    notes.append(
                        f"No electricity factor for {country_code.upper()} at or before {report_year}; used latest available year {chosen['year']}."
                    )
        else:
            chosen = rows[-1]
            notes.append(f"No report year provided; used latest available electricity factor year {chosen['year']}.")

        return FactorResolution(
            factor_key=chosen["factor_key"],
            factor_value=chosen["value_kg_co2e_per_kwh"],
            factor_unit=chosen["unit"],
            source_ref=chosen["source_ref"],
            status="resolved",
            notes=notes,
        )

    def _bootstrap_materials(
        self,
        product: Dict[str, Any],
        total_mass_kg: Optional[float],
        assumptions: List[str],
    ) -> List[Dict[str, Any]]:
        if total_mass_kg is None:
            return []
        mix = product.get("bootstrap_estimates", {}).get("material_mix_pct") or {}
        if not mix:
            return []
        assumptions.append("Used bootstrap material mix percentages to derive raw-material masses.")
        return [
            {
                "material_key": material_key,
                "mass_kg": total_mass_kg * (float(pct) / 100.0),
                "share_mass_pct": float(pct),
                "status": "estimated",
                "source_ref_ids": [product.get("bootstrap_estimates", {}).get("based_on_source_ref_id", "bootstrap")],
            }
            for material_key, pct in mix.items()
        ]

    def _material_inputs(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        total_mass_kg: Optional[float],
        use_bootstrap: bool,
        assumptions: List[str],
    ) -> List[Dict[str, Any]]:
        scenario_entries = scenario.get("raw_materials")
        if isinstance(scenario_entries, list) and scenario_entries:
            return scenario_entries

        official_entries = product.get("life_cycle_inputs", {}).get("raw_materials", [])
        resolved_entries: List[Dict[str, Any]] = []
        for entry in official_entries:
            mass_kg = _to_float(entry.get("mass_kg"))
            share_mass_pct = _to_float(entry.get("share_mass_pct"))
            if mass_kg is None and total_mass_kg is not None and share_mass_pct is not None:
                mass_kg = total_mass_kg * (share_mass_pct / 100.0)
            # Preserve ALL original fields (factor_value_kg_co2e_per_kg, factor_key,
            # status, notes, etc.) so product authors can ground specific lines to
            # manufacturer LCAs / EPDs.
            merged = dict(entry)
            merged["mass_kg"] = mass_kg
            merged["share_mass_pct"] = share_mass_pct
            merged.setdefault("source_ref_ids", entry.get("source_ref_ids", []))
            resolved_entries.append(merged)

        if any(_to_float(entry.get("mass_kg")) is not None for entry in resolved_entries):
            return resolved_entries
        if use_bootstrap:
            estimated_entries = product.get("estimation_profile", {}).get("raw_materials", []) or []
            if estimated_entries:
                assumptions.append("Used estimated raw-material profile because official material masses are missing.")
                return estimated_entries
            return self._bootstrap_materials(product, total_mass_kg, assumptions)
        return resolved_entries

    def _calculate_raw_materials_stage(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        total_mass_kg: Optional[float],
        use_bootstrap: bool,
        assumptions: List[str],
    ) -> CarbonStageResult:
        entries = self._material_inputs(product, scenario, total_mass_kg, use_bootstrap, assumptions)
        factor_values = scenario.get("raw_material_factor_values", {}) or {}
        factor_keys = scenario.get("raw_material_factor_keys", {}) or {}
        traces: List[CarbonTraceItem] = []
        missing_inputs: List[str] = []
        total = 0.0

        if not entries:
            missing_inputs.append("raw_materials: no material inputs are available.")

        for idx, entry in enumerate(entries, start=1):
            material_key = entry.get("material_key") or f"material_{idx}"
            mass_kg = _to_float(entry.get("mass_kg"))
            label = material_key
            factor = self._resolve_generic_factor(
                table=self._raw_material_factors,
                factor_key=entry.get("factor_key") or factor_keys.get(material_key),
                explicit_value=_to_float(entry.get("factor_value"))
                or _to_float(entry.get("factor_value_kg_co2e_per_kg"))
                or _to_float(factor_values.get(material_key)),
                unit="kg CO2e/kg",
                default_key=material_key,
                assumption_prefix="Used",
                explicit_source_ref=entry.get("source_ref"),
            )
            notes = factor.notes[:]
            if mass_kg is None:
                missing_inputs.append(f"raw_materials:{material_key} mass_kg is missing.")
                notes.append("Material mass is missing.")
            emissions = None
            status = "computed"
            if mass_kg is not None and factor.factor_value is not None:
                emissions = mass_kg * factor.factor_value
                total += emissions
            else:
                status = "missing"
                if factor.factor_value is None:
                    missing_inputs.append(f"raw_materials:{material_key} factor is missing.")
            traces.append(
                CarbonTraceItem(
                    item_id=material_key,
                    label=label,
                    stage="raw_materials",
                    activity_value=mass_kg,
                    activity_unit="kg",
                    factor_value=factor.factor_value,
                    factor_unit=factor.factor_unit,
                    emissions_kg_co2e=emissions,
                    formula="mass_kg × factor_kgCO2e_per_kg",
                    status=status,
                    source_refs=_collect_source_refs(entry.get("source_ref_ids"), entry.get("source_ref"), factor.source_ref),
                    notes=notes,
                )
            )

        stage_status = "complete" if traces and not missing_inputs else ("partial" if traces else "missing")
        estimated_inputs = [
            entry.get("material_key") or f"material_{idx + 1}"
            for idx, entry in enumerate(entries)
            if self._status_looks_estimated(entry.get("status"))
        ]
        if scenario.get("raw_materials"):
            quality_status = "scenario_override"
        elif estimated_inputs:
            quality_status = "estimated"
        else:
            quality_status = "exact"
        return CarbonStageResult(
            stage="raw_materials",
            total_kg_co2e=total,
            status=stage_status,
            traces=traces,
            missing_inputs=_unique(missing_inputs),
            quality_status=quality_status,
            uncertainty_pct=self._stage_uncertainty_default(product, "raw_materials", quality_status),
            estimated_inputs=_unique(estimated_inputs),
        )

    def _transport_inputs(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        use_bootstrap: bool,
        assumptions: List[str],
    ) -> List[Dict[str, Any]]:
        scenario_entries = scenario.get("transport_legs")
        if isinstance(scenario_entries, list) and scenario_entries:
            return scenario_entries
        official_entries = product.get("life_cycle_inputs", {}).get("transport_legs", []) or []
        has_official_value = any(
            _transport_mode_alias(entry.get("mode") or entry.get("mode_key")) is not None
            or _to_float(entry.get("distance_km")) is not None
            or _to_float(entry.get("mass_kg")) is not None
            for entry in official_entries
        )
        if has_official_value or not use_bootstrap:
            return official_entries

        estimated_entries = product.get("estimation_profile", {}).get("transport_legs", []) or []
        if estimated_entries:
            assumptions.append("Used estimated transport legs because official transport-route data is missing.")
            return estimated_entries
        return official_entries

    def _calculate_transport_stage(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        total_mass_kg: Optional[float],
        use_bootstrap: bool,
        assumptions: List[str],
    ) -> CarbonStageResult:
        entries = self._transport_inputs(product, scenario, use_bootstrap, assumptions)
        traces: List[CarbonTraceItem] = []
        missing_inputs: List[str] = []
        total = 0.0

        if not entries:
            missing_inputs.append("transportation: no transport leg inputs are available.")

        for idx, entry in enumerate(entries, start=1):
            leg_id = entry.get("leg_id") or f"transport_leg_{idx}"
            mode = _transport_mode_alias(entry.get("mode") or entry.get("mode_key"))
            distance_km = _to_float(entry.get("distance_km"))
            mass_kg = _to_float(entry.get("mass_kg"))
            notes: List[str] = []
            if mass_kg is None and total_mass_kg is not None:
                mass_kg = total_mass_kg
                notes.append("Used total product mass for transport leg mass.")
                assumptions.append(f"Used total product mass for transport leg {leg_id}.")
            factor = self._resolve_generic_factor(
                table=self._transport_factors,
                factor_key=entry.get("factor_key"),
                explicit_value=_to_float(entry.get("factor_value"))
                or _to_float(entry.get("factor_value_kg_co2e_per_ton_km")),
                unit="kg CO2e/ton-km",
                default_key=f"transport_{mode}_generic" if mode else None,
                assumption_prefix="Used",
                explicit_source_ref=entry.get("source_ref"),
            )
            notes.extend(factor.notes)
            if mode is None:
                missing_inputs.append(f"transportation:{leg_id} mode is missing.")
                notes.append("Transport mode is missing.")
            if distance_km is None:
                missing_inputs.append(f"transportation:{leg_id} distance_km is missing.")
                notes.append("Transport distance is missing.")
            if mass_kg is None:
                missing_inputs.append(f"transportation:{leg_id} mass_kg is missing.")
                notes.append("Transport mass is missing.")

            activity = None
            emissions = None
            status = "computed"
            if mass_kg is not None and distance_km is not None:
                activity = (mass_kg / 1000.0) * distance_km
            if activity is not None and factor.factor_value is not None:
                emissions = activity * factor.factor_value
                total += emissions
            else:
                status = "missing"
                if factor.factor_value is None:
                    missing_inputs.append(f"transportation:{leg_id} factor is missing.")
            traces.append(
                CarbonTraceItem(
                    item_id=leg_id,
                    label=leg_id,
                    stage="transportation",
                    activity_value=activity,
                    activity_unit="ton-km",
                    factor_value=factor.factor_value,
                    factor_unit=factor.factor_unit,
                    emissions_kg_co2e=emissions,
                    formula="(mass_kg / 1000) × distance_km × factor_kgCO2e_per_ton_km",
                    status=status,
                    source_refs=_collect_source_refs(entry.get("source_ref_ids"), entry.get("source_ref"), factor.source_ref),
                    notes=notes,
                )
            )

        stage_status = "complete" if traces and not missing_inputs else ("partial" if traces else "missing")
        estimated_inputs = [
            entry.get("leg_id") or f"transport_leg_{idx + 1}"
            for idx, entry in enumerate(entries)
            if self._status_looks_estimated(entry.get("status"))
        ]
        if scenario.get("transport_legs"):
            quality_status = "scenario_override"
        elif estimated_inputs:
            quality_status = "estimated"
        else:
            quality_status = "exact"
        return CarbonStageResult(
            stage="transportation",
            total_kg_co2e=total,
            status=stage_status,
            traces=traces,
            missing_inputs=_unique(missing_inputs),
            quality_status=quality_status,
            uncertainty_pct=self._stage_uncertainty_default(product, "transportation", quality_status),
            estimated_inputs=_unique(estimated_inputs),
        )

    def _calculate_use_phase_stage(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        use_bootstrap: bool,
        assumptions: List[str],
        warnings: List[str],
    ) -> CarbonStageResult:
        use_data = dict(product.get("life_cycle_inputs", {}).get("use_phase", {}) or {})
        use_data.update((scenario.get("use_phase") or {}))
        used_estimated_defaults = False
        if use_bootstrap:
            estimated_use = product.get("estimation_profile", {}).get("use_phase", {}) or {}
            if estimated_use:
                before = dict(use_data)
                use_data = self._merge_missing_values(use_data, estimated_use)
                if use_data != before:
                    used_estimated_defaults = True
                    assumptions.append("Used estimated use-phase defaults where official MX431adn use data is missing.")

        annual_energy_kwh = _to_float(use_data.get("annual_energy_kwh"))
        lifetime_years = _to_float(use_data.get("lifetime_years"))
        lifetime_energy_kwh = _to_float(use_data.get("lifetime_energy_kwh"))
        country_code = (
            use_data.get("country_code")
            or use_data.get("electricity_country_code")
            or scenario.get("use_country_code")
            or product.get("defaults", {}).get("use_country_code")
        )
        report_year = int(
            use_data.get("report_year")
            or use_data.get("electricity_year")
            or scenario.get("report_year")
            or product.get("defaults", {}).get("report_year")
            or 2021
        )
        include_paper = bool(use_data.get("include_paper_default", product.get("defaults", {}).get("include_paper", False)))

        notes: List[str] = []
        missing_inputs: List[str] = []
        if lifetime_energy_kwh is None and annual_energy_kwh is not None and lifetime_years is not None:
            lifetime_energy_kwh = annual_energy_kwh * lifetime_years
            assumptions.append("Derived lifetime use-phase electricity from annual energy and lifetime years.")
        if lifetime_energy_kwh is None:
            missing_inputs.append("use_phase:lifetime_energy_kwh is missing.")
            notes.append("Provide lifetime_energy_kwh or both annual_energy_kwh and lifetime_years.")

        factor = self._resolve_electricity_factor(
            country_code=country_code,
            report_year=report_year,
            factor_key=use_data.get("electricity_factor_key") or product.get("defaults", {}).get("electricity_factor_key"),
            explicit_value=_to_float(use_data.get("electricity_factor_value"))
            or _to_float(use_data.get("electricity_factor_value_kg_co2e_per_kwh")),
            explicit_source_ref=use_data.get("source_ref"),
        )
        notes.extend(factor.notes)
        if factor.factor_value is None:
            missing_inputs.append("use_phase:electricity factor is missing.")
        if include_paper:
            warnings.append("Paper-inclusive use-phase calculation is not implemented yet; current calculation excludes paper impacts.")
            notes.append("Paper impacts were not added to the use-phase calculation.")

        emissions = None
        status = "computed"
        if lifetime_energy_kwh is not None and factor.factor_value is not None:
            emissions = lifetime_energy_kwh * factor.factor_value
        else:
            status = "missing"

        trace = CarbonTraceItem(
            item_id="use_phase_electricity",
            label=f"Use-phase electricity ({country_code or 'unknown country'}, {report_year})",
            stage="use_phase",
            activity_value=lifetime_energy_kwh,
            activity_unit="kWh",
            factor_value=factor.factor_value,
            factor_unit=factor.factor_unit,
            emissions_kg_co2e=emissions,
            formula="lifetime_energy_kwh × electricity_factor_kgCO2e_per_kWh",
            status=status,
            source_refs=_collect_source_refs(use_data.get("source_ref_ids"), use_data.get("source_ref"), factor.source_ref),
            notes=notes,
        )
        if scenario.get("use_phase"):
            quality_status = "scenario_override"
        elif used_estimated_defaults:
            quality_status = "estimated"
        else:
            quality_status = "exact"
        return CarbonStageResult(
            stage="use_phase",
            total_kg_co2e=emissions or 0.0,
            status="complete" if status == "computed" and not missing_inputs else "partial",
            traces=[trace],
            missing_inputs=_unique(missing_inputs),
            quality_status=quality_status,
            uncertainty_pct=self._stage_uncertainty_default(product, "use_phase", quality_status),
            estimated_inputs=["use_phase"] if quality_status == "estimated" else [],
        )

    def _calculate_end_of_life_stage(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        total_mass_kg: Optional[float],
        use_bootstrap: bool,
        assumptions: List[str],
        warnings: List[str],
    ) -> CarbonStageResult:
        eol_data = dict(product.get("life_cycle_inputs", {}).get("end_of_life", {}) or {})
        eol_data.update((scenario.get("end_of_life") or {}))
        used_estimated_defaults = False
        if use_bootstrap:
            estimated_eol = product.get("estimation_profile", {}).get("end_of_life", {}) or {}
            if estimated_eol:
                before = dict(eol_data)
                eol_data = self._merge_missing_values(eol_data, estimated_eol)
                if eol_data != before:
                    used_estimated_defaults = True
                    assumptions.append("Used estimated end-of-life defaults where the declared MX431adn scenario is missing.")

        mass_kg = _to_float(eol_data.get("mass_kg"))
        if mass_kg is None and total_mass_kg is not None:
            mass_kg = total_mass_kg
            assumptions.append("Used total product mass for end-of-life waste mass.")

        route_factor_values = eol_data.get("route_factor_values", {}) or {}
        route_factor_keys = eol_data.get("route_factor_keys", {}) or {}
        route_factors = eol_data.get("route_factors", {}) or {}
        routes = [
            ("recycling", _to_float(eol_data.get("recycling_rate_pct")) or _to_float(eol_data.get("recyclability_pct"))),
            ("incineration", _to_float(eol_data.get("incineration_rate_pct")) or _to_float(eol_data.get("incineration_pct"))),
            ("landfill", _to_float(eol_data.get("landfill_rate_pct")) or _to_float(eol_data.get("landfill_pct"))),
        ]

        traces: List[CarbonTraceItem] = []
        missing_inputs: List[str] = []
        total = 0.0
        provided_rates = [rate for _, rate in routes if rate is not None]
        if not provided_rates:
            missing_inputs.append("end_of_life: no route percentages are available.")
        elif abs(sum(provided_rates) - 100.0) > 0.01:
            warnings.append(f"End-of-life route percentages sum to {sum(provided_rates):.3f} instead of 100.")

        for route, rate_pct in routes:
            notes: List[str] = []
            route_factor = route_factors.get(route, {}) or {}
            factor = self._resolve_generic_factor(
                table=self._end_of_life_factors,
                factor_key=route_factor.get("factor_key") or route_factor_keys.get(route),
                explicit_value=_to_float(route_factor.get("factor_value"))
                or _to_float(route_factor.get("factor_value_kg_co2e_per_kg"))
                or _to_float(route_factor_values.get(route)),
                unit="kg CO2e/kg",
                default_key=f"eol_{route}_mixed_electronics",
                assumption_prefix="Used",
                explicit_source_ref=route_factor.get("source_ref"),
            )
            notes.extend(factor.notes)
            if rate_pct is None:
                missing_inputs.append(f"end_of_life:{route} rate is missing.")
                notes.append("Route percentage is missing.")
            if mass_kg is None:
                missing_inputs.append("end_of_life:mass_kg is missing.")
                notes.append("Waste mass is missing.")

            activity = None
            emissions = None
            status = "computed"
            if mass_kg is not None and rate_pct is not None:
                activity = mass_kg * (rate_pct / 100.0)
            if activity is not None and factor.factor_value is not None:
                emissions = activity * factor.factor_value
                total += emissions
            else:
                status = "missing"
                if factor.factor_value is None:
                    missing_inputs.append(f"end_of_life:{route} factor is missing.")

            traces.append(
                CarbonTraceItem(
                    item_id=f"end_of_life_{route}",
                    label=f"End-of-life {route}",
                    stage="end_of_life",
                    activity_value=activity,
                    activity_unit="kg",
                    factor_value=factor.factor_value,
                    factor_unit=factor.factor_unit,
                    emissions_kg_co2e=emissions,
                    formula="route_mass_kg × factor_kgCO2e_per_kg",
                    status=status,
                    source_refs=_collect_source_refs(eol_data.get("source_ref_ids"), route_factor.get("source_ref"), factor.source_ref),
                    notes=notes,
                )
            )

        stage_status = "complete" if traces and not missing_inputs else ("partial" if traces else "missing")
        if scenario.get("end_of_life"):
            quality_status = "scenario_override"
        elif used_estimated_defaults:
            quality_status = "estimated"
        else:
            quality_status = "exact"
        return CarbonStageResult(
            stage="end_of_life",
            total_kg_co2e=total,
            status=stage_status,
            traces=traces,
            missing_inputs=_unique(missing_inputs),
            quality_status=quality_status,
            uncertainty_pct=self._stage_uncertainty_default(product, "end_of_life", quality_status),
            estimated_inputs=["end_of_life"] if quality_status == "estimated" else [],
        )

    def _calculate_recyclability(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        total_mass_kg: Optional[float],
        use_bootstrap: bool,
    ) -> RecyclabilityResult:
        eol_data = dict(product.get("life_cycle_inputs", {}).get("end_of_life", {}) or {})
        eol_data.update((scenario.get("end_of_life") or {}))
        if use_bootstrap:
            estimated_eol = product.get("estimation_profile", {}).get("end_of_life", {}) or {}
            if estimated_eol:
                eol_data = self._merge_missing_values(eol_data, estimated_eol)
        recycling_rate = _to_float(eol_data.get("recycling_rate_pct")) or _to_float(eol_data.get("recyclability_pct"))
        incineration_rate = _to_float(eol_data.get("incineration_rate_pct")) or _to_float(eol_data.get("incineration_pct"))
        landfill_rate = _to_float(eol_data.get("landfill_rate_pct")) or _to_float(eol_data.get("landfill_pct"))

        notes: List[str] = []
        if use_bootstrap:
            notes.extend(product.get("bootstrap_estimates", {}).get("circularity_notes", []))

        recoverable_mass = None
        incineration_mass = None
        landfill_mass = None
        if total_mass_kg is not None and recycling_rate is not None:
            recoverable_mass = total_mass_kg * (recycling_rate / 100.0)
        if total_mass_kg is not None and incineration_rate is not None:
            incineration_mass = total_mass_kg * (incineration_rate / 100.0)
        if total_mass_kg is not None and landfill_rate is not None:
            landfill_mass = total_mass_kg * (landfill_rate / 100.0)

        if recycling_rate is not None and incineration_rate is not None and landfill_rate is not None and total_mass_kg is not None:
            status = "complete"
        elif recycling_rate is not None:
            status = "partial"
            notes.append("Only partial recyclability inputs are available.")
        else:
            status = "missing"
            notes.append("Recycling split is not available.")

        return RecyclabilityResult(
            recyclability_pct=recycling_rate,
            recoverable_mass_kg=recoverable_mass,
            incineration_mass_kg=incineration_mass,
            landfill_mass_kg=landfill_mass,
            status=status,
            notes=notes,
        )

    def _build_provenance(
        self,
        product: Dict[str, Any],
        scenario: Dict[str, Any],
        result: Dict[str, CarbonStageResult],
        total_mass_kg: Optional[float],
        used_bootstrap: bool,
    ) -> List[CarbonProvenanceItem]:
        entries: List[CarbonProvenanceItem] = []
        observed_mass = self._observed_fact(product, "product_mass_kg")
        if _to_float(scenario.get("total_product_mass_kg")) is not None:
            entries.append(
                CarbonProvenanceItem(
                    field_name="total_product_mass_kg",
                    label="Product mass",
                    value=_to_float(scenario.get("total_product_mass_kg")),
                    unit="kg",
                    status="scenario_override",
                    method="direct_input",
                    source_refs=["scenario_override"],
                    uncertainty_pct=10.0,
                )
            )
        elif observed_mass and total_mass_kg is not None:
            entries.append(
                CarbonProvenanceItem(
                    field_name="total_product_mass_kg",
                    label="Product mass",
                    value=total_mass_kg,
                    unit="kg",
                    status=observed_mass.get("status", "exact"),
                    method="official_observation",
                    source_refs=self._resolve_source_refs(product, observed_mass.get("source_ref_ids", [])),
                    notes=list(observed_mass.get("notes", [])),
                    uncertainty_pct=8.0,
                )
            )

        for field_name, label in [
            ("packaged_mass_kg", "Packaged mass"),
            ("packaging_mass_kg", "Packaging mass"),
        ]:
            fact = self._observed_fact(product, field_name)
            value = self._fact_numeric_value(fact)
            if fact and value is not None:
                entries.append(
                    CarbonProvenanceItem(
                        field_name=field_name,
                        label=label,
                        value=value,
                        unit=str(fact.get("unit") or "kg"),
                        status=str(fact.get("status") or "exact"),
                        method="official_observation" if fact.get("status") == "exact" else "derived_from_observation",
                        source_refs=self._resolve_source_refs(product, fact.get("source_ref_ids", [])),
                        notes=list(fact.get("notes", [])),
                        uncertainty_pct=12.0 if fact.get("status") == "derived" else 8.0,
                    )
                )

        tec_fact = self._observed_fact(product, "tec_kwh_per_week")
        if tec_fact and ("use_phase" in result or used_bootstrap):
            entries.append(
                CarbonProvenanceItem(
                    field_name="tec_kwh_per_week",
                    label="Typical energy consumption",
                    value=_to_float(tec_fact.get("preferred_value")),
                    unit=str(tec_fact.get("unit") or "kWh/week"),
                    status=str(tec_fact.get("status") or "exact"),
                    method="certified_registry_preference",
                    source_refs=self._resolve_source_refs(product, tec_fact.get("source_ref_ids", [])),
                    notes=[str(tec_fact.get("selection_reason") or "")] + list(tec_fact.get("notes", [])),
                    uncertainty_pct=10.0,
                )
            )

        if result.get("raw_materials") and result["raw_materials"].quality_status == "estimated":
            mix = product.get("bootstrap_estimates", {}).get("material_mix_pct") or {}
            entries.append(
                CarbonProvenanceItem(
                    field_name="raw_material_mix",
                    label="Estimated raw-material mix",
                    value=mix,
                    unit="mass_pct",
                    status="estimated",
                    method="nearest_family_analog",
                    source_refs=self._resolve_source_refs(product, ["sample_mx622_profile", "internal_estimation_profile"]),
                    notes=["Derived from the nearest structured Lexmark family reference because official BOM masses are missing."],
                    uncertainty_pct=self._stage_uncertainty_default(product, "raw_materials", "estimated"),
                )
            )

        if result.get("transportation") and result["transportation"].quality_status == "estimated":
            transport_legs = product.get("estimation_profile", {}).get("transport_legs", []) or []
            entries.append(
                CarbonProvenanceItem(
                    field_name="transport_route",
                    label="Estimated transport route",
                    value=transport_legs,
                    unit="route_profile",
                    status="estimated",
                    method="generic_distribution_defaults",
                    source_refs=self._resolve_source_refs(product, ["internal_estimation_profile"]),
                    notes=["Used generic ship plus truck distribution defaults because the real route is not declared."],
                    uncertainty_pct=self._stage_uncertainty_default(product, "transportation", "estimated"),
                )
            )

        if result.get("use_phase") and result["use_phase"].quality_status == "estimated":
            use_phase = product.get("estimation_profile", {}).get("use_phase", {}) or {}
            entries.append(
                CarbonProvenanceItem(
                    field_name="use_phase_profile",
                    label="Estimated use-phase profile",
                    value={
                        "lifetime_years": use_phase.get("lifetime_years"),
                        "lifetime_energy_kwh": use_phase.get("lifetime_energy_kwh"),
                        "electricity_factor_value_kg_co2e_per_kwh": use_phase.get("electricity_factor_value_kg_co2e_per_kwh"),
                        "country_code": use_phase.get("country_code"),
                        "report_year": use_phase.get("report_year"),
                    },
                    unit="profile",
                    status="estimated",
                    method="tec_plus_default_lifetime",
                    source_refs=self._resolve_source_refs(product, use_phase.get("source_ref_ids", [])),
                    notes=list(use_phase.get("notes", [])),
                    uncertainty_pct=self._stage_uncertainty_default(product, "use_phase", "estimated"),
                )
            )

        if result.get("end_of_life") and result["end_of_life"].quality_status == "estimated":
            eol = product.get("estimation_profile", {}).get("end_of_life", {}) or {}
            entries.append(
                CarbonProvenanceItem(
                    field_name="end_of_life_split",
                    label="Estimated end-of-life split",
                    value={
                        "recycling_rate_pct": eol.get("recycling_rate_pct"),
                        "incineration_rate_pct": eol.get("incineration_rate_pct"),
                        "landfill_rate_pct": eol.get("landfill_rate_pct"),
                    },
                    unit="mass_pct",
                    status="estimated",
                    method="generic_weee_default",
                    source_refs=self._resolve_source_refs(product, eol.get("source_ref_ids", [])),
                    notes=list(eol.get("notes", [])),
                    uncertainty_pct=self._stage_uncertainty_default(product, "end_of_life", "estimated"),
                )
            )

        return entries

    def _summarize_quality(
        self,
        product: Dict[str, Any],
        stage_results: Dict[str, CarbonStageResult],
        total_kg_co2e: Optional[float],
        partial_total_kg_co2e: float,
    ) -> Tuple[str, List[str], Optional[float], Optional[float], Optional[Dict[str, float]]]:
        estimated_fields: List[str] = []
        has_estimated = False
        has_override = False
        weighted_uncertainty = 0.0
        weighted_base = 0.0
        total_reference = total_kg_co2e if total_kg_co2e is not None else (partial_total_kg_co2e or 0.0)

        for stage_name, stage in stage_results.items():
            if stage.quality_status == "estimated":
                has_estimated = True
                estimated_fields.extend(stage.estimated_inputs or [stage_name])
            elif stage.quality_status == "scenario_override":
                has_override = True
            stage_uncertainty = stage.uncertainty_pct or self._stage_uncertainty_default(product, stage_name, stage.quality_status)
            if stage.total_kg_co2e > 0 and stage_uncertainty is not None:
                weighted_uncertainty += stage.total_kg_co2e * stage_uncertainty
                weighted_base += stage.total_kg_co2e

        if has_estimated:
            quality_status = "hybrid_estimate" if total_kg_co2e is not None else "partial_estimate"
        elif has_override:
            quality_status = "scenario_override" if total_kg_co2e is not None else "partial_override"
        else:
            quality_status = "exact" if total_kg_co2e is not None else "partial"

        uncertainty_pct = None
        uncertainty_kg = None
        uncertainty_range = None
        if total_reference and total_reference > 0:
            if weighted_base > 0:
                uncertainty_pct = weighted_uncertainty / weighted_base
            else:
                uncertainty_pct = _to_float(product.get("estimation_profile", {}).get("uncertainty_defaults_pct", {}).get("total"))
            if uncertainty_pct is not None:
                uncertainty_kg = total_reference * (uncertainty_pct / 100.0)
                uncertainty_range = {
                    "low": max(0.0, total_reference - uncertainty_kg),
                    "high": total_reference + uncertainty_kg,
                }

        return quality_status, _unique(estimated_fields), uncertainty_pct, uncertainty_kg, uncertainty_range


_SERVICE_SINGLETON: Optional[CarbonCalculationService] = None


def get_carbon_service() -> CarbonCalculationService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = CarbonCalculationService()
    return _SERVICE_SINGLETON


def calculate_carbon_footprint(product_id: str, scenario: Optional[Dict[str, Any]] = None) -> CarbonCalculationResult:
    return get_carbon_service().calculate(product_id=product_id, scenario=scenario)
