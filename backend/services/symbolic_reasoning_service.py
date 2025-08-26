from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import RDFS
from rdflib.plugins.sparql import prepareQuery

try:
    from owlrl import DeductiveClosure, OWLRL_Semantics  # type: ignore
    OWL_AVAILABLE = True
except Exception:
    OWL_AVAILABLE = False

from backend.config.domain import resolve_ontology_path

logger = logging.getLogger(__name__)

# Namespaces per domain
EX_DPP = Namespace("http://example.com/dpp#")
EX_TXT = Namespace("http://example.com/textiles#")
EX_VSM = Namespace("http://example.com/viessmann#")
EX_LXM = Namespace("http://example.com/lexmark#")


@dataclass(frozen=True)
class ReasonerConfig:
    ontology_path: str
    run_owl_rl: bool = True
    domain: str = "battery"  # battery | textiles | viessmann | lexmark


class SymbolicReasoner:
    """
    RDF/OWL reasoning service with domain-aware rules and toggleable rule IDs.
    """

    def __init__(self, cfg: ReasonerConfig):
        self.cfg = cfg
        if not os.path.exists(cfg.ontology_path):
            raise FileNotFoundError(f"Ontology not found: {cfg.ontology_path}")

        g = Graph()
        g.parse(cfg.ontology_path, format="turtle")
        logger.info("Loaded ontology: %s (triples=%d)", cfg.ontology_path, len(g))

        if cfg.run_owl_rl and OWL_AVAILABLE:
            logger.info("Running OWL-RL deductive closure...")
            DeductiveClosure(OWLRL_Semantics).expand(g)
            logger.info("OWL-RL expansion complete (triples=%d)", len(g))
        else:
            if cfg.run_owl_rl:
                logger.warning("OWL-RL requested but 'owlrl' not available. Skipping.")
            else:
                logger.info("OWL-RL disabled by config.")

        # Domain namespace
        self.EX = {
            "battery":   EX_DPP,
            "textiles":  EX_TXT,
            "viessmann": EX_VSM,
            "lexmark":   EX_LXM,
        }[cfg.domain]

        # Base snapshot
        self._base_graph = Graph()
        for t in g: self._base_graph.add(t)
        self._base_graph.bind("ex", str(self.EX)); self._base_graph.bind("rdfs", str(RDFS))

        self.graph = Graph()
        for t in self._base_graph: self.graph.add(t)
        self.graph.bind("ex", str(self.EX)); self.graph.bind("rdfs", str(RDFS))

        self._rules: List[Tuple[str, Any]] = []
        self._disabled_rules: set[str] = set()
        self._prepare_rules()

    # ---------- Rules per domain ----------
    def _prepare_rules(self) -> None:
        d = self.cfg.domain
        if d == "battery":
            EX = "http://example.com/dpp#"
            self._rules = [
                ("bat_requires_battery_safety", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:BatterySafetyStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:Battery . }}
                """)),
                ("bat_requires_battery_step", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresStep ex:BatteryTestStep. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:Battery .
                            FILTER NOT EXISTS {{ ?p ex:hasStep ex:BatteryTestStep }} }}
                """)),
                ("bat_requires_wireless_compliance", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:WirelessComplianceStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:WirelessModule . }}
                """)),
                ("bat_requires_wireless_step", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresStep ex:WirelessTestStep. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:WirelessModule .
                            FILTER NOT EXISTS {{ ?p ex:hasStep ex:WirelessTestStep }} }}
                """)),
                ("bat_lead_implies_rohs", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:RoHSStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesMaterial ex:LeadMaterial . }}
                """)),
            ]
        elif d == "textiles":
            EX = "http://example.com/textiles#"
            self._rules = [
                ("txt_care_label_for_any_fabric", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:CareLabelStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesFabric ?f . ?f rdf:type ex:Fabric . }}
                """)),
                ("txt_wool_care_standard", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:WoolCareStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesFabric ?f . ?f rdf:type ex:WoolFabric . }}
                """)),
                ("txt_wool_wash_step", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresStep ex:WoolWashTest. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesFabric ?f . ?f rdf:type ex:WoolFabric .
                            FILTER NOT EXISTS {{ ?p ex:requiresStep ex:WoolWashTest }} }}
                """)),
                ("txt_label_check_step", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresStep ex:LabelCheckStep. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesFabric ?f .
                            FILTER NOT EXISTS {{ ?p ex:requiresStep ex:LabelCheckStep }} }}
                """)),
            ]
        elif d == "viessmann":
            EX = "http://example.com/viessmann#"
            self._rules = [
                # Any refrigerant -> F-Gas compliance
                ("hvac_fgas_for_refrigerant", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:FGasStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesRefrigerant ?r . }}
                """)),
                # Compressor -> pressure + leak tests
                ("hvac_compressor_tests", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresStep ex:PressureTestStep, ex:LeakCheckStep. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:Compressor . }}
                """)),
                # Electrical module -> electrical safety compliance + step
                ("hvac_electrical_safety", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:ElectricalSafetyStandard ;
                                   ex:requiresStep       ex:ElectricalSafetyTest . }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:ElectricalModule . }}
                """)),
                # Lead material -> RoHS
                ("hvac_lead_implies_rohs", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:RoHSStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesMaterial ex:LeadMaterial . }}
                """)),
                # Optional: wireless module on the heat pump -> wireless compliance/test
                ("hvac_wireless_rules", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:WirelessComplianceStandard ;
                                   ex:requiresStep       ex:WirelessTestStep . }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:WirelessModule . }}
                """)),
            ]
        else:  # lexmark
            EX = "http://example.com/lexmark#"
            self._rules = [
                # Wireless -> wireless compliance + wireless test
                ("prn_wireless_rules", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:WirelessComplianceStandard ;
                                   ex:requiresStep       ex:WirelessTestStep . }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:WirelessModule . }}
                """)),
                # Printer head -> print quality test
                ("prn_head_quality_step", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresStep ex:PrintQualityTest. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:PrinterHead . }}
                """)),
                # Toner -> WEEE/label check
                ("prn_toner_label_weee", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:WEEEStandard ;
                                   ex:requiresStep       ex:LabelCheckStep . }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:TonerCartridge . }}
                """)),
                # Main board/electrical -> EMC + Safety
                ("prn_emc_and_safety", prepareQuery(f"""
                    PREFIX ex:  <{EX}> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:EMCStandard, ex:Safety62368 . }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c rdf:type ex:MainBoard . }}
                """)),
                # Lead -> RoHS
                ("prn_lead_implies_rohs", prepareQuery(f"""
                    PREFIX ex:  <{EX}>
                    CONSTRUCT {{ ?p ex:requiresCompliance ex:RoHSStandard. }}
                    WHERE {{ ?p ex:hasComponent ?c . ?c ex:usesMaterial ex:LeadMaterial . }}
                """)),
            ]

    # ---------- Apply / toggles ----------
    def _fresh_from_base(self) -> Graph:
        g = Graph()
        for t in self._base_graph: g.add(t)
        g.bind("ex", str(self.EX)); g.bind("rdfs", str(RDFS))
        return g

    def apply_rules(self) -> int:
        self.graph = self._fresh_from_base()
        before = len(self.graph)
        for rid, q in self._rules:
            if rid in self._disabled_rules: continue
            for triple in self.graph.query(q): self.graph.add(triple)
        added = len(self.graph) - before
        logger.info("Rule application added ~%d triples (total=%d). Disabled=%s",
                    added, len(self.graph), sorted(self._disabled_rules) or "[]")
        return added

    def disable_rules(self, rule_ids: List[str]) -> None:
        self._disabled_rules = set(rule_ids or []); self.apply_rules()

    def enable_all_rules(self) -> None:
        self._disabled_rules.clear(); self.apply_rules()

    # ---------- Queries ----------
    def requires_compliance(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(self.EX[product_uri], self.EX.requiresCompliance)]

    def requires_steps(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(self.EX[product_uri], self.EX.requiresStep)]


def build_reasoner(ontology_path: Optional[str] = None,
                   run_owl_rl: bool = True,
                   domain: str | None = None) -> SymbolicReasoner:
    dom = (domain or os.environ.get("DPP_DOMAIN") or "battery").strip().lower()
    onto = ontology_path or resolve_ontology_path(dom)
    cfg = ReasonerConfig(ontology_path=onto, run_owl_rl=run_owl_rl, domain=dom)
    r = SymbolicReasoner(cfg); r.apply_rules(); return r


@dataclass
class SymTrace:
    product: str
    asserted: List[Tuple[str, str, str]]
    inferred: List[Tuple[str, str, str]]
    rules_fired: List[str]


@dataclass
class SymAnswer:
    text: str
    evidence: List[Tuple[str, str, str]]
    fired: bool
    trace: SymTrace


_REASONER_SINGLETON: Optional[SymbolicReasoner] = None

def _ensure_reasoner() -> SymbolicReasoner:
    global _REASONER_SINGLETON
    try:
        from backend.main import app
        r = getattr(app.state, "reasoner", None)
        if r is not None:
            return r
    except Exception:
        pass
    if _REASONER_SINGLETON is None:
        _REASONER_SINGLETON = build_reasoner(run_owl_rl=True)
    return _REASONER_SINGLETON


def _label_or_qname(g: Graph, node: URIRef) -> str:
    lab = g.value(node, RDFS.label)
    if lab: return str(lab)
    try: return g.qname(node)
    except Exception: return str(node)


def answer_symbolic(query: str, product: Optional[str], session: str) -> Optional[SymAnswer]:
    if not product: return None
    r = _ensure_reasoner()
    stds = r.requires_compliance(product)
    steps = r.requires_steps(product)
    if not stds and not steps: return None

    ev: List[Tuple[str, str, str]] = []
    if stds:
        for uri in stds:
            ev.append((f"ex:{product}", "requiresCompliance", _label_or_qname(r.graph, URIRef(uri))))
    if steps:
        for uri in steps:
            ev.append((f"ex:{product}", "requiresStep", _label_or_qname(r.graph, URIRef(uri))))

    domain_text = {
        "battery":   "standards",
        "textiles":  "care/standards",
        "viessmann": "compliance",
        "lexmark":   "compliance",
    }.get(r.cfg.domain, "standards")

    parts = []
    if stds:
        parts.append(f"Symbolic: {domain_text} for {product}: " +
                     ", ".join(_label_or_qname(r.graph, URIRef(u)) for u in stds) + ".")
    if steps:
        parts.append("Required steps: " +
                     ", ".join(_label_or_qname(r.graph, URIRef(u)) for u in steps) + ".")
    text = " ".join(parts)

    trace = SymTrace(product=product, asserted=[], inferred=ev.copy(),
                     rules_fired=(["requiresCompliance"] if stds else []) +
                                 (["requiresStep"] if steps else []))
    return SymAnswer(text=text, evidence=ev, fired=True, trace=trace)


def sym_fire_flags(query: str, product: Optional[str]) -> bool:
    if not product: return False
    r = _ensure_reasoner()
    p = r.EX[product]
    return ((p, r.EX.requiresCompliance, None) in r.graph) or ((p, r.EX.requiresStep, None) in r.graph)


# Rule toggle wrappers (for faithfulness)
def disable_rules(rule_ids: List[str]) -> None:
    r = _ensure_reasoner(); r.disable_rules(rule_ids or [])

def enable_all_rules() -> None:
    r = _ensure_reasoner(); r.enable_all_rules()
