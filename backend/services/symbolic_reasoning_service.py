from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import OWL, RDFS, XSD
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

    # Backward-compatible query names used by the original evaluation tests.
    def list_products(self) -> List[str]:
        return sorted(str(s) for s in self.graph.subjects(RDF.type, self.EX.Product))

    def check_compliance_requirements(self, product_uri: str) -> List[str]:
        return self.requires_compliance(product_uri)

    def suggest_missing_steps(self, product_uri: str) -> List[str]:
        return self.requires_steps(product_uri)


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


SUPPORTED_DOMAINS = ("battery", "textiles", "viessmann", "lexmark")
_REASONER_SINGLETON: Optional[SymbolicReasoner] = None
_REASONER_CACHE: Dict[str, SymbolicReasoner] = {}


def _normalize_domain(domain: Optional[str]) -> str:
    dom = (domain or os.environ.get("DPP_DOMAIN") or "battery").strip().lower()
    return dom if dom in SUPPORTED_DOMAINS else "battery"


def _ensure_reasoner(domain: Optional[str] = None) -> SymbolicReasoner:
    global _REASONER_SINGLETON
    dom = _normalize_domain(domain)
    try:
        from backend.main import app
        reasoners = getattr(app.state, "reasoners", None)
        if isinstance(reasoners, dict) and dom in reasoners:
            return reasoners[dom]
        if dom == "battery":
            r = getattr(app.state, "reasoner", None)
            if r is not None:
                return r
    except Exception:
        pass

    if dom in _REASONER_CACHE:
        return _REASONER_CACHE[dom]

    reasoner = build_reasoner(run_owl_rl=True, domain=dom)
    _REASONER_CACHE[dom] = reasoner
    if dom == "battery":
        _REASONER_SINGLETON = reasoner
    return reasoner


def _label_or_qname(g: Graph, node: URIRef) -> str:
    lab = g.value(node, RDFS.label)
    if lab: return str(lab)
    try: return g.qname(node)
    except Exception: return str(node)


_CLASS_QUERY_RE = re.compile(r"^\s*is\s+(.+?)\s+an?\s+(.+?)\??\s*$", re.IGNORECASE)
_KB_RECALL_RE = re.compile(r"^\s*what is the\s+(.+?)\s+of\s+(.+?)\??\s*$", re.IGNORECASE)


def _compact_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _uri_local(node: URIRef) -> str:
    text = str(node)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def _all_uri_nodes(g: Graph) -> set[URIRef]:
    nodes: set[URIRef] = set()
    for s, p, o in g:
        if isinstance(s, URIRef):
            nodes.add(s)
        if isinstance(p, URIRef):
            nodes.add(p)
        if isinstance(o, URIRef):
            nodes.add(o)
    return nodes


def _nodes_matching(g: Graph, label: str) -> set[URIRef]:
    target = _compact_token(label)
    out: set[URIRef] = set()
    for node in _all_uri_nodes(g):
        if _compact_token(_uri_local(node)) == target:
            out.add(node)
            continue
        lab = g.value(node, RDFS.label)
        if lab is not None and _compact_token(str(lab)) == target:
            out.add(node)
    return out


def _class_nodes_matching(g: Graph, label: str) -> set[URIRef]:
    target = _compact_token(label)
    builtins = {
        "thing": OWL.Thing,
        "class": OWL.Class,
        "datatype": RDFS.Datatype,
        "annotationproperty": OWL.AnnotationProperty,
        "objectproperty": OWL.ObjectProperty,
        "datatypeproperty": OWL.DatatypeProperty,
        "property": RDF.Property,
        "resource": RDFS.Resource,
        "namedindividual": OWL.NamedIndividual,
    }
    out: set[URIRef] = set()
    if target in builtins:
        out.add(URIRef(builtins[target]))
    for node in _all_uri_nodes(g):
        if _compact_token(_uri_local(node)) == target:
            out.add(node)
            continue
        lab = g.value(node, RDFS.label)
        if lab is not None and _compact_token(str(lab)) == target:
            out.add(node)
    return out


def _properties_matching(g: Graph, label: str) -> set[URIRef]:
    target = _compact_token(label)
    out: set[URIRef] = set()
    for node in _all_uri_nodes(g):
        declared_property = (
            (node, RDF.type, RDF.Property) in g
            or (node, RDF.type, OWL.ObjectProperty) in g
            or (node, RDF.type, OWL.DatatypeProperty) in g
            or (node, RDF.type, OWL.AnnotationProperty) in g
        )
        lab = g.value(node, RDFS.label)
        if lab is not None and _compact_token(str(lab)) == target:
            out.add(node)
            continue
        local = _uri_local(node)
        if declared_property and _compact_token(local) == target:
            out.add(node)
    return out


def _type_closure(g: Graph, node: URIRef) -> set[URIRef]:
    out: set[URIRef] = set()
    stack = [t for t in g.objects(node, RDF.type) if isinstance(t, URIRef)]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(
            p for p in g.objects(cur, RDFS.subClassOf)
            if isinstance(p, URIRef) and p not in out
        )
    return out


def _is_xsd_datatype_name(name: str) -> bool:
    target = _compact_token(name)
    for attr in (
        "integer", "string", "boolean", "decimal", "float", "double",
        "dateTime", "dateTimeStamp", "date", "time", "nonNegativeInteger",
        "nonPositiveInteger", "positiveInteger", "negativeInteger",
        "unsignedLong", "unsignedInt", "unsignedShort", "unsignedByte",
        "long", "int", "short", "byte", "anyURI",
    ):
        if _compact_token(attr) == target:
            return True
        uri = getattr(XSD, attr, None)
        if uri is not None and _compact_token(_uri_local(URIRef(uri))) == target:
            return True
    return False


def _answer_class_membership(r: SymbolicReasoner, query: str) -> Optional[SymAnswer]:
    m = _CLASS_QUERY_RE.match((query or "").strip())
    if not m:
        return None
    entity_label = m.group(1).strip()
    class_label = m.group(2).strip()
    class_c = _compact_token(class_label)

    nodes = _nodes_matching(r.graph, entity_label)
    class_nodes = _class_nodes_matching(r.graph, class_label)

    if class_c == "thing":
        proved = bool(nodes)
    elif class_c == "datatype":
        proved = _is_xsd_datatype_name(entity_label) or any(
            RDFS.Datatype in _type_closure(r.graph, node) for node in nodes
        )
    elif class_c == "annotationproperty" and _compact_token(entity_label) in {
        "comment", "label", "seealso", "isdefinedby",
    }:
        proved = True
    else:
        proved = any(
            target in _type_closure(r.graph, node)
            for node in nodes
            for target in class_nodes
        )

    if not proved:
        return None
    text = f"Yes. {entity_label} is {'an' if class_label[:1].lower() in 'aeiou' else 'a'} {class_label}."
    ev = [(entity_label, "rdf:type", class_label)]
    trace = SymTrace(product=entity_label, asserted=[], inferred=ev.copy(),
                     rules_fired=["class_membership"] if proved else [])
    return SymAnswer(text=text, evidence=ev, fired=True, trace=trace)


def _answer_kb_recall(r: SymbolicReasoner, query: str) -> Optional[SymAnswer]:
    m = _KB_RECALL_RE.match((query or "").strip())
    if not m:
        return None
    relation_label = m.group(1).strip()
    entity_label = m.group(2).strip()
    subjects = _nodes_matching(r.graph, entity_label)
    predicates = _properties_matching(r.graph, relation_label)
    if not subjects or not predicates:
        return None

    values: List[str] = []
    for subj in subjects:
        for pred in predicates:
            for obj in r.graph.objects(subj, pred):
                if isinstance(obj, URIRef):
                    values.append(_label_or_qname(r.graph, obj))
                else:
                    values.append(str(obj))

    distinct: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = _compact_token(value)
        if key and key not in seen:
            seen.add(key)
            distinct.append(value)
    if len(distinct) != 1:
        return None

    value = distinct[0]
    text = f"{value}."
    ev = [(entity_label, relation_label, value)]
    trace = SymTrace(product=entity_label, asserted=[], inferred=ev.copy(),
                     rules_fired=["kb_recall"])
    return SymAnswer(text=text, evidence=ev, fired=True, trace=trace)


def _component_has_type(r: SymbolicReasoner, component: URIRef, class_local: str) -> bool:
    target = _compact_token(class_local)
    for cls in _type_closure(r.graph, component):
        if _compact_token(_uri_local(cls)) == target:
            return True
        lab = r.graph.value(cls, RDFS.label)
        if lab is not None and _compact_token(str(lab)) == target:
            return True
    return False


def _answer_records_logic(r: SymbolicReasoner, query: str, product: str) -> Optional[SymAnswer]:
    q = (query or "").lower()
    if "(records)" not in q and not q.startswith(("does ", "do ", "would ", "is ")):
        return None

    p = r.EX[product]
    components = [c for c in r.graph.objects(p, r.EX.hasComponent) if isinstance(c, URIRef)]

    verdict: Optional[bool] = None
    reason = ""
    if "lead" in q:
        verdict = any((c, r.EX.usesMaterial, r.EX.LeadMaterial) in r.graph for c in components)
        reason = "lead-containing component"
    elif "refrigerant" in q:
        verdict = any((c, r.EX.usesRefrigerant, None) in r.graph for c in components)
        reason = "refrigerant use"
    elif "wireless" in q:
        verdict = any(_component_has_type(r, c, "WirelessModule") for c in components)
        reason = "wireless module"
    elif "battery" in q:
        verdict = any(_component_has_type(r, c, "Battery") for c in components)
        reason = "battery component"
    elif "compliance standard" in q or "linked to any compliance" in q:
        verdict = bool(list(r.graph.objects(p, r.EX.requiresCompliance)) or
                       list(r.graph.objects(p, r.EX.conformsTo)))
        reason = "compliance standard"

    if verdict is None:
        return None

    text = f"{'Yes' if verdict else 'No'}. {product} {'has' if verdict else 'does not have'} {reason} evidence."
    ev = [(f"ex:{product}", "records_logic", reason)] if verdict else []
    trace = SymTrace(product=product, asserted=[], inferred=ev.copy(),
                     rules_fired=["records_logic"] if verdict else [])
    return SymAnswer(text=text, evidence=ev, fired=True, trace=trace)


def _answer_component_lookup(r: SymbolicReasoner, query: str, product: str) -> Optional[SymAnswer]:
    if "component" not in (query or "").lower():
        return None
    p = r.EX[product]
    components = [c for c in r.graph.objects(p, r.EX.hasComponent) if isinstance(c, URIRef)]
    if not components:
        return None
    labels = [_label_or_qname(r.graph, c) for c in components]
    clean = [label.split(":", 1)[-1] for label in labels]
    text = "Components: " + ", ".join(clean) + "."
    ev = [(f"ex:{product}", "hasComponent", label) for label in clean]
    trace = SymTrace(product=product, asserted=[], inferred=ev.copy(),
                     rules_fired=["component_lookup"])
    return SymAnswer(text=text, evidence=ev, fired=True, trace=trace)


def answer_symbolic(query: str, product: Optional[str], session: str, domain: Optional[str] = None) -> Optional[SymAnswer]:
    r = _ensure_reasoner(domain)
    class_answer = _answer_class_membership(r, query)
    if class_answer:
        return class_answer
    recall_answer = _answer_kb_recall(r, query)
    if recall_answer:
        return recall_answer

    if not product: return None
    component_answer = _answer_component_lookup(r, query, product)
    if component_answer:
        return component_answer
    records_answer = _answer_records_logic(r, query, product)
    if records_answer:
        return records_answer

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


def sym_fire_flags(query: str, product: Optional[str], domain: Optional[str] = None) -> bool:
    r = _ensure_reasoner(domain)
    if _answer_class_membership(r, query):
        return True
    if _answer_kb_recall(r, query):
        return True
    if product and _answer_component_lookup(r, query, product):
        return True
    if product and _answer_records_logic(r, query, product):
        return True
    if not product: return False
    p = r.EX[product]
    return ((p, r.EX.requiresCompliance, None) in r.graph) or ((p, r.EX.requiresStep, None) in r.graph)


# Rule toggle wrappers (for faithfulness)
def disable_rules(rule_ids: List[str]) -> None:
    r = _ensure_reasoner(); r.disable_rules(rule_ids or [])

def enable_all_rules() -> None:
    r = _ensure_reasoner(); r.enable_all_rules()
