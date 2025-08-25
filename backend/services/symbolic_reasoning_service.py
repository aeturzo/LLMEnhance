# backend/services/symbolic_reasoning_service.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set

from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import RDFS
from rdflib.plugins.sparql import prepareQuery

# Optional: enable OWL RL closure. Safe to disable if performance sensitive.
try:
    from owlrl import DeductiveClosure, OWLRL_Semantics  # type: ignore
    OWL_AVAILABLE = True
except Exception:
    OWL_AVAILABLE = False

logger = logging.getLogger(__name__)

# IMPORTANT: keep this in sync with your TTL prefix (you use example.com)
EX = Namespace("http://example.com/dpp#")


# -----------------------------------------------------------------------------
# Core reasoner (your existing structure, preserved)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonerConfig:
    ontology_path: str
    run_owl_rl: bool = True  # allow to disable OWL-RL when scaling


class SymbolicReasoner:
    """
    Production-leanable RDF/OWL reasoning service for DPP.
    - Loads ontology & instance data
    - Applies OWL-RL (optional) and custom business rules
    - Serves queries: compliance requirements, missing steps, product listing, etc.
    """

    def __init__(self, cfg: ReasonerConfig):
        self.cfg = cfg
        if not os.path.exists(cfg.ontology_path):
            raise FileNotFoundError(f"Ontology not found: {cfg.ontology_path}")

        self.graph = Graph()
        self.graph.parse(cfg.ontology_path, format="turtle")
        logger.info("Loaded ontology: %s (triples=%d)", cfg.ontology_path, len(self.graph))

        if cfg.run_owl_rl and OWL_AVAILABLE:
            logger.info("Running OWL-RL deductive closure...")
            DeductiveClosure(OWLRL_Semantics).expand(self.graph)
            logger.info("OWL-RL expansion complete (triples=%d)", len(self.graph))
        else:
            if cfg.run_owl_rl:
                logger.warning("OWL-RL requested but 'owlrl' not available. Skipping OWL-RL.")
            else:
                logger.info("OWL-RL disabled by config.")

        # Pre-compile construct queries (rules)
        self._prepare_rules()

    # ---------- Public API ----------
    def apply_rules(self) -> int:
        """
        Run all rule CONSTRUCTs and add inferred triples.
        Returns the number of triples added (best-effort estimate).
        """
        before = len(self.graph)
        for q in (
            self._q_battery_requires_compliance,
            self._q_battery_requires_step,
            self._q_wireless_requires_compliance,
            self._q_wireless_requires_step,
            self._q_lead_requires_rohs,
        ):
            self._construct_into_graph(q)
        added = len(self.graph) - before
        logger.info("Rule application added ~%d triples (total=%d)", added, len(self.graph))
        return added

    def list_products(self) -> List[str]:
        return [str(s) for s in self.graph.subjects(RDF.type, EX.Product)]

    def list_components(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(EX[product_uri], EX.hasComponent)]

    def check_compliance_requirements(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(EX[product_uri], EX.requiresCompliance)]

    def suggest_missing_steps(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(EX[product_uri], EX.requiresStep)]

    def list_process_steps(self, product_uri: str) -> List[str]:
        return [str(o) for o in self.graph.objects(EX[product_uri], EX.hasStep)]

    # ---------- Internal: rules & helpers ----------
    def _construct_into_graph(self, q):
        # Important: iterate results explicitly; CONSTRUCT returns triples
        for triple in self.graph.query(q):
            self.graph.add(triple)

    def _prepare_rules(self):
        # Battery compliance
        self._q_battery_requires_compliance = prepareQuery("""
            PREFIX ex:  <http://example.com/dpp#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            CONSTRUCT { ?p ex:requiresCompliance ex:BatterySafetyStandard. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:Battery .
            }
        """)

        # Battery test step (only if not already present)
        self._q_battery_requires_step = prepareQuery("""
            PREFIX ex:  <http://example.com/dpp#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            CONSTRUCT { ?p ex:requiresStep ex:BatteryTestStep. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:Battery .
              FILTER NOT EXISTS { ?p ex:hasStep ex:BatteryTestStep }
            }
        """)

        # Wireless compliance
        self._q_wireless_requires_compliance = prepareQuery("""
            PREFIX ex:  <http://example.com/dpp#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            CONSTRUCT { ?p ex:requiresCompliance ex:WirelessComplianceStandard. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:WirelessModule .
            }
        """)

        # Wireless test step (only if not already present)
        self._q_wireless_requires_step = prepareQuery("""
            PREFIX ex:  <http://example.com/dpp#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            CONSTRUCT { ?p ex:requiresStep ex:WirelessTestStep. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:WirelessModule .
              FILTER NOT EXISTS { ?p ex:hasStep ex:WirelessTestStep }
            }
        """)

        # Lead → RoHS compliance
        self._q_lead_requires_rohs = prepareQuery("""
            PREFIX ex: <http://example.com/dpp#>
            CONSTRUCT { ?p ex:requiresCompliance ex:RoHSStandard. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c ex:usesMaterial ex:LeadMaterial .
            }
        """)


# ---- factory for DI / app startup ----
def build_reasoner(ontology_path: Optional[str] = None, run_owl_rl: bool = True) -> SymbolicReasoner:
    """
    Create a SymbolicReasoner with sane defaults.
    - ontology_path: if None, use ENV DPP_ONTOLOGY (fallback to ./backend/ontologies/dpp_ontology.ttl)
    """
    if ontology_path is None:
        ontology_path = os.environ.get(
            "DPP_ONTOLOGY",
            os.path.join(os.path.dirname(__file__), "..", "ontologies", "dpp_ontology.ttl"),
        )
        ontology_path = os.path.abspath(ontology_path)

    cfg = ReasonerConfig(ontology_path=ontology_path, run_owl_rl=run_owl_rl)
    sr = SymbolicReasoner(cfg)
    sr.apply_rules()  # populate requiresCompliance/requiresStep now
    return sr


# -----------------------------------------------------------------------------
# Day-3 public API: richer answers, traces, rule toggles, and fire flags
# -----------------------------------------------------------------------------
@dataclass
class SymTrace:
    product: Optional[str]
    asserted: List[Tuple[str, str, str]]   # triples already in KG (labels/qnames)
    inferred: List[Tuple[str, str, str]]   # triples predicted by rules
    rules_fired: List[str]                 # rule ids that produced at least one triple

@dataclass
class SymAnswer:
    text: str
    evidence: List[Tuple[str, str, str]]
    fired: bool
    trace: SymTrace

# Runtime toggle for ablations
_DISABLED_RULES: Set[str] = set()

def disable_rules(rule_ids: List[str]) -> None:
    _DISABLED_RULES.update(rule_ids)

def enable_all_rules() -> None:
    _DISABLED_RULES.clear()

# ---------- helpers for readable traces ----------
def _label_or_qname(g: Graph, node) -> str:
    lab = g.value(node, RDFS.label)
    if lab:
        return str(lab)
    try:
        return g.qname(node)
    except Exception:
        return str(node)

def _fmt_triple(g: Graph, s, p, o) -> Tuple[str, str, str]:
    return (_label_or_qname(g, s), _label_or_qname(g, p), _label_or_qname(g, o))

# ---------- Python versions of rules (for trace/inference & toggling) ----------
def _rule_R1(g: Graph, p: URIRef) -> List[Tuple[URIRef, URIRef, URIRef]]:
    """Battery present → requiresCompliance BatterySafetyStandard"""
    for c in g.objects(p, EX.hasComponent):
        if (c, RDF.type, EX.Battery) in g:
            return [(p, EX.requiresCompliance, EX.BatterySafetyStandard)]
    return []

def _rule_R2(g: Graph, p: URIRef) -> List[Tuple[URIRef, URIRef, URIRef]]:
    """Wireless present → requiresCompliance WirelessComplianceStandard"""
    for c in g.objects(p, EX.hasComponent):
        if (c, RDF.type, EX.WirelessModule) in g:
            return [(p, EX.requiresCompliance, EX.WirelessComplianceStandard)]
    return []

def _rule_R3(g: Graph, p: URIRef) -> List[Tuple[URIRef, URIRef, URIRef]]:
    """Any component uses Lead → requiresCompliance RoHS"""
    for c in g.objects(p, EX.hasComponent):
        if (c, EX.usesMaterial, EX.LeadMaterial) in g:
            return [(p, EX.requiresCompliance, EX.RoHSStandard)]
    return []

_RULES: Dict[str, callable] = {
    "R1_BATTERY_SAFETY": _rule_R1,
    "R2_WIRELESS_COMPLIANCE": _rule_R2,
    "R3_ROHS": _rule_R3,
}

def _apply_rules_for(product: str, g: Graph) -> Tuple[List[Tuple[URIRef, URIRef, URIRef]], List[str]]:
    inferred: List[Tuple[URIRef, URIRef, URIRef]] = []
    fired: List[str] = []
    p = EX[product]
    for rid, fn in _RULES.items():
        if rid in _DISABLED_RULES:
            continue
        triples = fn(g, p)
        if triples:
            fired.append(rid)
            inferred.extend(triples)
    # de-dup while preserving order
    seen = set()
    uniq: List[Tuple[URIRef, URIRef, URIRef]] = []
    for t in triples if False else inferred:  # keep order; no walrus tricks
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq, fired

def _collect_asserted_standards(product: str, g: Graph) -> List[Tuple[URIRef, URIRef, URIRef]]:
    p = EX[product]
    triples: List[Tuple[URIRef, URIRef, URIRef]] = []
    for pred in (EX.requiresCompliance, EX.conformsTo):
        for std in g.objects(p, pred):
            triples.append((p, pred, std))
    return triples

def _partition_rule_like(
    product: str,
    g: Graph,
    asserted: List[Tuple[URIRef, URIRef, URIRef]]
) -> Tuple[List[Tuple[URIRef, URIRef, URIRef]], List[Tuple[URIRef, URIRef, URIRef]]]:
    """
    Separate asserted triples that 'look like' they came from our rules
    (so we can treat them as inferred when rules are disabled).
    """
    rule_like: List[Tuple[URIRef, URIRef, URIRef]] = []
    base_like: List[Tuple[URIRef, URIRef, URIRef]] = []

    p = EX[product]
    has_batt = any((c, RDF.type, EX.Battery) in g for c in g.objects(p, EX.hasComponent))
    has_wire = any((c, RDF.type, EX.WirelessModule) in g for c in g.objects(p, EX.hasComponent))
    has_lead = any((c, EX.usesMaterial, EX.LeadMaterial) in g for c in g.objects(p, EX.hasComponent))

    for (s, pred, o) in asserted:
        if pred not in (EX.requiresCompliance, EX.conformsTo):
            base_like.append((s, pred, o))
            continue
        if o == EX.BatterySafetyStandard and has_batt:
            rule_like.append((s, pred, o)); continue
        if o == EX.WirelessComplianceStandard and has_wire:
            rule_like.append((s, pred, o)); continue
        if o == EX.RoHSStandard and has_lead:
            rule_like.append((s, pred, o)); continue
        # otherwise consider base KG fact (e.g., EN 62133-2)
        base_like.append((s, pred, o))

    return base_like, rule_like

# ---------- Singleton access ----------
_REASONER_SINGLETON: Optional[SymbolicReasoner] = None

def _ensure_reasoner() -> SymbolicReasoner:
    global _REASONER_SINGLETON
    if _REASONER_SINGLETON is None:
        _REASONER_SINGLETON = build_reasoner(run_owl_rl=True)
    return _REASONER_SINGLETON

# ---------- Feature flag used by policy_features ----------
def sym_fire_flags(query: str, product: Optional[str]) -> bool:
    """
    True if product has (asserted OR rule-inferable) link to a standard.
    """
    if not product:
        return False
    r = _ensure_reasoner()
    asserted = _collect_asserted_standards(product, r.graph)
    if asserted:
        return True
    inferred, _ = _apply_rules_for(product, r.graph)
    return any(pred in (EX.requiresCompliance, EX.conformsTo) for _, pred, _ in inferred)

# ---------- Public QA API consumed by /solve ----------
def answer_symbolic(query: str, product: Optional[str], session: str) -> Optional[SymAnswer]:
    """
    Visible, template-based KG answerer with trace:
    - For 'standard' queries: list asserted + rule-inferred standards for product.
    - For 'label/include' queries: surface short labels attached to product requirements.
    Returns None if nothing relevant is found.
    """
    if not product:
        return None

    r = _ensure_reasoner()
    g = r.graph
    ql = (query or "").lower().strip()

    # Gather asserted and inferable triples
    asserted_raw = _collect_asserted_standards(product, g)
    base_like, rule_like_from_asserted = _partition_rule_like(product, g, asserted_raw)
    inferred_raw, rules_fired = _apply_rules_for(product, g)

    # Build display lists
    asserted_disp = [_fmt_triple(g, s, p, o) for (s, p, o) in base_like]
    inferred_disp = [_fmt_triple(g, s, p, o) for (s, p, o) in (rule_like_from_asserted + inferred_raw)]

    # Make a unique ordered list of standard labels
    std_nodes: List[URIRef] = []
    for _, _, o in base_like + rule_like_from_asserted + inferred_raw:
        std_nodes.append(o)

    seen_labels: Set[str] = set()
    labels: List[str] = []
    for node in std_nodes:
        lab = _label_or_qname(g, node)
        if lab not in seen_labels:
            seen_labels.add(lab)
            labels.append(lab)

    # 1) Standards-style question
    if "standard" in ql or "appl" in ql:
        if labels:
            text = f"Symbolic: Standards for {product}: " + "; ".join(labels) + "."
            return SymAnswer(
                text=text,
                evidence=asserted_disp + inferred_disp,
                fired=bool(asserted_disp or inferred_disp),
                trace=SymTrace(
                    product=product,
                    asserted=asserted_disp,
                    inferred=inferred_disp,
                    rules_fired=rules_fired,
                ),
            )
        # Nothing relevant found
        return SymAnswer(
            text="No result found.",
            evidence=asserted_disp + inferred_disp,
            fired=bool(asserted_disp or inferred_disp),
            trace=SymTrace(
                product=product,
                asserted=asserted_disp,
                inferred=inferred_disp,
                rules_fired=rules_fired,
            ),
        )

    # 2) Label/requirements (very lightweight)
    if "label" in ql or "include" in ql:
        reqs: List[str] = []
        for node in std_nodes:
            lab = g.value(node, RDFS.label)
            if lab:
                s = str(lab).strip()
                if s and len(s) <= 40 and s.lower() != "rohs":
                    if s not in reqs:
                        reqs.append(s)
        if reqs:
            text = "Symbolic: A battery label must include: " + ", ".join(reqs) + "."
            return SymAnswer(
                text=text,
                evidence=asserted_disp + inferred_disp,
                fired=bool(asserted_disp or inferred_disp),
                trace=SymTrace(
                    product=product,
                    asserted=asserted_disp,
                    inferred=inferred_disp,
                    rules_fired=rules_fired,
                ),
            )

    # 3) Fallback: not a symbolic-ready query
    return None
