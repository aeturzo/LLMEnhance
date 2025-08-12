from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple

from rdflib import Graph, Namespace, RDF
from rdflib.namespace import RDFS
from rdflib.plugins.sparql import prepareQuery

# Optional: enable OWL RL closure. Safe to disable if performance sensitive.
try:
    from owlrl import DeductiveClosure, OWLRL_Semantics  # type: ignore
    OWL_AVAILABLE = True
except Exception:
    OWL_AVAILABLE = False

logger = logging.getLogger(__name__)
EX = Namespace("http://example.com/dpp#")


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
            PREFIX ex: <http://example.com/dpp#>
            CONSTRUCT { ?p ex:requiresCompliance ex:BatterySafetyStandard. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:Battery .
            }
        """)

        # Battery test step (only if not already present)
        self._q_battery_requires_step = prepareQuery("""
            PREFIX ex: <http://example.com/dpp#>
            CONSTRUCT { ?p ex:requiresStep ex:BatteryTestStep. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:Battery .
              FILTER NOT EXISTS { ?p ex:hasStep ex:BatteryTestStep }
            }
        """)

        # Wireless compliance
        self._q_wireless_requires_compliance = prepareQuery("""
            PREFIX ex: <http://example.com/dpp#>
            CONSTRUCT { ?p ex:requiresCompliance ex:WirelessComplianceStandard. }
            WHERE {
              ?p ex:hasComponent ?c .
              ?c rdf:type ex:WirelessModule .
            }
        """)

        # Wireless test step (only if not already present)
        self._q_wireless_requires_step = prepareQuery("""
            PREFIX ex: <http://example.com/dpp#>
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