# backend/config/domain.py
from __future__ import annotations
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

def resolve_ontology_path(domain: str | None) -> str:
    """
    Map a logical domain -> ontology path.
      - battery   -> backend/ontologies/dpp_ontology.ttl
      - textiles  -> backend/ontologies/textiles_ontology.ttl
      - viessmann -> backend/ontologies/viessmann_ontology.ttl
      - lexmark   -> backend/ontologies/lexmark_ontology.ttl
    Highest precedence: DPP_ONTOLOGY env (absolute/relative path).
    """
    env_path = os.environ.get("DPP_ONTOLOGY")
    if env_path:
        return str(pathlib.Path(env_path).resolve())

    dom = (domain or os.environ.get("DPP_DOMAIN") or "battery").strip().lower()
    onto_dir = ROOT / "backend" / "ontologies"
    mapping = {
        "battery":   onto_dir / "dpp_ontology.ttl",
        "textiles":  onto_dir / "textiles_ontology.ttl",
        "viessmann": onto_dir / "viessmann_ontology.ttl",
        "lexmark":   onto_dir / "lexmark_ontology.ttl",
    }
    path = mapping.get(dom, mapping["battery"])
    return str(path.resolve())
