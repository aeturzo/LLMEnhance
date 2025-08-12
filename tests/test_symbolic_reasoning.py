import os
from backend.services.symbolic_reasoning_service import build_reasoner

def test_reasoner_loads_and_rules_apply():
    # Ensure ontology path resolves (ENV override optional)
    ontology = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "ontologies", "dpp_ontology.ttl"))
    sr = build_reasoner(ontology_path=ontology, run_owl_rl=True)

    products = [p.split("#")[-1] for p in sr.list_products()]
    assert set(products) >= {"ProductA", "ProductB", "ProductC"}

    # ProductA has battery + wireless + lead -> 3 compliances + 2 steps required
    reqA = [r.split("#")[-1] for r in sr.check_compliance_requirements("ProductA")]
    assert set(reqA) == {"BatterySafetyStandard", "WirelessComplianceStandard", "RoHSStandard"}

    stepsA = [s.split("#")[-1] for s in sr.suggest_missing_steps("ProductA")]
    assert set(stepsA) == {"BatteryTestStep", "WirelessTestStep"}

    # ProductB (lead only) -> RoHS only, no required steps
    reqB = [r.split("#")[-1] for r in sr.check_compliance_requirements("ProductB")]
    assert set(reqB) == {"RoHSStandard"}
    stepsB = [s.split("#")[-1] for s in sr.suggest_missing_steps("ProductB")]
    assert len(stepsB) == 0

    # ProductC (wireless module, already has WirelessTestStep)
    reqC = [r.split("#")[-1] for r in sr.check_compliance_requirements("ProductC")]
    assert set(reqC) == {"WirelessComplianceStandard"}
    stepsC = [s.split("#")[-1] for s in sr.suggest_missing_steps("ProductC")]
    assert len(stepsC) == 0
