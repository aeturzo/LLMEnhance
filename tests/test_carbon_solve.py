import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).parent
CARBON_CASES = ROOT / "carbon" / "test.jsonl"
SOLVE_PATH = ROOT.parent / "backend" / "api" / "solve.py"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_solve_module_with_stubs():
    class _RouterModel:
        @staticmethod
        def load(path):
            return None

    class _HybridRetriever:
        def __init__(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

    stubbed = {
        "backend.services.memory_service": _stub_module(
            "backend.services.memory_service",
            retrieve=lambda *args, **kwargs: [],
        ),
        "backend.services.search_service": _stub_module(
            "backend.services.search_service",
            search=lambda *args, **kwargs: [],
        ),
        "backend.services.policy_router": _stub_module(
            "backend.services.policy_router",
            RouterModel=_RouterModel,
            MODEL_PATH="stub-policy-router",
        ),
        "backend.services.symbolic_reasoning_service": _stub_module(
            "backend.services.symbolic_reasoning_service",
            answer_symbolic=lambda *args, **kwargs: None,
            sym_fire_flags=lambda *args, **kwargs: 0,
        ),
        "backend.retrieval.hybrid": _stub_module(
            "backend.retrieval.hybrid",
            HybridRetriever=_HybridRetriever,
        ),
        "backend.api.answerer_ctx": _stub_module(
            "backend.api.answerer_ctx",
            answer_with_context=lambda question, passages: "Insufficient context.",
        ),
    }

    previous = {name: sys.modules.get(name) for name in stubbed}
    try:
        sys.modules.update(stubbed)
        module_name = "test_carbon_solve_module"
        spec = importlib.util.spec_from_file_location(module_name, SOLVE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class CarbonSolveRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solve_module = _load_solve_module_with_stubs()
        app = FastAPI()
        app.include_router(solve_module.router)
        cls.client = TestClient(app)

    def test_carbon_cases_from_jsonl(self):
        for case in _load_jsonl(CARBON_CASES):
            with self.subTest(case_id=case["id"], question=case["question"]):
                response = self.client.post(
                    "/solve",
                    json={"query": case["question"], "product": case.get("product"), "session": "carbon-test"},
                )
                self.assertEqual(response.status_code, 200)
                body = response.json()

                self.assertEqual(body["mode"], case["expected_mode"])
                self.assertEqual(body.get("product"), case["expected_response_product"])
                self.assertIn(case["expected_answer_contains"], body["answer"])

                source_ids = [item.get("id") for item in body.get("sources", [])]
                for expected_source_id in case["expected_source_ids"]:
                    self.assertIn(expected_source_id, source_ids)

                step = body["steps"][0]
                self.assertEqual(step["status"], case["expected_status"])

                if case["expected_product_id"] is None:
                    self.assertIsNone(body.get("carbon"))
                    continue

                carbon = body["carbon"]
                self.assertEqual(carbon["status"], case["expected_status"])
                self.assertEqual(carbon["requested_stages"], case["expected_stage_scope"])
                self.assertEqual(carbon["quality_status"], case["expected_quality_status"])

                if case["type"] == "stage_breakdown":
                    self.assertEqual(
                        set(carbon["stage_results"].keys()),
                        {"raw_materials", "transportation", "use_phase", "end_of_life"},
                    )

                if case["type"] == "total_footprint":
                    self.assertIsNotNone(carbon["total_kg_co2e"])
                    self.assertGreater(carbon["uncertainty_pct"], 0.0)

                if case["type"] == "recyclability":
                    self.assertEqual(list(carbon["stage_results"].keys()), ["end_of_life"])

                if case["type"] == "exact_only":
                    self.assertGreater(len(carbon["missing_inputs"]), 0)


if __name__ == "__main__":
    unittest.main()
