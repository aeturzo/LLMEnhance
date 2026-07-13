import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.solve_auto import router as solve_auto_router
from backend.services import memory_service


class SolveAutoRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(solve_auto_router)
        cls.client = TestClient(app)

    def setUp(self):
        memory_service.flush_session("auto-test")

    def tearDown(self):
        memory_service.flush_session("auto-test")

    def test_memory_like_query_prefers_session_memory(self):
        memory_service.add_memory(
            "auto-test",
            "For ProductA, the preferred packaging is recycled corrugated cardboard and the preferred supplier is GreenCells.",
        )

        response = self.client.post(
            "/solve_auto",
            json={
                "query": "What packaging did I say ProductA prefers?",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "AUTO_COMPOSE")
        self.assertEqual(body["session"], "auto-test")
        self.assertIn("recycled corrugated cardboard", body["answer"])
        self.assertIn("answer_trace", body)
        self.assertFalse(body["answer_trace"]["llm_used"])

        mem_step = next(step for step in body["steps"] if step["source"] == "MEM")
        self.assertTrue(mem_step["included"])
        self.assertGreater(mem_step["hit_count"], 0)

    def test_packaging_variant_query_still_returns_memory_answer(self):
        memory_service.add_memory(
            "auto-test",
            "For ProductA, the preferred packaging is recycled corrugated cardboard and the preferred supplier is GreenCells.",
        )

        response = self.client.post(
            "/solve_auto",
            json={
                "query": "What packaging ProductA prefers?",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "AUTO_COMPOSE")
        self.assertIn("preferred packaging", body["answer"].lower())
        self.assertIn("recycled corrugated cardboard", body["answer"].lower())
        self.assertEqual(body["product"], "ProductA")

        compose_step = next(step for step in body["steps"] if step["source"] == "COMPOSE")
        self.assertTrue(compose_step["memory_like_query"] or compose_step["memory_dominant"])

    def test_carbon_query_delegates_to_existing_carbon_flow(self):
        response = self.client.post(
            "/solve_auto",
            json={
                "query": "What is the carbon footprint of Lexmark MX431adn?",
                "product": "lexmark_mx431adn",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "CARBON")
        self.assertIn("carbon", body)
        self.assertEqual(body["product"], "lexmark_mx431adn")

    def test_symbolic_query_infers_product_from_question_text(self):
        response = self.client.post(
            "/solve_auto",
            json={
                "query": "Name two compliance standards that apply to ProductA.",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "AUTO_COMPOSE")
        self.assertEqual(body["product"], "ProductA")

        sym_step = next(step for step in body["steps"] if step["source"] == "SYM")
        compose_step = next(step for step in body["steps"] if step["source"] == "COMPOSE")
        self.assertTrue(sym_step["included"])
        self.assertEqual(compose_step["effective_product"], "ProductA")
        self.assertEqual(compose_step["product_inferred_from_query"], "ProductA")
        self.assertEqual(compose_step["included_passage_ids"], ["sym:ProductA"])

    def test_blended_query_without_product_field_still_includes_symbolic_context(self):
        memory_service.add_memory(
            "auto-test",
            "For ProductA, the preferred packaging is recycled corrugated cardboard and the preferred supplier is GreenCells.",
        )

        response = self.client.post(
            "/solve_auto",
            json={
                "query": "What packaging did I say ProductA prefers, and name one compliance standard for ProductA.",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["product"], "ProductA")
        self.assertIn("recycled corrugated cardboard", body["answer"].lower())
        self.assertTrue("standard" in body["answer"].lower() or "[sym:producta]" in body["answer"].lower())

        sym_step = next(step for step in body["steps"] if step["source"] == "SYM")
        self.assertTrue(sym_step["included"])
        self.assertNotEqual(body["answer_trace"]["path"], "memory_direct")

    def test_lexmark_domain_selection_enables_symbolic_reasoning(self):
        response = self.client.post(
            "/solve_auto",
            json={
                "query": "Which compliance requirements apply to PrinterL1?",
                "domain": "lexmark",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "AUTO_COMPOSE")
        self.assertEqual(body["domain"], "lexmark")
        self.assertEqual(body["product"], "PrinterL1")

        sym_step = next(step for step in body["steps"] if step["source"] == "SYM")
        compose_step = next(step for step in body["steps"] if step["source"] == "COMPOSE")
        self.assertTrue(sym_step["included"])
        self.assertEqual(sym_step["domain"], "lexmark")
        self.assertEqual(compose_step["selected_domain"], "lexmark")
        self.assertEqual(compose_step["effective_domain"], "lexmark")

    def test_viessmann_domain_selection_enables_symbolic_reasoning(self):
        response = self.client.post(
            "/solve_auto",
            json={
                "query": "Which compliance requirements apply to ProductV1?",
                "domain": "viessmann",
                "session": "auto-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "AUTO_COMPOSE")
        self.assertEqual(body["domain"], "viessmann")
        self.assertEqual(body["product"], "ProductV1")

        sym_step = next(step for step in body["steps"] if step["source"] == "SYM")
        compose_step = next(step for step in body["steps"] if step["source"] == "COMPOSE")
        self.assertTrue(sym_step["included"])
        self.assertEqual(sym_step["domain"], "viessmann")
        self.assertEqual(compose_step["selected_domain"], "viessmann")
        self.assertEqual(compose_step["effective_domain"], "viessmann")

    def test_symbolic_query_falls_back_to_symbolic_direct_when_llm_says_insufficient(self):
        with patch(
            "backend.api.solve_auto.answer_with_context_detailed",
            return_value={
                "answer": "Insufficient context.",
                "trace": {
                    "configured_provider": "openai",
                    "configured_model": "gpt-5",
                    "llm_disabled": False,
                    "llm_attempted": True,
                    "llm_used": True,
                    "provider": "openai",
                    "model": "gpt-5",
                    "api": "chat_completions",
                    "path": "llm",
                    "passage_count": 1,
                },
            },
        ):
            response = self.client.post(
                "/solve_auto",
                json={
                    "query": "Name two compliance standards that apply to ProductA.",
                    "domain": "battery",
                    "session": "auto-test",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("EN 62133-2", body["answer"])
        self.assertIn("[sym:ProductA]", body["answer"])
        self.assertFalse(body["answer_trace"]["llm_used"])
        self.assertEqual(body["answer_trace"]["path"], "symbolic_direct")
        compose_step = next(step for step in body["steps"] if step["source"] == "COMPOSE")
        self.assertEqual(compose_step["included_passage_ids"], ["sym:ProductA"])


if __name__ == "__main__":
    unittest.main()
