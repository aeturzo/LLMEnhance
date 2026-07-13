import unittest

from backend.services.carbon_calculation_service import calculate_carbon_footprint


def _complete_override_scenario():
    return {
        "requested_stages": ["raw_materials", "transportation", "use_phase", "end_of_life"],
        "total_product_mass_kg": 10.0,
        "raw_materials": [
            {
                "material_key": "plastics_generic",
                "mass_kg": 4.5,
                "factor_value_kg_co2e_per_kg": 2.7,
                "source_ref": "scenario:plastics",
            },
            {
                "material_key": "steel_generic",
                "mass_kg": 3.5,
                "factor_value_kg_co2e_per_kg": 2.1,
                "source_ref": "scenario:steel",
            },
            {
                "material_key": "electronics_generic",
                "mass_kg": 1.2,
                "factor_value_kg_co2e_per_kg": 8.4,
                "source_ref": "scenario:electronics",
            },
            {
                "material_key": "elastomers_generic",
                "mass_kg": 0.4,
                "factor_value_kg_co2e_per_kg": 3.2,
                "source_ref": "scenario:elastomers",
            },
        ],
        "transport_legs": [
            {
                "leg_id": "factory_to_market",
                "mode": "ship",
                "distance_km": 1000.0,
                "mass_kg": 10.0,
                "factor_value_kg_co2e_per_ton_km": 0.009,
                "source_ref": "scenario:ship",
            },
            {
                "leg_id": "regional_distribution",
                "mode": "truck",
                "distance_km": 200.0,
                "mass_kg": 10.0,
                "factor_value_kg_co2e_per_ton_km": 0.18,
                "source_ref": "scenario:truck",
            },
        ],
        "use_phase": {
            "lifetime_energy_kwh": 20.0,
            "electricity_factor_value_kg_co2e_per_kwh": 0.2,
            "source_ref": "scenario:electricity",
        },
        "end_of_life": {
            "mass_kg": 10.0,
            "recycling_rate_pct": 80.0,
            "incineration_rate_pct": 15.0,
            "landfill_rate_pct": 5.0,
            "route_factors": {
                "recycling": {
                    "factor_value_kg_co2e_per_kg": 0.1,
                    "source_ref": "scenario:eol_recycling",
                },
                "incineration": {
                    "factor_value_kg_co2e_per_kg": 0.3,
                    "source_ref": "scenario:eol_incineration",
                },
                "landfill": {
                    "factor_value_kg_co2e_per_kg": 0.05,
                    "source_ref": "scenario:eol_landfill",
                },
            },
        },
    }


class CarbonCalculationServiceTests(unittest.TestCase):
    def test_complete_override_scenario_returns_total_and_breakdown(self):
        result = calculate_carbon_footprint("lexmark_mx431adn", _complete_override_scenario())

        self.assertEqual(result.status, "complete")
        self.assertAlmostEqual(result.total_kg_co2e, 36.585, places=6)
        self.assertAlmostEqual(result.partial_total_kg_co2e, 36.585, places=6)
        self.assertEqual(result.missing_inputs, [])

        self.assertEqual(set(result.stage_results.keys()), {"raw_materials", "transportation", "use_phase", "end_of_life"})
        self.assertEqual(result.stage_results["raw_materials"].status, "complete")
        self.assertEqual(result.stage_results["transportation"].status, "complete")
        self.assertEqual(result.stage_results["use_phase"].status, "complete")
        self.assertEqual(result.stage_results["end_of_life"].status, "complete")

        self.assertAlmostEqual(result.stage_results["raw_materials"].total_kg_co2e, 30.86, places=6)
        self.assertAlmostEqual(result.stage_results["transportation"].total_kg_co2e, 0.45, places=6)
        self.assertAlmostEqual(result.stage_results["use_phase"].total_kg_co2e, 4.0, places=6)
        self.assertAlmostEqual(result.stage_results["end_of_life"].total_kg_co2e, 1.275, places=6)

        self.assertEqual(result.recyclability.status, "complete")
        self.assertAlmostEqual(result.recyclability.recyclability_pct, 80.0, places=6)
        self.assertAlmostEqual(result.recyclability.recoverable_mass_kg, 8.0, places=6)
        self.assertAlmostEqual(result.recyclability.incineration_mass_kg, 1.5, places=6)
        self.assertAlmostEqual(result.recyclability.landfill_mass_kg, 0.5, places=6)

    def test_official_product_without_overrides_returns_partial_and_missing_inputs(self):
        result = calculate_carbon_footprint("lexmark_mx431adn", {})

        self.assertEqual(result.product_id, "lexmark_mx431adn")
        self.assertEqual(result.status, "partial")
        self.assertIsNone(result.total_kg_co2e)
        self.assertGreater(len(result.missing_inputs), 0)
        self.assertEqual(result.requested_stages, ["raw_materials", "transportation", "use_phase", "end_of_life"])

        self.assertIn("raw_materials", result.stage_results)
        self.assertIn("transportation", result.stage_results)
        self.assertIn("use_phase", result.stage_results)
        self.assertIn("end_of_life", result.stage_results)
        self.assertEqual(result.recyclability.status, "missing")
        self.assertIn("Recycling split is not available.", result.recyclability.notes)

    def test_estimation_profile_produces_complete_hybrid_estimate(self):
        result = calculate_carbon_footprint("lexmark_mx431adn", {"use_bootstrap_estimates": True})

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.quality_status, "hybrid_estimate")
        self.assertIsNotNone(result.total_kg_co2e)
        self.assertGreater(result.total_kg_co2e, 0.0)
        self.assertIsNotNone(result.uncertainty_pct)
        self.assertGreater(result.uncertainty_pct, 0.0)
        self.assertEqual(result.missing_inputs, [])

        provenance_fields = {item.field_name for item in result.provenance}
        self.assertIn("total_product_mass_kg", provenance_fields)
        self.assertIn("raw_material_mix", provenance_fields)
        self.assertIn("use_phase_profile", provenance_fields)
        self.assertIn("transport_route", provenance_fields)
        self.assertIn("end_of_life_split", provenance_fields)


if __name__ == "__main__":
    unittest.main()
