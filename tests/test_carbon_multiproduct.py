"""
Multi-product carbon-subsystem regression tests.

Verifies that the calculator produces totals within publication-quality tolerance
(±5% of manufacturer-declared lifecycle GWP) for each new product category
introduced in the carbon-subsystem expansion (paper revision).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.carbon_calculation_service import CarbonCalculationService  # noqa: E402


# (product_id, manufacturer_declared_total_kg_co2e, tolerance_pct)
PRODUCT_CASES = [
    ("apple_iphone15_pro_128gb", 66.0, 5.0),
    ("fairphone_4", 43.0, 5.0),
    ("dell_xps14_da14260", 311.0, 5.0),
    ("daikin_altherma_m_hw_260l", 7420.0, 5.0),
]


@pytest.fixture(scope="module")
def carbon_service() -> CarbonCalculationService:
    return CarbonCalculationService()


@pytest.mark.parametrize("product_id,target_kg,tol_pct", PRODUCT_CASES)
def test_product_total_within_tolerance(
    carbon_service: CarbonCalculationService,
    product_id: str,
    target_kg: float,
    tol_pct: float,
) -> None:
    result = carbon_service.calculate(product_id)
    assert result.status == "complete", f"{product_id}: status={result.status} (expected complete)"
    assert result.total_kg_co2e is not None
    err_pct = abs(result.total_kg_co2e - target_kg) / target_kg * 100.0
    assert err_pct <= tol_pct, (
        f"{product_id}: total={result.total_kg_co2e:.2f} kg CO2e differs from declared "
        f"{target_kg:.2f} kg by {err_pct:.2f}% (tolerance {tol_pct:.2f}%)"
    )


def test_generic_bev_pack_status_complete(carbon_service: CarbonCalculationService) -> None:
    """EV battery has no single manufacturer-declared total, so we only require
    that the calculator returns a complete result with sane stage totals."""
    result = carbon_service.calculate("generic_bev_pack_60kwh")
    assert result.status == "complete"
    assert result.total_kg_co2e is not None
    # cradle-to-gate production should be ~5400 kg (60 kWh × 90 kg/kWh)
    rm = result.stage_results["raw_materials"].total_kg_co2e
    assert 5000 <= rm <= 6000, f"raw_materials={rm} expected ~5400"
    # use_phase should dominate cradle-to-grave
    up = result.stage_results["use_phase"].total_kg_co2e
    assert up > rm, "use_phase should exceed raw_materials for an EV battery"


def test_lexmark_regression_preserved(carbon_service: CarbonCalculationService) -> None:
    """The Lexmark printer was the original carbon-subsystem demo; make sure the
    new product profiles + calculator fix didn't regress it."""
    result = carbon_service.calculate(
        "lexmark_mx431adn",
        {"use_bootstrap_estimates": True},
    )
    assert result.status == "complete"
    # Paper reports ~100.485 kg CO2e for Lexmark MX431adn with estimate-assisted run.
    assert 95.0 <= result.total_kg_co2e <= 110.0, (
        f"lexmark total={result.total_kg_co2e}"
    )


def test_product_catalogue_visible() -> None:
    """The /carbon/products endpoint discovers products via glob — make sure all
    six product JSONs are physically present."""
    products_dir = REPO_ROOT / "backend" / "data" / "carbon" / "products"
    catalogue = {path.stem for path in products_dir.glob("*.json")}
    expected = {
        "lexmark_mx431adn",
        "apple_iphone15_pro_128gb",
        "fairphone_4",
        "dell_xps14_da14260",
        "daikin_altherma_m_hw_260l",
        "generic_bev_pack_60kwh",
    }
    missing = expected - catalogue
    assert not missing, f"Missing product profiles: {sorted(missing)}"
