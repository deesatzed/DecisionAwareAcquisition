"""Automated pytest gate for the deterministic `< 1s` Paper 3 v2.2.0 table & ledger verifier."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_paper3_v2_2_tables import verify_all_paper3_v2_2_ledgers


def test_paper3_v2_2_deterministic_ledger_verifier() -> None:
    res_json = REPO_ROOT / "results" / "paper3_v2_2_remediation_results.json"
    if not res_json.exists():
        pytest.skip("paper3_v2_2_remediation_results.json not yet generated.")
    t0 = time.perf_counter()
    summary = verify_all_paper3_v2_2_ledgers(res_json)
    elapsed = time.perf_counter() - t0
    assert summary["status"] == "PASS"
    assert summary["checks_verified"] >= 190
    assert elapsed < 1.5
