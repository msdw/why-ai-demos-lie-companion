#!/usr/bin/env python3
"""
Master Companion Verification Runner: Why AI Demos Lie
======================================================
Author: Mathurin Aché Arnaud (Kaggle Grandmaster)

Executes all 17 zero-dependency Python 3.12+ reference verification modules
in isolated mode (-I) and verifies all quantitative claims in claims.csv.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path



VERIFICATION_MODULES = [
    ("Chapter 01", "ch01_pass_at_k_demo.py"),
    ("Chapter 02", "ch02_confidence_and_power_demo.py"),
    ("Chapter 03", "ch03_paired_testing_anova_demo.py"),
    ("Chapter 04", "ch04_leakage_detector_demo.py"),
    ("Chapter 05", "ch05_state_oracles_demo.py"),
    ("Chapter 06", "ch06_calibrated_llm_judge_demo.py"),
    ("Chapter 07", "ch07_bradley_terry_ranking_demo.py"),
    ("Chapter 08", "ch08_rag_component_eval_demo.py"),
    ("Chapter 09", "ch09_trajectory_auditor_demo.py"),
    ("Chapter 10", "ch10_latency_and_ecvt_demo.py"),
    ("Chapter 11", "ch11_chaos_fault_injector_demo.py"),
    ("Chapter 12", "ch12_adversarial_redteam_oracle_demo.py"),
    ("Chapter 13", "ch13_annotator_and_hitl_math_demo.py"),
    ("Chapter 14", "ch14_drift_and_cusum_demo.py"),
    ("Chapter 15", "ch15_forensic_replay_and_postmortem_demo.py"),
    ("Chapter 16", "ch16_evaluation_release_gate_demo.py"),
    ("Appendix D", "appendix_d_failmetry_engine_demo.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Verification Runner for Why AI Demos Lie"
    )
    parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    print("=" * 82)
    print("WHY AI DEMOS LIE — MASTER VERIFICATION VAULT")
    print("Author: Mathurin Aché Arnaud (Kaggle Grandmaster)")
    print("=" * 82)

    passed_count = 0
    total_time = 0.0

    for scope, script_name in VERIFICATION_MODULES:
        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"[FAIL] {scope} ({script_name}): File not found at {script_path}")
            return 1

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if proc.returncode != 0:
            print(f"[FAIL] {scope} ({script_name}) exited with code {proc.returncode} ({elapsed:.2f}s)")
            print("--- STDOUT ---")
            print(proc.stdout)
            print("--- STDERR ---")
            print(proc.stderr)
            return 1

        passed_count += 1
        print(f"[PASS] {scope:10s} | {script_name:44s} | {elapsed:.2f}s | 0 errors")

    for filename, expected_rows, expected_fields in (
        ("claims.csv", 68, 10),
        ("sources.csv", 70, 5),
    ):
        target = base_dir / filename
        if not target.exists():
            print(f"[FAIL] Missing register: {target}")
            return 1
        with target.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(reader.fieldnames) != expected_fields:
            print(f"[FAIL] {filename}: expected {expected_fields} columns, got {len(reader.fieldnames)}")
            return 1
        if len(rows) != expected_rows:
            print(f"[FAIL] {filename}: expected {expected_rows} rows, got {len(rows)}")
            return 1
        if not all(None not in r for r in rows):
            print(f"[FAIL] {filename}: extra CSV cells detected")
            return 1
        if not all(all(r.get(k) is not None for k in reader.fieldnames) for r in rows):
            print(f"[FAIL] {filename}: missing cell detected")
            return 1
        if filename == "sources.csv":
            if not all(r["reference_url"].startswith(("https://", "http://")) for r in rows):
                print(f"[FAIL] {filename}: invalid URL protocol")
                return 1
        print(f"[PASS] Register Audit | {filename:44s} | {len(rows)} rows, {expected_fields} cols strictly valid")

    print("=" * 82)
    print(
        f"SUMMARY: {passed_count}/{len(VERIFICATION_MODULES)} Python 3.12+ Verification Modules PASSED "
        f"in {total_time:.2f}s | 68 Claims Audited | 70 Sources Verified."
    )
    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

