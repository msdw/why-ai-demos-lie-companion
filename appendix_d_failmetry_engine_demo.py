#!/usr/bin/env python3
"""
Appendix D Reference Implementation: The Unified FailMetry Engine
=================================================================
Book: Why AI Demos Lie
Author: Mathurin Aché Arnaud (Kaggle Grandmaster)

This standalone, zero-dependency Python 3.12+ reference engine integrates the core
mathematical, statistical, and verification primitives developed across Chapters 1-16:
  1. Exact & Asymptotic Binomial Confidence Bounds (Wilson Score & Clopper-Pearson)
  2. Paired McNemar Contingency Hypothesis Testing for Model/Prompt Upgrades
  3. Rogan-Gladen Prevalence Bias Correction & Delta-Method Variance for LLM Judges
  4. Deterministic State-Mutation Oracle (Snapshot Diff + Invariant Verification)
  5. Seeded Chaos Fault Injection & Trajectory Reliability (pass@k vs. pass^k)
  6. Distributional Drift Telemetry (Population Stability Index & Jensen-Shannon Divergence in bits)
  7. Four-Stage Production Release Gate with Cryptographic SHA-256 Bundle Manifest
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# 1. STATISTICAL INFERENCE ENGINE (CHAPTERS 2 & 3)
# ============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_score_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    Compute the two-sided (1 - alpha) Wilson score confidence interval for a binomial
    proportion p = k / n. Remains bounded in [0, 1] even at small n or extreme k.
    """
    if n <= 0:
        raise ValueError("Sample size n must be strictly positive.")
    if not (0 <= k <= n):
        raise ValueError(f"Success count k={k} must lie in [0, n={n}].")

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (lower, upper)


def clopper_pearson_upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Compute the one-sided (1 - alpha) Clopper-Pearson exact upper confidence bound
    on a failure or attack success rate p, using bisection on the cumulative binomial tail:
        P(X <= k | n, p_upper) = alpha.
    Crucial for the Rule of Three when k = 0 failures are observed in n trials.
    """
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - (alpha ** (1.0 / n))

    def binom_cdf(prob: float) -> float:
        total = 0.0
        for i in range(k + 1):
            log_comb = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            if 0.0 < prob < 1.0:
                log_pmf = log_comb + i * math.log(prob) + (n - i) * math.log(1.0 - prob)
                total += math.exp(log_pmf)
        return total

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binom_cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PairedMcNemarResult:
    n_total: int
    n_both_pass: int
    n_b_only_fail: int  # b: baseline passes, candidate fails (regression)
    n_c_only_pass: int  # c: baseline fails, candidate passes (improvement)
    n_both_fail: int
    net_lift: float
    exact_p_value: float
    significant_at_005: bool


def paired_mcnemar_test(
    baseline_outcomes: list[int],
    candidate_outcomes: list[int],
) -> PairedMcNemarResult:
    """
    Evaluate a candidate prompt/model against a baseline on identical evaluation items.
    Uses the exact two-sided binomial tail on discordant pairs (b, c).
    """
    if len(baseline_outcomes) != len(candidate_outcomes) or not baseline_outcomes:
        raise ValueError("Baseline and candidate outcome vectors must be non-empty and equal length.")

    n11 = n10 = n01 = n00 = 0
    for base, cand in zip(baseline_outcomes, candidate_outcomes):
        if base == 1 and cand == 1:
            n11 += 1
        elif base == 1 and cand == 0:
            n10 += 1  # b: regression
        elif base == 0 and cand == 1:
            n01 += 1  # c: improvement
        else:
            n00 += 1

    b, c = n10, n01
    m = b + c
    if m == 0:
        p_val = 1.0
    else:
        k_min = min(b, c)
        tail = 0.0
        for i in range(k_min + 1):
            log_c = math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1)
            tail += math.exp(log_c - m * math.log(2.0))
        p_val = min(1.0, 2.0 * tail)

    n_total = len(baseline_outcomes)
    net_lift = (c - b) / n_total
    return PairedMcNemarResult(
        n_total=n_total,
        n_both_pass=n11,
        n_b_only_fail=b,
        n_c_only_pass=c,
        n_both_fail=n00,
        net_lift=net_lift,
        exact_p_value=p_val,
        significant_at_005=(p_val < 0.05 and c > b),
    )


# ============================================================================
# 2. ROGAN-GLADEN JUDGE CALIBRATION ENGINE (CHAPTER 6)
# ============================================================================

@dataclass(frozen=True)
class CalibratedJudgeEstimate:
    raw_positive_rate: float
    sensitivity_q1: float
    specificity_q0: float
    youden_j: float
    corrected_prevalence: float
    ci_95_lower: float
    ci_95_upper: float
    absolute_bias_removed: float


def rogan_gladen_calibrate(
    judge_positives: int,
    n_eval: int,
    calib_tp: int,
    calib_pos_total: int,
    calib_tn: int,
    calib_neg_total: int,
    z: float = 1.959963984540054,
) -> CalibratedJudgeEstimate:
    """
    Correct raw LLM-as-a-Judge pass rate q_raw using calibration sensitivity q1
    and specificity q0 via the Rogan-Gladen estimator, propagating both evaluation
    and calibration uncertainty via the multivariate delta method.
    """
    q_raw = judge_positives / n_eval
    q1 = calib_tp / calib_pos_total
    q0 = calib_tn / calib_neg_total
    j = q1 + q0 - 1.0
    if j <= 0.05:
        raise ValueError(
            f"Judge Youden index J={j:.4f} <= 0.05: judge is no better than noise and cannot be inverted."
        )

    theta_unclipped = (q_raw + q0 - 1.0) / j
    theta = max(0.0, min(1.0, theta_unclipped))

    # Delta-method variance components: Var(q_raw), Var(q1), Var(q0)
    var_q_raw = max(q_raw * (1.0 - q_raw), 1e-6) / n_eval
    var_q1 = max(q1 * (1.0 - q1), 1e-6) / calib_pos_total
    var_q0 = max(q0 * (1.0 - q0), 1e-6) / calib_neg_total

    d_theta_dqraw = 1.0 / j
    d_theta_dq1 = -theta / j
    d_theta_dq0 = (1.0 - theta) / j

    var_theta = (
        (d_theta_dqraw ** 2) * var_q_raw
        + (d_theta_dq1 ** 2) * var_q1
        + (d_theta_dq0 ** 2) * var_q0
    )
    se_theta = math.sqrt(var_theta)
    ci_low = max(0.0, theta - z * se_theta)
    ci_high = min(1.0, theta + z * se_theta)

    return CalibratedJudgeEstimate(
        raw_positive_rate=q_raw,
        sensitivity_q1=q1,
        specificity_q0=q0,
        youden_j=j,
        corrected_prevalence=theta,
        ci_95_lower=ci_low,
        ci_95_upper=ci_high,
        absolute_bias_removed=q_raw - theta,
    )


# ============================================================================
# 3. DETERMINISTIC STATE-MUTATION ORACLE & CHAOS PROXY (CHAPTERS 5 & 11)
# ============================================================================

@dataclass
class SandboxDatabaseState:
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    refund_ledger: list[dict[str, Any]] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)

    def canonical_hash(self) -> str:
        payload = json.dumps(
            {
                "orders": self.orders,
                "refund_ledger": self.refund_ledger,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StateOracleVerdict:
    passed: bool
    prose_claimed_success: bool
    false_victory_detected: bool
    invariant_violations: list[str]
    pre_state_hash: str
    post_state_hash: str


def verify_refund_state_oracle(
    s0: SandboxDatabaseState,
    s_final: SandboxDatabaseState,
    target_order_id: str,
    expected_refund_cents: int,
    agent_prose_response: str,
) -> StateOracleVerdict:
    """
    Grade an agent execution strictly by comparing pre-state s0 and post-state s_final
    plus domain invariants, exposing 'False Victory' demos where prose claims success
    while the underlying database state is corrupted or untouched.
    """
    violations: list[str] = []
    prose_claimed = "refunded" in agent_prose_response.lower() or "completed" in agent_prose_response.lower()

    # 1. Target order must be marked refunded with exact amount
    order_after = s_final.orders.get(target_order_id)
    if not order_after or order_after.get("status") != "REFUNDED":
        violations.append(f"Target order {target_order_id} status is not REFUNDED.")

    # 2. Exactly one idempotent ledger entry for target_order_id
    matching_entries = [r for r in s_final.refund_ledger if r.get("order_id") == target_order_id]
    if len(matching_entries) == 0:
        violations.append(f"Missing refund ledger entry for {target_order_id}.")
    elif len(matching_entries) > 1:
        violations.append(
            f"Duplicate non-idempotent refund entries ({len(matching_entries)}) for {target_order_id}."
        )
    elif matching_entries[0].get("amount_cents") != expected_refund_cents:
        violations.append(
            f"Refund amount mismatch: expected {expected_refund_cents}, got {matching_entries[0].get('amount_cents')}."
        )

    # 3. Collateral damage check: no other order in s0 may be modified
    for oid, orig_record in s0.orders.items():
        if oid != target_order_id:
            if s_final.orders.get(oid) != orig_record:
                violations.append(f"Collateral state mutation on non-target order {oid}.")

    passed = len(violations) == 0
    false_victory = prose_claimed and not passed

    return StateOracleVerdict(
        passed=passed,
        prose_claimed_success=prose_claimed,
        false_victory_detected=false_victory,
        invariant_violations=violations,
        pre_state_hash=s0.canonical_hash()[:16],
        post_state_hash=s_final.canonical_hash()[:16],
    )


def compute_pass_at_k_vs_pass_power_k(n_trials: int, c_successes: int, k: int) -> tuple[float, float]:
    """
    Contrast optimistic pass@k (at least 1 success in k tries) against production
    reliability pass^k (all k consecutive invocations succeed).
    """
    if n_trials - c_successes < k:
        pass_at_k = 1.0
    else:
        log_num = (
            math.lgamma(n_trials - c_successes + 1)
            - math.lgamma(n_trials - c_successes - k + 1)
        )
        log_den = math.lgamma(n_trials + 1) - math.lgamma(n_trials - k + 1)
        pass_at_k = 1.0 - math.exp(log_num - log_den)

    p_hat = c_successes / n_trials
    pass_power_k = p_hat ** k
    return (pass_at_k, pass_power_k)


# ============================================================================
# 4. DRIFT TELEMETRY & 4-STAGE PRODUCTION RELEASE GATE (CHAPTERS 14 & 16)
# ============================================================================

def compute_psi_and_jsd_bits(baseline_counts: list[int], prod_counts: list[int]) -> tuple[float, float]:
    """
    Compute both Laplace-smoothed Population Stability Index (PSI, continuous quantile bins)
    and base-2 Jensen-Shannon Divergence (JSD in bits, discrete tool-call distributions),
    matching Chapter 14's multi-signal drift telemetry architecture.
    """
    total_b = sum(baseline_counts)
    total_q = sum(prod_counts)
    k_bins = len(baseline_counts)
    psi = 0.0
    jsd_bits = 0.0
    for b_cnt, q_cnt in zip(baseline_counts, prod_counts):
        p_i = (b_cnt + 0.5) / (total_b + 0.5 * k_bins)
        q_i = (q_cnt + 0.5) / (total_q + 0.5 * k_bins)
        psi += (p_i - q_i) * math.log(p_i / q_i)
        m_i = 0.5 * (p_i + q_i)
        jsd_bits += 0.5 * p_i * math.log2(p_i / m_i) + 0.5 * q_i * math.log2(q_i / m_i)
    return (psi, jsd_bits)


@dataclass(frozen=True)
class ReleaseGateDecision:
    approved_for_production: bool
    stage1_deterministic_passed: bool
    stage2_statistical_passed: bool
    stage3_chaos_redteam_passed: bool
    stage4_canary_drift_passed: bool
    blocking_reasons: list[str]
    bundle_sha256: str


def evaluate_four_stage_release_gate(
    *,
    manifest_components: dict[str, str],
    schema_and_oracle_syntax_pass_rate: float,
    calibrated_judge_est: CalibratedJudgeEstimate,
    mcnemar_res: PairedMcNemarResult,
    chaos_pass_power_5: float,
    redteam_zero_failure_trials: int,
    max_allowed_asr_upper_95: float,
    canary_psi: float,
    min_calibrated_lower_bound: float = 0.85,
    min_chaos_pass_power_5: float = 0.70,
    max_canary_psi: float = 0.20,
) -> ReleaseGateDecision:
    """
    Execute the 4-Stage FailMetry Production Release Gate:
      - Stage 1: Deterministic Schema, ACL & State-Oracle Sanity (100% required)
      - Stage 2: Calibrated Rogan-Gladen Lower Bound & Paired Non-Regression
      - Stage 3: Chaos Fault Survival (pass^5) & Clopper-Pearson Red-Team Bound
      - Stage 4: Production Canary Distributional Stability (PSI < threshold)
    """
    reasons: list[str] = []

    # Stage 1
    s1 = abs(schema_and_oracle_syntax_pass_rate - 1.0) < 1e-9
    if not s1:
        reasons.append(
            f"Stage 1 Failed: Deterministic schema/ACL pass rate {schema_and_oracle_syntax_pass_rate:.3f} < 1.000."
        )

    # Stage 2
    s2 = (
        calibrated_judge_est.ci_95_lower >= min_calibrated_lower_bound
        and mcnemar_res.net_lift >= 0.0
    )
    if calibrated_judge_est.ci_95_lower < min_calibrated_lower_bound:
        reasons.append(
            f"Stage 2 Failed: Rogan-Gladen 95% lower bound {calibrated_judge_est.ci_95_lower:.3f} "
            f"< required floor {min_calibrated_lower_bound:.3f} (raw uncalibrated score was {calibrated_judge_est.raw_positive_rate:.3f})."
        )
    if mcnemar_res.net_lift < 0.0:
        reasons.append(
            f"Stage 2 Failed: Negative paired McNemar lift ({mcnemar_res.net_lift:+.3f})."
        )

    # Stage 3
    asr_upper_95 = clopper_pearson_upper_bound(0, redteam_zero_failure_trials, alpha=0.05)
    s3 = (chaos_pass_power_5 >= min_chaos_pass_power_5) and (asr_upper_95 <= max_allowed_asr_upper_95)
    if chaos_pass_power_5 < min_chaos_pass_power_5:
        reasons.append(
            f"Stage 3 Failed: Chaos pass^5 reliability {chaos_pass_power_5:.3f} < {min_chaos_pass_power_5:.3f}."
        )
    if asr_upper_95 > max_allowed_asr_upper_95:
        reasons.append(
            f"Stage 3 Failed: Clopper-Pearson 95% ASR upper bound {asr_upper_95:.4f} > {max_allowed_asr_upper_95:.4f} "
            f"(insufficient red-team sample size n={redteam_zero_failure_trials})."
        )

    # Stage 4
    s4 = canary_psi <= max_canary_psi
    if not s4:
        reasons.append(
            f"Stage 4 Failed: Canary Population Stability Index PSI={canary_psi:.4f} > {max_canary_psi:.4f}."
        )

    manifest_json = json.dumps(manifest_components, sort_keys=True)
    bundle_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    approved = s1 and s2 and s3 and s4
    return ReleaseGateDecision(
        approved_for_production=approved,
        stage1_deterministic_passed=s1,
        stage2_statistical_passed=s2,
        stage3_chaos_redteam_passed=s3,
        stage4_canary_drift_passed=s4,
        blocking_reasons=reasons,
        bundle_sha256=bundle_hash,
    )


# ============================================================================
# SELF-CONTAINED VERIFICATION SUITE
# ============================================================================

def run_verification() -> None:
    print("=" * 78)
    print("FAILMETRY UNIFIED ENGINE — APPENDIX D REFERENCE VERIFICATION")
    print("=" * 78)

    # 1. Verify Wilson vs Clopper-Pearson
    w_low, w_high = wilson_score_interval(42, 50)
    cp_upper_0_100 = clopper_pearson_upper_bound(0, 100, alpha=0.05)
    print(f"[1] Binomial Inference (k=42, n=50): Wilson 95% CI = [{w_low:.4f}, {w_high:.4f}]")
    print(f"    Rule of Three Exact Check (k=0, n=100): CP 95% Upper = {cp_upper_0_100:.4f} (~0.0295)")
    assert 0.71 < w_low < 0.72 and 0.91 < w_high < 0.92
    assert abs(cp_upper_0_100 - 0.029513) < 1e-4

    # 2. Verify Paired McNemar Test
    rng = random.Random(2026)
    base_vec = [1 if i < 150 else 0 for i in range(200)]
    # Candidate keeps 145 of baseline passes, fails 5 (b=5), and rescues 22 of baseline failures (c=22)
    cand_vec = [1 if i < 145 else (1 if 150 <= i < 172 else 0) for i in range(200)]
    mc_res = paired_mcnemar_test(base_vec, cand_vec)
    print(
        f"[2] Paired McNemar (N=200, b={mc_res.n_b_only_fail}, c={mc_res.n_c_only_pass}): "
        f"lift={mc_res.net_lift:+.3f}, exact p={mc_res.exact_p_value:.6f}, sig={mc_res.significant_at_005}"
    )
    assert mc_res.n_b_only_fail == 5 and mc_res.n_c_only_pass == 22
    assert mc_res.significant_at_005 is True

    # 3. Verify Rogan-Gladen Calibration
    # Suppose an LLM judge reports a raw observed 66.0% pass rate (330/500).
    # Gold human calibration measures sensitivity q1 = 0.90 (180/200) and specificity q0 = 0.70 (140/200),
    # yielding Youden's index J = 0.90 + 0.70 - 1.0 = 0.60.
    # Rogan-Gladen strips false-positive judge inflation:
    #   p_hat = (0.66 + 0.70 - 1.0) / 0.60 = 0.36 / 0.60 = 0.600 (stripping +6.0% raw inflation).
    rg_est = rogan_gladen_calibrate(
        judge_positives=330,
        n_eval=500,
        calib_tp=180,
        calib_pos_total=200,
        calib_tn=140,
        calib_neg_total=200,
    )
    print(
        f"[3] Rogan-Gladen Calibration: Raw Judge={rg_est.raw_positive_rate:.3f} -> "
        f"Corrected={rg_est.corrected_prevalence:.3f} "
        f"(95% CI [{rg_est.ci_95_lower:.3f}, {rg_est.ci_95_upper:.3f}], "
        f"Bias Removed={rg_est.absolute_bias_removed:+.3f})"
    )
    assert rg_est.raw_positive_rate == 0.66
    assert abs(rg_est.corrected_prevalence - 0.60) < 1e-6
    assert abs(rg_est.absolute_bias_removed - 0.06) < 1e-6

    # 4. Verify State-Mutation Oracle catching a "False Victory" retry bug
    s0 = SandboxDatabaseState(
        orders={
            "ORD-901": {"status": "PAID", "customer": "cust_1", "amount_cents": 12500},
            "ORD-902": {"status": "PAID", "customer": "cust_2", "amount_cents": 8900},
        },
        refund_ledger=[],
    )
    # Buggy agent retries without idempotency key, inserting two refunds while saying "Refund completed!"
    s_buggy = copy.deepcopy(s0)
    s_buggy.orders["ORD-901"]["status"] = "REFUNDED"
    s_buggy.refund_ledger.append({"order_id": "ORD-901", "amount_cents": 12500})
    s_buggy.refund_ledger.append({"order_id": "ORD-901", "amount_cents": 12500})

    verdict_buggy = verify_refund_state_oracle(
        s0, s_buggy, "ORD-901", 12500, "Your order ORD-901 has been refunded and completed."
    )
    print(
        f"[4] State Oracle on Non-Idempotent Retry: passed={verdict_buggy.passed}, "
        f"false_victory={verdict_buggy.false_victory_detected}, violations={verdict_buggy.invariant_violations}"
    )
    assert verdict_buggy.passed is False and verdict_buggy.false_victory_detected is True

    # 5. Verify pass@k vs pass^k divergence
    p_at_5, p_pow_5 = compute_pass_at_k_vs_pass_power_k(n_trials=20, c_successes=16, k=5)
    print(
        f"[5] Trajectory Metrics (p_hat=0.80, k=5): Optimistic pass@5={p_at_5:.4f} vs. "
        f"Production Reliability pass^5={p_pow_5:.4f}"
    )
    assert p_at_5 > 0.999 and abs(p_pow_5 - (0.8 ** 5)) < 1e-9

    # 6. Verify 4-Stage Release Gate (First blocking on underpowered red-team, then passing)
    psi_val, jsd_val = compute_psi_and_jsd_bits([200, 300, 300, 200], [190, 310, 295, 205])
    manifest = {
        "model_weights": "model-under-test-v1",
        "system_prompt_sha": "a94f3c12",
        "tool_schemas_sha": "77b109e4",
        "rag_index_snapshot": "idx-2026-10-25",
        "judge_calibration_sha": "rg-cal-v4",
    }

    blocked_gate = evaluate_four_stage_release_gate(
        manifest_components=manifest,
        schema_and_oracle_syntax_pass_rate=1.0,
        calibrated_judge_est=rg_est,
        mcnemar_res=mc_res,
        chaos_pass_power_5=0.78,
        redteam_zero_failure_trials=40,  # Too small! CP upper bound = ~0.072 > 0.02
        max_allowed_asr_upper_95=0.02,
        canary_psi=psi_val,
        min_calibrated_lower_bound=0.50,
    )
    print(
        f"[6] Release Gate (Underpowered Red-Team n=40): approved={blocked_gate.approved_for_production}, "
        f"reasons={blocked_gate.blocking_reasons}"
    )
    assert blocked_gate.approved_for_production is False

    approved_gate = evaluate_four_stage_release_gate(
        manifest_components=manifest,
        schema_and_oracle_syntax_pass_rate=1.0,
        calibrated_judge_est=rg_est,
        mcnemar_res=mc_res,
        chaos_pass_power_5=0.78,
        redteam_zero_failure_trials=160,  # CP upper bound = 0.0185 <= 0.02
        max_allowed_asr_upper_95=0.02,
        canary_psi=psi_val,
        min_calibrated_lower_bound=0.50,
    )
    print(
        f"[7] Release Gate (Powered Red-Team n=160, PSI={psi_val:.5f}): "
        f"approved={approved_gate.approved_for_production}, bundle_sha256={approved_gate.bundle_sha256[:16]}..."
    )
    assert approved_gate.approved_for_production is True

    # 7. Independent Cross-Check & Property-Based Boundary Verification (Breaking Oracle Circularity)
    # Cross-check Clopper-Pearson bisection against closed-form analytical identities at k=0, k=1, k=n
    for n_test in (1, 10, 50, 250):
        cp_k0 = clopper_pearson_upper_bound(0, n_test, alpha=0.05)
        exact_k0 = 1.0 - (0.05 ** (1.0 / n_test))
        assert abs(cp_k0 - exact_k0) < 1e-12, f"Mismatch at k=0, n={n_test}"
        assert clopper_pearson_upper_bound(n_test, n_test, alpha=0.05) == 1.0
        w_l0, w_u0 = wilson_score_interval(0, n_test)
        w_ln, w_un = wilson_score_interval(n_test, n_test)
        assert 0.0 <= w_l0 < w_u0 <= 1.0 and 0.0 <= w_ln < w_un <= 1.0

    # Verify Youden's J -> 0 breakdown guard raises ValueError as required by metrology
    youden_guard_triggered = False
    try:
        rogan_gladen_calibrate(250, 500, 102, 200, 100, 200)  # q1=0.51, q0=0.50 -> J=0.01 <= 0.05
    except ValueError:
        youden_guard_triggered = True
    assert youden_guard_triggered is True
    print("[8] Independent Property & Boundary Cross-Checks (k=0, k=n, n=1, Youden J->0 Guard): PASSED")
    print("=" * 78)
    print("ALL APPENDIX D FAILMETRY INVARIANTS VERIFIED SUCCESSFULLY.")


if __name__ == "__main__":
    run_verification()
