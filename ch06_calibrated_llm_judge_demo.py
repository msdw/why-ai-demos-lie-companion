"""
Chapter 6: Quantifying LLM-as-a-Judge Biases & Rogan-Gladen Prevalence Calibration.
Why AI Demos Lie — Companion Verification Module (Python 3.12+, Zero External Dependencies).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PairwiseJudgeAudit:
    n_pairs: int
    win_rate_a_order_ab: float
    win_rate_a_order_ba: float
    position_bias_delta: float
    debiased_win_rate_a: float
    consistent_verdict_rate: float
    length_partial_corr: float
    self_preference_lift: float


@dataclass(frozen=True)
class RoganGladenEstimate:
    p_obs: float
    p_hat: float
    q1_sensitivity: float
    q0_specificity: float
    youden_j: float
    se_total: float
    ci_95_lower: float
    ci_95_upper: float
    bootstrap_ci_95: tuple[float, float]


class PositionDebiasedJudgeSwapper:
    """Audits pairwise LLM judge runs across (A,B) and (B,A) orderings and measures bias."""

    @staticmethod
    def partial_correlation(x: list[float], y: list[float], z: list[float]) -> float:
        """Computes r_{XY.Z}: correlation between X (length diff) and Y (judge score) controlling for Z (truth)."""
        def pearson(u: list[float], v: list[float]) -> float:
            mean_u, mean_v = statistics.fmean(u), statistics.fmean(v)
            num = sum((ui - mean_u) * (vi - mean_v) for ui, vi in zip(u, v))
            den = math.sqrt(sum((ui - mean_u) ** 2 for ui in u) * sum((vi - mean_v) ** 2 for vi in v))
            return num / den if den > 1e-12 else 0.0

        r_xy = pearson(x, y)
        r_xz = pearson(x, z)
        r_yz = pearson(y, z)
        denom = math.sqrt(max(1e-12, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
        return (r_xy - r_xz * r_yz) / denom

    @classmethod
    def audit_pairwise_swaps(
        cls,
        verdicts_ab: list[int],  # 1 if A wins in (A,B) order, 0 otherwise
        verdicts_ba: list[int],  # 1 if A wins in (B,A) order, 0 otherwise
        token_len_diffs: list[float],
        ground_truth_a: list[float],
        same_family_flags: list[int],
    ) -> PairwiseJudgeAudit:
        n = len(verdicts_ab)
        win_ab = sum(verdicts_ab) / n
        win_ba = sum(verdicts_ba) / n
        delta_pos = abs(win_ab - win_ba)
        debiased_a = 0.5 * (win_ab + win_ba)
        consistent = sum(1 for ab, ba in zip(verdicts_ab, verdicts_ba) if ab == ba) / n

        avg_judge_vote = [0.5 * (ab + ba) for ab, ba in zip(verdicts_ab, verdicts_ba)]
        r_len_partial = cls.partial_correlation(token_len_diffs, avg_judge_vote, ground_truth_a)

        # Self-preference lift controlling for ground truth quality
        fam_residuals = [
            vote - truth for vote, truth, fam in zip(avg_judge_vote, ground_truth_a, same_family_flags) if fam == 1
        ]
        other_residuals = [
            vote - truth for vote, truth, fam in zip(avg_judge_vote, ground_truth_a, same_family_flags) if fam == 0
        ]
        self_pref_lift = statistics.fmean(fam_residuals) - statistics.fmean(other_residuals)

        return PairwiseJudgeAudit(
            n_pairs=n,
            win_rate_a_order_ab=win_ab,
            win_rate_a_order_ba=win_ba,
            position_bias_delta=delta_pos,
            debiased_win_rate_a=debiased_a,
            consistent_verdict_rate=consistent,
            length_partial_corr=r_len_partial,
            self_preference_lift=self_pref_lift,
        )


class RoganGladenCalibrator:
    """Corrects raw LLM judge pass rates using sensitivity/specificity and multivariate Delta-method variance."""

    def __init__(self, tp_cal: int, n_pos_cal: int, tn_cal: int, n_neg_cal: int) -> None:
        if n_pos_cal <= 0 or n_neg_cal <= 0:
            raise ValueError("Calibration set must contain both positive and negative gold examples.")
        self.n1 = n_pos_cal
        self.n0 = n_neg_cal
        self.q1 = tp_cal / n_pos_cal
        self.q0 = tn_cal / n_neg_cal
        self.youden_j = self.q1 + self.q0 - 1.0
        if self.youden_j <= 0.15:
            raise ValueError(
                f"Judge failed metrological gate: Youden's J = {self.youden_j:.3f} <= 0.15. "
                "Calibration variance explodes near J -> 0."
            )

    def calibrate(
        self, k_obs_pass: int, n_eval: int, n_boot: int = 4000, seed: int = 42
    ) -> RoganGladenEstimate:
        p_obs = k_obs_pass / n_eval
        j = self.youden_j
        raw_p_hat = (p_obs + self.q0 - 1.0) / j
        p_hat = min(1.0, max(0.0, raw_p_hat))

        # Multivariate Delta-method variance combining test-set and calibration-set uncertainty
        var_p_obs = p_obs * (1.0 - p_obs) / n_eval
        var_q1 = self.q1 * (1.0 - self.q1) / self.n1
        var_q0 = self.q0 * (1.0 - self.q0) / self.n0

        # Partial derivatives of p_hat = (p_obs + q0 - 1) / (q1 + q0 - 1)
        # dp/dp_obs = 1/J, dp/dq1 = -p_hat/J, dp/dq0 = (1 - p_hat)/J
        var_total = (var_p_obs + (p_hat**2) * var_q1 + ((1.0 - p_hat) ** 2) * var_q0) / (j**2)
        se_total = math.sqrt(var_total)

        ci_low = max(0.0, p_hat - 1.95996 * se_total)
        ci_high = min(1.0, p_hat + 1.95996 * se_total)

        # Non-parametric binomial bootstrap cross-check
        rng = random.Random(seed)
        boot_vals: list[float] = []
        for _ in range(n_boot):
            # Binomial draws via exact Bernoulli sum
            b_obs = sum(1 for _ in range(n_eval) if rng.random() < p_obs) / n_eval
            b_q1 = sum(1 for _ in range(self.n1) if rng.random() < self.q1) / self.n1
            b_q0 = sum(1 for _ in range(self.n0) if rng.random() < self.q0) / self.n0
            b_j = max(0.05, b_q1 + b_q0 - 1.0)
            boot_vals.append(min(1.0, max(0.0, (b_obs + b_q0 - 1.0) / b_j)))
        boot_vals.sort()
        boot_ci = (boot_vals[int(0.025 * n_boot)], boot_vals[int(0.975 * n_boot)])

        return RoganGladenEstimate(
            p_obs=p_obs,
            p_hat=p_hat,
            q1_sensitivity=self.q1,
            q0_specificity=self.q0,
            youden_j=j,
            se_total=se_total,
            ci_95_lower=ci_low,
            ci_95_upper=ci_high,
            bootstrap_ci_95=boot_ci,
        )


def run_lab_6_1() -> None:
    rng = random.Random(202606)
    # Part A: Pairwise Position, Verbosity & Self-Preference Audit (N = 400 pairs)
    n_pairs = 400
    verdicts_ab, verdicts_ba, len_diffs, truth_a, same_fam = [], [], [], [], []
    for i in range(n_pairs):
        t_a = 1.0 if i % 2 == 0 else 0.0
        l_diff = rng.gauss(45.0 if t_a == 0.0 else 10.0, 60.0)
        fam = 1 if (i % 3 == 0) else 0
        # Latent judge logit with position primacy bias (+0.14 in AB), verbosity bias, and self-preference
        p_ab = min(0.98, max(0.02, 0.18 + 0.52 * t_a + 0.14 + 0.0012 * l_diff + 0.08 * fam))
        p_ba = min(0.98, max(0.02, 0.18 + 0.52 * t_a - 0.04 + 0.0012 * l_diff + 0.08 * fam))
        verdicts_ab.append(1 if rng.random() < p_ab else 0)
        verdicts_ba.append(1 if rng.random() < p_ba else 0)
        len_diffs.append(l_diff)
        truth_a.append(t_a)
        same_fam.append(fam)

    audit = PositionDebiasedJudgeSwapper.audit_pairwise_swaps(
        verdicts_ab, verdicts_ba, len_diffs, truth_a, same_fam
    )
    print("=== CHAPTER 6: PAIRWISE JUDGE BIAS AUDIT (N = 400 PAIRS) ===")
    print(f"Win Rate A (Order AB) : {audit.win_rate_a_order_ab:.3f}")
    print(f"Win Rate A (Order BA) : {audit.win_rate_a_order_ba:.3f}")
    print(f"Position Bias Delta   : {audit.position_bias_delta:.3f} ({audit.position_bias_delta*100:.1f}%)")
    print(f"Debiased Win Rate A   : {audit.debiased_win_rate_a:.3f}")
    print(f"Order Consistency Rate: {audit.consistent_verdict_rate:.3f}")
    print(f"Verbosity Partial r   : {audit.length_partial_corr:.3f}")
    print(f"Self-Preference Lift  : +{audit.self_preference_lift*100:.1f}%\n")

    # Part B: Rogan-Gladen Calibration Lab across p_true in [0.50, 0.95]
    # Gold calibration set: n1 = 200 positive items (TP = 176 -> q1 = 0.88), n0 = 200 negative items (TN = 152 -> q0 = 0.76)
    calibrator = RoganGladenCalibrator(tp_cal=176, n_pos_cal=200, tn_cal=152, n_neg_cal=200)
    print("=== LAB 6.1: ROGAN-GLADEN CALIBRATION SWEEP (q1=0.88, q0=0.76, J=0.64, N_eval=1000) ===")
    print(f"{'p_true':>8} | {'E[p_obs]':>8} | {'Raw Bias':>9} | {'p_hat':>8} | {'Delta 95% CI':>19} | {'Boot 95% CI':>19}")
    print("-" * 84)
    for p_true in [0.272, 0.500, 0.700, 0.800, 0.900, 0.950]:
        expected_p_obs = p_true * 0.88 + (1.0 - p_true) * (1.0 - 0.76)
        k_obs = round(expected_p_obs * 1000)
        est = calibrator.calibrate(k_obs_pass=k_obs, n_eval=1000, n_boot=3000, seed=100 + int(p_true * 100))
        raw_bias = est.p_obs - p_true
        print(
            f"{p_true:8.3f} | {est.p_obs:8.3f} | {raw_bias:+9.3f} | {est.p_hat:8.3f} | "
            f"[{est.ci_95_lower:.3f}, {est.ci_95_upper:.3f}] | "
            f"[{est.bootstrap_ci_95[0]:.3f}, {est.bootstrap_ci_95[1]:.3f}]"
        )


if __name__ == "__main__":
    run_lab_6_1()

    # Self-test assertions
    cal_check = RoganGladenCalibrator(tp_cal=176, n_pos_cal=200, tn_cal=152, n_neg_cal=200)
    assert abs(cal_check.youden_j - 0.64) < 1e-12
    est_check = cal_check.calibrate(k_obs_pass=414, n_eval=1000, n_boot=200, seed=42)
    assert abs(est_check.p_hat - 0.271875) < 1e-6
