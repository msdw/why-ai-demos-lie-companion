"""Chapter 2 Verification & Lab 2.1: Small-Sample Illusions, Exact Confidence Intervals, and Power.

Implements SmallSampleInferenceEngine:
- Wald, Wilson Score, Agresti-Coull, and exact Clopper-Pearson confidence intervals
- Regularized incomplete Beta I_x(a, b) via Lentz's continued-fraction algorithm + bisection root-finding
- Rule of Three upper bound (-ln(alpha)/n) vs. exact Binomial 1 - alpha^(1/n)
- Cohen's h arcsine effect size, exact minimum detectable effect (MDE), and required sample size n
- Reproducible 50,000-run coverage probability simulation (Lab 2.1)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ProportionIntervalSummary:
    """Comparison of binomial proportion confidence intervals for k successes in n trials."""

    n: int
    k: int
    p_hat: float
    wald: tuple[float, float]
    wilson: tuple[float, float]
    agresti_coull: tuple[float, float]
    clopper_pearson: tuple[float, float]


class SmallSampleInferenceEngine:
    """Zero-dependency exact and asymptotic binomial inference and power calculator."""

    @staticmethod
    def normal_quantile(p: float) -> float:
        """Acklam's rational approximation for standard normal inverse CDF Phi^{-1}(p)."""
        if not 0.0 < p < 1.0:
            raise ValueError("Probability p must lie strictly inside (0, 1).")
        a = (
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        )
        b = (
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        )
        c = (
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        )
        d = (
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        )
        p_low = 0.02425
        p_high = 1.0 - p_low
        if p < p_low:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            )
        if p <= p_high:
            q = p - 0.5
            r = q * q
            return (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
            )
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    @staticmethod
    def _betacf(a: float, b: float, x: float) -> float:
        """Evaluate continued fraction for incomplete Beta via modified Lentz's method."""
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-30
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    @classmethod
    def reg_incomplete_beta(cls, a: float, b: float, x: float) -> float:
        """Compute regularized incomplete Beta function I_x(a, b)."""
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        log_bt = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1.0 - x)
        )
        bt = math.exp(log_bt)
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * cls._betacf(a, b, x) / a
        return 1.0 - bt * cls._betacf(b, a, 1.0 - x) / b

    @classmethod
    def beta_quantile(cls, target_p: float, a: float, b: float) -> float:
        """Invert I_x(a, b) = target_p via 60-step bisection."""
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (low + high)
            if cls.reg_incomplete_beta(a, b, mid) < target_p:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    @classmethod
    def wald_interval(cls, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
        """Standard asymptotic Wald interval."""
        z = cls.normal_quantile(1.0 - alpha / 2.0)
        p_hat = k / n
        se = math.sqrt(p_hat * (1.0 - p_hat) / n)
        return max(0.0, p_hat - z * se), min(1.0, p_hat + z * se)

    @classmethod
    def wilson_interval(cls, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
        """Quadratic Wilson Score interval."""
        z = cls.normal_quantile(1.0 - alpha / 2.0)
        z2 = z * z
        p_hat = k / n
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2.0 * n)) / denom
        half_width = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
        return max(0.0, center - half_width), min(1.0, center + half_width)

    @classmethod
    def agresti_coull_interval(cls, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
        """Agresti-Coull ('add z^2/2 successes and z^2/2 failures') interval."""
        z = cls.normal_quantile(1.0 - alpha / 2.0)
        z2 = z * z
        n_tilde = n + z2
        p_tilde = (k + 0.5 * z2) / n_tilde
        se_tilde = math.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
        return max(0.0, p_tilde - z * se_tilde), min(1.0, p_tilde + z * se_tilde)

    @classmethod
    def clopper_pearson_interval(cls, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
        """Exact Clopper-Pearson interval via Beta quantiles."""
        low = 0.0 if k == 0 else cls.beta_quantile(alpha / 2.0, float(k), float(n - k + 1))
        high = 1.0 if k == n else cls.beta_quantile(1.0 - alpha / 2.0, float(k + 1), float(n - k))
        return low, high

    @staticmethod
    def zero_failure_upper_bound(n: int, alpha: float = 0.05) -> tuple[float, float]:
        """Return (Rule-of-Three approximation 3/n, exact one-sided bound 1 - alpha^(1/n))."""
        rule_of_three = 3.0 / n
        exact_bound = 1.0 - (alpha ** (1.0 / n))
        return rule_of_three, exact_bound

    @classmethod
    def cohens_h_sample_size(
        cls, p1: float, p2: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True
    ) -> tuple[float, int]:
        """Compute Cohen's h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)) and required per-arm n."""
        h = abs(2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2)))
        z_alpha = cls.normal_quantile(1.0 - (alpha / 2.0 if two_sided else alpha))
        z_beta = cls.normal_quantile(power)
        n_req = math.ceil(2.0 * ((z_alpha + z_beta) / h) ** 2)
        return h, n_req

    @classmethod
    def compare_intervals(cls, n: int, k: int, alpha: float = 0.05) -> ProportionIntervalSummary:
        """Compute all four confidence intervals for (n, k)."""
        return ProportionIntervalSummary(
            n=n,
            k=k,
            p_hat=k / n,
            wald=cls.wald_interval(n, k, alpha),
            wilson=cls.wilson_interval(n, k, alpha),
            agresti_coull=cls.agresti_coull_interval(n, k, alpha),
            clopper_pearson=cls.clopper_pearson_interval(n, k, alpha),
        )

    @classmethod
    def run_coverage_lab(
        cls,
        n: int,
        true_p: float,
        alpha: float = 0.05,
        num_runs: int = 50_000,
        seed: int = 202602,
    ) -> dict[str, float]:
        """Run Lab 2.1: 50,000 synthetic evaluations measuring empirical 95% CI coverage."""
        rng = random.Random(seed)
        # Pre-cache intervals for all k in 0..n for fast exact evaluation across 50,000 draws
        table = {k: cls.compare_intervals(n, k, alpha) for k in range(n + 1)}
        wald_cov = wilson_cov = ac_cov = cp_cov = 0
        zero_width_count = 0

        for _ in range(num_runs):
            k = sum(1 for _ in range(n) if rng.random() < true_p)
            row = table[k]
            if row.wald[0] <= true_p <= row.wald[1]:
                wald_cov += 1
            if row.wald[1] - row.wald[0] < 1e-12:
                zero_width_count += 1
            if row.wilson[0] <= true_p <= row.wilson[1]:
                wilson_cov += 1
            if row.agresti_coull[0] <= true_p <= row.agresti_coull[1]:
                ac_cov += 1
            if row.clopper_pearson[0] <= true_p <= row.clopper_pearson[1]:
                cp_cov += 1

        return {
            "wald_cov": wald_cov / num_runs,
            "wald_zero_width_rate": zero_width_count / num_runs,
            "wilson_cov": wilson_cov / num_runs,
            "agresti_coull_cov": ac_cov / num_runs,
            "clopper_pearson_cov": cp_cov / num_runs,
        }


if __name__ == "__main__":
    engine = SmallSampleInferenceEngine()
    print("=== WORKED EXAMPLES: INTERVALS & ZERO-FAILURE BOUNDS ===")
    for n_val, k_val in ((50, 48), (25, 25), (100, 99), (300, 300)):
        res = engine.compare_intervals(n_val, k_val, alpha=0.05)
        print(
            f"n={n_val}, k={k_val} (p_hat={res.p_hat:.4f}) -> "
            f"Wald=[{res.wald[0]:.4f}, {res.wald[1]:.4f}], "
            f"Wilson=[{res.wilson[0]:.4f}, {res.wilson[1]:.4f}], "
            f"AC=[{res.agresti_coull[0]:.4f}, {res.agresti_coull[1]:.4f}], "
            f"CP=[{res.clopper_pearson[0]:.4f}, {res.clopper_pearson[1]:.4f}]"
        )
    for n_zero in (25, 50, 100, 300, 1000):
        r3, exact_ub = engine.zero_failure_upper_bound(n_zero, alpha=0.05)
        print(f"Zero failures n={n_zero:4d}: RuleOf3={r3:.4%}, ExactOneSided95={exact_ub:.4%}")

    print("\n=== POWER & SAMPLE SIZE (COHEN'S h) ===")
    for p_base, p_new in ((0.85, 0.90), (0.90, 0.95), (0.95, 0.98), (0.80, 0.90)):
        h80, n80 = engine.cohens_h_sample_size(p_base, p_new, alpha=0.05, power=0.80)
        h90, n90 = engine.cohens_h_sample_size(p_base, p_new, alpha=0.05, power=0.90)
        print(f"p1={p_base:.2f} vs p2={p_new:.2f}: h={h80:.4f} | n_per_arm(80% power)={n80}, n_per_arm(90% power)={n90}")

    print("\n=== LAB 2.1: 50,000-RUN COVERAGE SIMULATION (Nominal 95% CI) ===")
    for n_sim in (25, 50, 100):
        for p_true in (0.90, 0.95, 0.98, 0.995):
            cov = engine.run_coverage_lab(n=n_sim, true_p=p_true, alpha=0.05, num_runs=50_000)
            print(
                f"n={n_sim:3d}, p={p_true:.3f} | "
                f"Wald={cov['wald_cov']:.4f} (ZeroWidth={cov['wald_zero_width_rate']:.4f}) | "
                f"Wilson={cov['wilson_cov']:.4f} | AC={cov['agresti_coull_cov']:.4f} | "
                f"CP={cov['clopper_pearson_cov']:.4f}"
            )

    # Self-test assertions
    w_res = engine.compare_intervals(25, 25, alpha=0.05)
    assert abs(w_res.wald[0] - 1.0) < 1e-9 and 0.86 < w_res.wilson[0] < 0.87 and abs(w_res.wilson[1] - 1.0) < 1e-9
    r3_30, ub_30 = engine.zero_failure_upper_bound(30, alpha=0.05)
    assert abs(ub_30 - (1.0 - 0.05 ** (1.0 / 30.0))) < 1e-12
