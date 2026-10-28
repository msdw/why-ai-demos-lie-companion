"""Chapter 1 Verification & Lab 1.1: The pass@k vs. pass^k Trap.

Implements TrajectoryReliabilityAnalyzer:
- Unbiased combinatorial pass@k estimator (Chen et al., 2021) via log-gamma
- Theoretical capability pass@k vs. dependability pass^k across m-step trajectories
- Beta-Binomial task heterogeneity mixture moments E[P^k] = B(alpha + k, beta) / B(alpha, beta)
- First-order Markov correlated step failure trajectories
- Reproducible 10,000-trial Monte Carlo simulation (Lab 1.1)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryReport:
    """Summary of capability vs. dependability metrics for a trajectory configuration."""

    step_success_p: float
    num_steps_m: int
    k_runs: int
    traj_p: float
    pass_at_k: float
    pass_hat_k: float
    illusion_gap: float
    beta_pass_at_k: float
    beta_pass_hat_k: float
    markov_traj_p: float
    markov_pass_hat_k: float


class TrajectoryReliabilityAnalyzer:
    """Analyzes multi-step agent trajectory reliability under i.i.d., Beta, and Markov regimes."""

    @staticmethod
    def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
        """Compute exact unbiased estimator 1 - C(n - c, k) / C(n, k) in log-space."""
        if n < k:
            raise ValueError(f"Total samples n={n} must be >= k={k}.")
        if c < 0 or c > n:
            raise ValueError(f"Correct samples c={c} must lie in [0, n={n}].")
        if n - c < k:
            return 1.0
        if c == 0:
            return 0.0
        # Compute log(C(n-c, k) / C(n, k)) = lgamma(n-c+1) - lgamma(n-c-k+1) - lgamma(n+1) + lgamma(n-k+1)
        log_ratio = (
            math.lgamma(n - c + 1)
            - math.lgamma(n - c - k + 1)
            - math.lgamma(n + 1)
            + math.lgamma(n - k + 1)
        )
        return 1.0 - math.exp(log_ratio)

    @staticmethod
    def trajectory_success_iid(step_p: float, num_steps: int) -> float:
        """Probability that all m independent steps succeed in a single trajectory."""
        if not 0.0 <= step_p <= 1.0:
            raise ValueError("step_p must be in [0, 1].")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        return step_p**num_steps

    @staticmethod
    def capability_pass_at_k(traj_p: float, k: int) -> float:
        """Probability that at least 1 of k independent runs succeeds: 1 - (1 - P)^k."""
        return 1.0 - (1.0 - traj_p) ** k

    @staticmethod
    def dependability_pass_hat_k(traj_p: float, k: int) -> float:
        """Probability that all k consecutive runs succeed: P^k."""
        return traj_p**k

    @staticmethod
    def beta_moment(alpha: float, beta: float, order_k: int) -> float:
        """Compute E[P^k] = B(alpha + k, beta) / B(alpha, beta) for P ~ Beta(alpha, beta)."""
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Beta parameters alpha and beta must be positive.")
        log_moment = (
            math.lgamma(alpha + order_k)
            + math.lgamma(alpha + beta)
            - math.lgamma(alpha)
            - math.lgamma(alpha + beta + order_k)
        )
        return math.exp(log_moment)

    @classmethod
    def beta_heterogeneous_metrics(
        cls, mean_p: float, concentration: float, k: int
    ) -> tuple[float, float]:
        """Compute marginal (E[pass@k], E[pass^k]) under P_traj ~ Beta(alpha, beta).

        Note: E[1 - (1 - P)^k] = 1 - E[(1 - P)^k] = 1 - B(alpha, beta + k) / B(alpha, beta).
        """
        alpha = mean_p * concentration
        beta = (1.0 - mean_p) * concentration
        pass_hat_k = cls.beta_moment(alpha, beta, k)
        fail_hat_k = cls.beta_moment(beta, alpha, k)
        pass_at_k = 1.0 - fail_hat_k
        return pass_at_k, pass_hat_k

    @staticmethod
    def markov_trajectory_success(
        p_init: float, p_stay_ok: float, num_steps: int
    ) -> float:
        """Compute P(S_1=1, ..., S_m=1) = P(S_1=1) * P(S_t=1 | S_{t-1}=1)^{m-1}."""
        if num_steps == 1:
            return p_init
        return p_init * (p_stay_ok ** (num_steps - 1))

    @classmethod
    def evaluate_configuration(
        cls,
        step_p: float,
        num_steps: int,
        k: int,
        concentration: float = 4.0,
        markov_contagion_penalty: float = 0.04,
    ) -> TrajectoryReport:
        """Produce a complete deterministic analytical report for a given configuration."""
        traj_p = cls.trajectory_success_iid(step_p, num_steps)
        p_at_k = cls.capability_pass_at_k(traj_p, k)
        p_hat_k = cls.dependability_pass_hat_k(traj_p, k)
        b_at_k, b_hat_k = cls.beta_heterogeneous_metrics(traj_p, concentration, k)
        # Under step error contagion, conditional step stability drops or cascades
        p_stay = max(0.0, min(1.0, step_p - markov_contagion_penalty))
        m_traj_p = cls.markov_trajectory_success(step_p, p_stay, num_steps)
        m_hat_k = cls.dependability_pass_hat_k(m_traj_p, k)
        return TrajectoryReport(
            step_success_p=step_p,
            num_steps_m=num_steps,
            k_runs=k,
            traj_p=traj_p,
            pass_at_k=p_at_k,
            pass_hat_k=p_hat_k,
            illusion_gap=p_at_k - p_hat_k,
            beta_pass_at_k=b_at_k,
            beta_pass_hat_k=b_hat_k,
            markov_traj_p=m_traj_p,
            markov_pass_hat_k=m_hat_k,
        )

    @staticmethod
    def run_monte_carlo_lab(
        step_p: float,
        num_steps: int,
        k: int,
        concentration: float,
        num_trials: int = 10_000,
        seed: int = 202601,
    ) -> dict[str, float]:
        """Run Lab 1.1 Monte Carlo verification comparing homogeneous vs. Beta-heterogeneous runs."""
        rng = random.Random(seed)
        traj_p = step_p**num_steps
        alpha = traj_p * concentration
        beta = (1.0 - traj_p) * concentration

        iid_at_k_hits = 0
        iid_hat_k_hits = 0
        beta_at_k_hits = 0
        beta_hat_k_hits = 0

        for _ in range(num_trials):
            # Homogeneous i.i.d. multi-step simulation
            iid_successes = 0
            for _ in range(k):
                step_ok = True
                for _ in range(num_steps):
                    if rng.random() > step_p:
                        step_ok = False
                        break
                if step_ok:
                    iid_successes += 1
            if iid_successes >= 1:
                iid_at_k_hits += 1
            if iid_successes == k:
                iid_hat_k_hits += 1

            # Beta-heterogeneous prompt difficulty simulation
            prompt_p = rng.betavariate(alpha, beta)
            beta_successes = sum(1 for _ in range(k) if rng.random() <= prompt_p)
            if beta_successes >= 1:
                beta_at_k_hits += 1
            if beta_successes == k:
                beta_hat_k_hits += 1

        return {
            "mc_iid_pass_at_k": iid_at_k_hits / num_trials,
            "mc_iid_pass_hat_k": iid_hat_k_hits / num_trials,
            "mc_beta_pass_at_k": beta_at_k_hits / num_trials,
            "mc_beta_pass_hat_k": beta_hat_k_hits / num_trials,
        }


if __name__ == "__main__":
    analyzer = TrajectoryReliabilityAnalyzer()
    print("=== WORKED AUTOPSY & COMBINATORIAL VERIFICATION ===")
    # 1.1 Autopsy numbers: 5-step billing agent with step_p = 0.93115 (so traj_p = 0.7000, pass^5 = 0.16807)
    # and hard-cohort traj_p = 0.6000 (where pass@5 = 1 - 0.4^5 = 0.98976 = 98.98%)
    step_p_autopsy = 0.70 ** (1.0 / 5.0)
    rep_autopsy = analyzer.evaluate_configuration(step_p=step_p_autopsy, num_steps=5, k=5, concentration=3.0)
    rep_demo_tier = analyzer.evaluate_configuration(step_p=0.60 ** (1.0 / 5.0), num_steps=5, k=5, concentration=3.0)
    print(
        f"Autopsy Standard Tier (step_p={rep_autopsy.step_success_p:.5f}, m=5, k=5): "
        f"traj_p={rep_autopsy.traj_p:.4f}, pass@5={rep_autopsy.pass_at_k:.5f}, "
        f"pass^5={rep_autopsy.pass_hat_k:.5f}, gap={rep_autopsy.illusion_gap:.5f}, "
        f"beta_pass@5={rep_autopsy.beta_pass_at_k:.5f}, beta_pass^5={rep_autopsy.beta_pass_hat_k:.5f}"
    )
    print(
        f"Autopsy Demo Showcase Tier (step_p={rep_demo_tier.step_success_p:.5f}, m=5, k=5): "
        f"traj_p={rep_demo_tier.traj_p:.4f}, pass@5={rep_demo_tier.pass_at_k:.5f} (98.98%), "
        f"pass^5={rep_demo_tier.pass_hat_k:.5f} (7.78%)"
    )
    # Unbiased pass@k example: n=20 samples, c=6 correct, k in {1, 3, 5, 10}
    for k_val in (1, 3, 5, 10):
        ub = analyzer.unbiased_pass_at_k(n=20, c=6, k=k_val)
        naive = 1.0 - (1.0 - 6 / 20) ** k_val
        print(f"Unbiased pass@{k_val} (n=20, c=6): exact={ub:.5f} vs naive={naive:.5f}")
        if k_val > 1:
            assert naive < ub, "Concavity of g(p) = 1-(1-p)^k implies naive plug-in < unbiased estimator"

    # Exact verification of Jensen's Inequality negative bias on plug-in pass@k at (p=0.5, n=2, k=2)
    true_pass_at_2 = 1.0 - (1.0 - 0.5) ** 2  # 0.75
    pmf_n2 = [(0, 0.25), (1, 0.50), (2, 0.25)]
    exp_plugin = sum(prob * (1.0 - (1.0 - (c_val / 2.0)) ** 2) for c_val, prob in pmf_n2)
    exp_unbiased = sum(prob * analyzer.unbiased_pass_at_k(n=2, c=c_val, k=2) for c_val, prob in pmf_n2)
    plugin_bias = exp_plugin - true_pass_at_2
    unbiased_bias = exp_unbiased - true_pass_at_2
    assert abs(true_pass_at_2 - 0.75) < 1e-12
    assert abs(exp_plugin - 0.625) < 1e-12
    assert abs(plugin_bias - (-0.125)) < 1e-12
    assert abs(unbiased_bias) < 1e-12
    print(
        f"Jensen Concavity Check (p=0.5, n=2, k=2): true={true_pass_at_2:.3f}, "
        f"E[plug-in]={exp_plugin:.3f} (bias={plugin_bias:+.3f}), "
        f"E[unbiased]={exp_unbiased:.3f} (bias={unbiased_bias:+.3f})"
    )

    print("\n=== LAB 1.1: 10,000-TRIAL MONTE CARLO GRID (step_p = 0.95, concentration = 4.0) ===")
    for m_steps in (1, 5, 10, 20):
        for k_runs in (1, 3, 5, 10):
            rep = analyzer.evaluate_configuration(step_p=0.95, num_steps=m_steps, k=k_runs, concentration=4.0)
            mc = analyzer.run_monte_carlo_lab(
                step_p=0.95, num_steps=m_steps, k=k_runs, concentration=4.0, num_trials=10_000
            )
            print(
                f"m={m_steps:2d}, k={k_runs:2d} | P_traj={rep.traj_p:.4f} | "
                f"pass@k={rep.pass_at_k:.4f} (MC:{mc['mc_iid_pass_at_k']:.4f}) | "
                f"pass^k={rep.pass_hat_k:.4f} (MC:{mc['mc_iid_pass_hat_k']:.4f}) | "
                f"Beta pass@k={rep.beta_pass_at_k:.4f} (MC:{mc['mc_beta_pass_at_k']:.4f}) | "
                f"Beta pass^k={rep.beta_pass_hat_k:.4f} (MC:{mc['mc_beta_pass_hat_k']:.4f})"
            )
