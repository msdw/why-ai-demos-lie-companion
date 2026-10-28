"""Chapter 14 Verification & Simulation Module:
Production Drift Detection (PSI with Laplace Smoothing, 1-Wasserstein Earth Mover's Distance,
Two-Sample Kolmogorov-Smirnov Statistic, Jensen-Shannon Divergence, and Page's Two-Sided CUSUM).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DriftSnapshot:
    day: int
    psi_score: float
    wasserstein_dist: float
    ks_stat: float
    js_divergence: float
    observed_pass_rate: float
    cusum_pos: float
    cusum_neg: float
    cusum_alarm: bool


class ProductionDriftMonitor:
    """Computes feature, latency, and tool-distribution drift plus Page's sequential CUSUM."""

    @staticmethod
    def compute_psi(
        baseline: Sequence[float],
        target: Sequence[float],
        bin_edges: Sequence[float],
        laplace_eps: float = 1e-4,
    ) -> float:
        """Computes Population Stability Index (PSI) across fixed baseline quantile bins."""
        num_bins = len(bin_edges) - 1
        p_counts = [0] * num_bins
        q_counts = [0] * num_bins

        def assign_bin(val: float) -> int:
            for b in range(num_bins - 1):
                if val < bin_edges[b + 1]:
                    return b
            return num_bins - 1

        for x in baseline:
            p_counts[assign_bin(x)] += 1
        for y in target:
            q_counts[assign_bin(y)] += 1

        n_p = len(baseline)
        n_q = len(target)
        psi = 0.0
        for b in range(num_bins):
            p_b = (p_counts[b] + laplace_eps) / (n_p + num_bins * laplace_eps)
            q_b = (q_counts[b] + laplace_eps) / (n_q + num_bins * laplace_eps)
            psi += (p_b - q_b) * math.log(p_b / q_b)
        return psi

    @staticmethod
    def wasserstein_1d(u_values: Sequence[float], v_values: Sequence[float]) -> float:
        """Exact 1D Earth Mover's (1-Wasserstein) distance between two empirical distributions."""
        u_sorted = sorted(u_values)
        v_sorted = sorted(v_values)
        all_pts = sorted(set(u_sorted + v_sorted))
        if len(all_pts) <= 1:
            return 0.0

        n_u = len(u_sorted)
        n_v = len(v_sorted)
        idx_u = 0
        idx_v = 0
        w_dist = 0.0

        for i in range(len(all_pts) - 1):
            x = all_pts[i]
            next_x = all_pts[i + 1]
            while idx_u < n_u and u_sorted[idx_u] <= x:
                idx_u += 1
            while idx_v < n_v and v_sorted[idx_v] <= x:
                idx_v += 1
            cdf_u = idx_u / n_u
            cdf_v = idx_v / n_v
            w_dist += abs(cdf_u - cdf_v) * (next_x - x)
        return w_dist

    @staticmethod
    def ks_two_sample(u_values: Sequence[float], v_values: Sequence[float]) -> tuple[float, float]:
        """Returns (D_KS statistic, asymptotic critical value at alpha=0.05)."""
        u_sorted = sorted(u_values)
        v_sorted = sorted(v_values)
        all_pts = sorted(set(u_sorted + v_sorted))
        n_u = len(u_sorted)
        n_v = len(v_sorted)
        idx_u = 0
        idx_v = 0
        d_max = 0.0
        for x in all_pts:
            while idx_u < n_u and u_sorted[idx_u] <= x:
                idx_u += 1
            while idx_v < n_v and v_sorted[idx_v] <= x:
                idx_v += 1
            d_max = max(d_max, abs((idx_u / n_u) - (idx_v / n_v)))
        crit_05 = 1.358 * math.sqrt((n_u + n_v) / (n_u * n_v))
        return d_max, crit_05

    @staticmethod
    def jensen_shannon_divergence(
        p_counts: Mapping[str, int],
        q_counts: Mapping[str, int],
        laplace_eps: float = 1e-5,
    ) -> float:
        """Computes base-2 Jensen-Shannon Divergence in [0, 1] across discrete tool categories."""
        keys = sorted(set(p_counts.keys()) | set(q_counts.keys()))
        k_len = len(keys)
        tot_p = sum(p_counts.values())
        tot_q = sum(q_counts.values())
        p_probs = [(p_counts.get(k, 0) + laplace_eps) / (tot_p + k_len * laplace_eps) for k in keys]
        q_probs = [(q_counts.get(k, 0) + laplace_eps) / (tot_q + k_len * laplace_eps) for k in keys]
        m_probs = [0.5 * (p + q) for p, q in zip(p_probs, q_probs)]

        kl_pm = sum(p * math.log2(p / m) for p, m in zip(p_probs, m_probs))
        kl_qm = sum(q * math.log2(q / m) for q, m in zip(q_probs, m_probs))
        return 0.5 * (kl_pm + kl_qm)

    @staticmethod
    def run_tabular_cusum(
        series: Sequence[float],
        mu_0: float,
        sigma: float,
        k_allowance: float,
        h_threshold: float,
    ) -> list[tuple[float, float, bool]]:
        """Page's two-sided tabular CUSUM returning (S_pos, S_neg, alarm) at each step."""
        s_pos = 0.0
        s_neg = 0.0
        history: list[tuple[float, float, bool]] = []
        for x_t in series:
            z_t = (x_t - mu_0) / sigma
            s_pos = max(0.0, s_pos + z_t - k_allowance)
            s_neg = max(0.0, s_neg - z_t - k_allowance)
            alarm = (s_pos >= h_threshold) or (s_neg >= h_threshold)
            history.append((s_pos, s_neg, alarm))
        return history


if __name__ == "__main__":
    monitor = ProductionDriftMonitor()

    # 1. Worked 5-Bin PSI Example in Text
    # Baseline proportions: [0.20, 0.20, 0.20, 0.20, 0.20]
    # Target proportions:   [0.12, 0.15, 0.18, 0.25, 0.30]
    p_vec = [0.20, 0.20, 0.20, 0.20, 0.20]
    q_vec = [0.12, 0.15, 0.18, 0.25, 0.30]
    exact_psi = sum((p - q) * math.log(p / q) for p, q in zip(p_vec, q_vec))
    print(f"Worked 5-Bin Exact PSI = {exact_psi:.5f}")

    # 2. Worked JSD Tool-Call Shift Example in Text
    base_tools = {"sql_query": 500, "rag_search": 350, "billing_refund": 100, "escalate_human": 50}
    drifted_tools = {"sql_query": 280, "rag_search": 220, "billing_refund": 340, "escalate_human": 160}
    jsd_val = monitor.jensen_shannon_divergence(base_tools, drifted_tools)
    print(f"Worked Tool JSD = {jsd_val:.5f}")

    # 3. Lab 14.1: 30-Day Production Telemetry Simulation
    rng = random.Random(202614)
    baseline_prompts = [rng.gauss(420.0, 85.0) for _ in range(2000)]
    bin_edges = [0.0, 320.0, 390.0, 450.0, 520.0, 2000.0]

    snapshots: list[DriftSnapshot] = []
    daily_pass_rates: list[float] = []
    raw_metrics: list[tuple[int, float, float, float, float, float]] = []

    for day in range(1, 31):
        # Days 1-11: Stationary regime (mean prompt 420 tokens, pass rate ~0.920)
        # Days 12-20: Covariate shift in user prompt lengths & tool selection, mild pass decay to ~0.885
        # Days 21-30: Silent provider model update + upstream API regression, pass rate drops to ~0.825
        if day <= 11:
            mu_len, sd_len = 420.0, 85.0
            true_pass = 0.920
            tool_counts = {"sql_query": 500, "rag_search": 350, "billing_refund": 100, "escalate_human": 50}
        elif day <= 20:
            shift_factor = (day - 11) / 9.0
            mu_len = 420.0 + 95.0 * shift_factor
            sd_len = 85.0 + 25.0 * shift_factor
            true_pass = 0.920 - 0.038 * shift_factor
            tool_counts = {
                "sql_query": int(500 - 140 * shift_factor),
                "rag_search": int(350 - 70 * shift_factor),
                "billing_refund": int(100 + 130 * shift_factor),
                "escalate_human": int(50 + 80 * shift_factor),
            }
        else:
            mu_len, sd_len = 535.0, 115.0
            true_pass = 0.825
            tool_counts = {"sql_query": 280, "rag_search": 220, "billing_refund": 340, "escalate_human": 160}

        day_prompts = [rng.gauss(mu_len, sd_len) for _ in range(500)]
        psi = monitor.compute_psi(baseline_prompts, day_prompts, bin_edges)
        w1 = monitor.wasserstein_1d(baseline_prompts, day_prompts)
        ks_d, _ = monitor.ks_two_sample(baseline_prompts, day_prompts)
        jsd = monitor.jensen_shannon_divergence(base_tools, tool_counts)
        obs_pass = sum(1 for _ in range(500) if rng.random() < true_pass) / 500.0
        daily_pass_rates.append(obs_pass)
        raw_metrics.append((day, psi, w1, ks_d, jsd, obs_pass))

    # Run Page's CUSUM on daily oracle pass rates: baseline mu_0=0.920, sigma=sqrt(0.92*0.08/500)=0.01213
    sigma_pass = math.sqrt(0.920 * 0.080 / 500.0)
    cusum_trace = monitor.run_tabular_cusum(daily_pass_rates, mu_0=0.920, sigma=sigma_pass, k_allowance=0.5, h_threshold=4.0)

    for (day, psi, w1, ks_d, jsd, obs_pass), (s_p, s_n, alarm) in zip(raw_metrics, cusum_trace):
        if day in {1, 5, 10, 12, 14, 15, 16, 18, 20, 21, 22, 25, 30}:
            print(
                f"Day {day:02d} | PSI={psi:.4f} | W1={w1:6.2f} | KS={ks_d:.4f} | "
                f"JSD={jsd:.4f} | Pass={obs_pass:.4f} | CUSUM_neg={s_n:5.2f} | Alarm={alarm}"
            )

    # Self-test assertions
    assert raw_metrics[0][1] < 0.05 and raw_metrics[-1][1] > 0.25
    assert any(alarm for _, _, alarm in cusum_trace[14:])
