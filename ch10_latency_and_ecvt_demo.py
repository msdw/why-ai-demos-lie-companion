"""Chapter 10: The Mathematics of Agentic Latency, Tail Amplification, and Unit Economics.

Deterministic reference implementation of AgentLatencyAndEconomicsSimulator and Lab 10.1
across 10,000 simulated multi-step agentic workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class ServingProfile:
    name: str
    steps_m: int
    fanout_k: int
    base_prompt_tokens: int
    step_growth_tokens: int
    output_tokens_per_step: int
    kv_cache_hit_ratio: float
    kv_cache_discount: float
    input_price_per_1m: float
    output_price_per_1m: float
    tool_cost_per_step: float
    step_logn_mu: float
    step_logn_sigma: float
    pass_k_reliability: float
    escalation_prob: float
    human_review_cost: float
    incident_prob: float
    incident_remediation_cost: float


@dataclass(frozen=True)
class LatencyAndCostReport:
    name: str
    tail_stall_prob_p99: float
    fw_mean_sec: float
    fw_p50_sec: float
    fw_p95_sec: float
    fw_p99_sec: float
    mc_p50_sec: float
    mc_p95_sec: float
    mc_p99_sec: float
    slo_breach_rate_8s: float
    cumulative_input_tokens: int
    nominal_token_cost_usd: float
    cached_token_cost_usd: float
    ecvt_usd: float


class AgentLatencyAndEconomicsSimulator:
    """Models TTFT/ITL latency compounding, Fenton-Wilkinson tails, and ECVT."""

    Z_95: float = 1.6448536269514722
    Z_99: float = 2.3263478740408408

    @staticmethod
    def decompose_single_call_latency(
        queue_ms: float,
        net_ms: float,
        input_tokens: int,
        prefill_tok_per_sec: float,
        output_tokens: int,
        itl_ms: float,
        kv_cache_hit_ratio: float = 0.0,
    ) -> tuple[float, float, float]:
        """Returns (TTFT_ms, decode_ms, total_call_ms)."""
        effective_prefill_tokens = input_tokens * (1.0 - 0.85 * kv_cache_hit_ratio)
        prefill_ms = (effective_prefill_tokens / prefill_tok_per_sec) * 1000.0
        ttft_ms = queue_ms + net_ms + prefill_ms
        decode_ms = output_tokens * itl_ms
        return ttft_ms, decode_ms, ttft_ms + decode_ms

    @staticmethod
    def sequential_tail_amplification(steps_m: int, quantile_q: float = 0.99) -> float:
        """Probability that at least one step in an M-step chain exceeds quantile_q."""
        return 1.0 - (quantile_q ** steps_m)

    @classmethod
    def fenton_wilkinson_lognormal_sum(
        cls, steps_m: int, mu: float, sigma: float
    ) -> tuple[float, float, float, float]:
        """Approximates sum of M i.i.d. LogNormal(mu, sigma^2) variables.

        Returns (mean, p50, p95, p99) in seconds.
        """
        step_mean = math.exp(mu + 0.5 * sigma * sigma)
        step_var = (math.exp(sigma * sigma) - 1.0) * math.exp(2.0 * mu + sigma * sigma)
        total_mean = steps_m * step_mean
        total_var = steps_m * step_var

        sigma_z2 = math.log(1.0 + total_var / (total_mean * total_mean))
        sigma_z = math.sqrt(sigma_z2)
        mu_z = math.log(total_mean) - 0.5 * sigma_z2

        p50 = math.exp(mu_z)
        p95 = math.exp(mu_z + cls.Z_95 * sigma_z)
        p99 = math.exp(mu_z + cls.Z_99 * sigma_z)
        return total_mean, p50, p95, p99

    @staticmethod
    def cumulative_tokens_and_cost(profile: ServingProfile) -> tuple[int, float, float]:
        """Computes quadratic token accumulation M*L0 + M*(M+1)/2 * dL and costs."""
        m = profile.steps_m
        cum_in = m * profile.base_prompt_tokens + (m * (m + 1) // 2) * profile.step_growth_tokens
        cum_out = m * profile.output_tokens_per_step
        total_calls = m * profile.fanout_k
        cum_in_total = cum_in * profile.fanout_k
        cum_out_total = cum_out * profile.fanout_k

        nominal_in_cost = (cum_in_total / 1_000_000.0) * profile.input_price_per_1m
        out_cost = (cum_out_total / 1_000_000.0) * profile.output_price_per_1m
        effective_in_multiplier = 1.0 - (profile.kv_cache_hit_ratio * profile.kv_cache_discount)
        cached_in_cost = nominal_in_cost * effective_in_multiplier

        tool_cost = total_calls * profile.tool_cost_per_step
        return (
            cum_in_total,
            nominal_in_cost + out_cost + tool_cost,
            cached_in_cost + out_cost + tool_cost,
        )

    @classmethod
    def evaluate_profile(
        cls, profile: ServingProfile, trials: int = 10_000, slo_sec: float = 8.0, seed: int = 202610
    ) -> LatencyAndCostReport:
        rng = random.Random(seed)
        total_invocations = profile.steps_m * profile.fanout_k
        tail_prob = cls.sequential_tail_amplification(total_invocations, 0.99)
        fw_mean, fw_p50, fw_p95, fw_p99 = cls.fenton_wilkinson_lognormal_sum(
            profile.steps_m, profile.step_logn_mu, profile.step_logn_sigma
        )

        sim_totals: list[float] = []
        for _ in range(trials):
            workflow_sec = 0.0
            for _step in range(profile.steps_m):
                if profile.fanout_k == 1:
                    workflow_sec += rng.lognormvariate(profile.step_logn_mu, profile.step_logn_sigma)
                else:
                    branch_max = max(
                        rng.lognormvariate(profile.step_logn_mu, profile.step_logn_sigma)
                        for _b in range(profile.fanout_k)
                    )
                    workflow_sec += branch_max
            sim_totals.append(workflow_sec)

        sim_totals.sort()
        mc_p50 = sim_totals[int(0.50 * trials)]
        mc_p95 = sim_totals[int(0.95 * trials)]
        mc_p99 = sim_totals[int(0.99 * trials)]
        breaches = sum(1 for t in sim_totals if t > slo_sec) / trials

        cum_in, nom_cost, cached_cost = cls.cumulative_tokens_and_cost(profile)
        expected_esc = profile.escalation_prob * profile.human_review_cost
        expected_inc = profile.incident_prob * profile.incident_remediation_cost
        ecvt = (cached_cost + expected_esc + expected_inc) / profile.pass_k_reliability

        return LatencyAndCostReport(
            name=profile.name,
            tail_stall_prob_p99=tail_prob,
            fw_mean_sec=fw_mean,
            fw_p50_sec=fw_p50,
            fw_p95_sec=fw_p95,
            fw_p99_sec=fw_p99,
            mc_p50_sec=mc_p50,
            mc_p95_sec=mc_p95,
            mc_p99_sec=mc_p99,
            slo_breach_rate_8s=breaches,
            cumulative_input_tokens=cum_in,
            nominal_token_cost_usd=nom_cost,
            cached_token_cost_usd=cached_cost,
            ecvt_usd=ecvt,
        )


def run_worked_examples_and_lab() -> None:
    sim = AgentLatencyAndEconomicsSimulator()
    ttft, dec, total_ms = sim.decompose_single_call_latency(
        queue_ms=35.0, net_ms=25.0, input_tokens=12_000,
        prefill_tok_per_sec=24_000.0, output_tokens=250, itl_ms=14.0, kv_cache_hit_ratio=0.0
    )
    ttft_c, _, total_c_ms = sim.decompose_single_call_latency(
        queue_ms=35.0, net_ms=25.0, input_tokens=12_000,
        prefill_tok_per_sec=24_000.0, output_tokens=250, itl_ms=14.0, kv_cache_hit_ratio=0.80
    )
    print("=== WORKED EXAMPLE 1: TTFT & ITL DECOMPOSITION ===")
    print(f"Uncached: TTFT={ttft:.1f}ms, Decode={dec:.1f}ms, Total={total_ms:.1f}ms")
    print(f"80% KV-Cached: TTFT={ttft_c:.1f}ms, Decode={dec:.1f}ms, Total={total_c_ms:.1f}ms")
    print(f"P99 Tail Stall Prob: M=1 -> {sim.sequential_tail_amplification(1)*100:.2f}%, "
          f"M=10 -> {sim.sequential_tail_amplification(10)*100:.2f}%, "
          f"M=20 -> {sim.sequential_tail_amplification(20)*100:.2f}%")

    profiles = [
        ServingProfile(
            name="1. Bounded 4-Step State Machine (Cached)",
            steps_m=4, fanout_k=1, base_prompt_tokens=2000, step_growth_tokens=600,
            output_tokens_per_step=180, kv_cache_hit_ratio=0.80, kv_cache_discount=0.75,
            input_price_per_1m=3.00, output_price_per_1m=15.00, tool_cost_per_step=0.002,
            step_logn_mu=-0.35, step_logn_sigma=0.45, pass_k_reliability=0.88,
            escalation_prob=0.06, human_review_cost=4.50, incident_prob=0.002,
            incident_remediation_cost=120.00,
        ),
        ServingProfile(
            name="2. Parallel 4x Fan-Out Map-Reduce (3 Stages)",
            steps_m=3, fanout_k=4, base_prompt_tokens=2000, step_growth_tokens=500,
            output_tokens_per_step=160, kv_cache_hit_ratio=0.70, kv_cache_discount=0.75,
            input_price_per_1m=3.00, output_price_per_1m=15.00, tool_cost_per_step=0.002,
            step_logn_mu=-0.35, step_logn_sigma=0.45, pass_k_reliability=0.84,
            escalation_prob=0.08, human_review_cost=4.50, incident_prob=0.004,
            incident_remediation_cost=120.00,
        ),
        ServingProfile(
            name="3. Sequential 10-Step ReAct Agent (Uncached)",
            steps_m=10, fanout_k=1, base_prompt_tokens=2500, step_growth_tokens=800,
            output_tokens_per_step=220, kv_cache_hit_ratio=0.0, kv_cache_discount=0.0,
            input_price_per_1m=3.00, output_price_per_1m=15.00, tool_cost_per_step=0.002,
            step_logn_mu=-0.30, step_logn_sigma=0.50, pass_k_reliability=0.52,
            escalation_prob=0.22, human_review_cost=4.50, incident_prob=0.015,
            incident_remediation_cost=120.00,
        ),
        ServingProfile(
            name="4. Autonomous 4-Agent Chat Loop (16 Turns)",
            steps_m=16, fanout_k=1, base_prompt_tokens=3000, step_growth_tokens=1100,
            output_tokens_per_step=260, kv_cache_hit_ratio=0.15, kv_cache_discount=0.75,
            input_price_per_1m=3.00, output_price_per_1m=15.00, tool_cost_per_step=0.002,
            step_logn_mu=-0.25, step_logn_sigma=0.55, pass_k_reliability=0.31,
            escalation_prob=0.38, human_review_cost=4.50, incident_prob=0.045,
            incident_remediation_cost=120.00,
        ),
    ]

    print("\n=== LAB 10.1: 10,000-WORKFLOW LATENCY & ECVT SIMULATION ===")
    for p in profiles:
        r = sim.evaluate_profile(p)
        print(
            f"{r.name}\n"
            f"  TailProb(P99)={r.tail_stall_prob_p99*100:5.2f}% | "
            f"FW(P50={r.fw_p50_sec:5.2f}s, P95={r.fw_p95_sec:5.2f}s, P99={r.fw_p99_sec:5.2f}s) | "
            f"MC(P50={r.mc_p50_sec:5.2f}s, P95={r.mc_p95_sec:5.2f}s, P99={r.mc_p99_sec:5.2f}s) | "
            f"SLO>8s={r.slo_breach_rate_8s*100:5.2f}%\n"
            f"  CumInputTok={r.cumulative_input_tokens:,d} | "
            f"NominalCost=${r.nominal_token_cost_usd:.4f} | "
            f"CachedCost=${r.cached_token_cost_usd:.4f} | "
            f"ECVT=${r.ecvt_usd:.4f}"
        )


if __name__ == "__main__":
    run_worked_examples_and_lab()

    # Self-test assertions
    sim_check = AgentLatencyAndEconomicsSimulator()
    assert sim_check.sequential_tail_amplification(10, 0.99) > 0.09
    fw_check = sim_check.fenton_wilkinson_lognormal_sum(5, -0.5, 0.4)
    assert fw_check[2] > fw_check[0] > 0.0
