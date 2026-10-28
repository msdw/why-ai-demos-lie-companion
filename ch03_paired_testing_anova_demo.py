"""Chapter 3 Verification & Lab 3.1: Paired Significance Testing and Multi-Source Variance Decomposition.

Implements PairedEvaluationTester and MultiFactorVarianceDecomposer:
- Unpaired vs. paired standard error under item correlation rho, and variance reduction factor
- Exact two-sided McNemar binomial test and continuity-corrected McNemar chi-square on (b, c)
- Bowker's K x K test of internal symmetry for graded ordinal rubrics
- Cochran's Q test across M >= 3 models or prompts + Holm-Bonferroni step-down correction
- Multi-factor random-effects variance decomposition across items, temperature seeds, prompts, and judges
- Lab 3.1: Power simulation across item correlations rho in {0.0, 0.3, 0.5, 0.72, 0.85}
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class McNemarResult:
    """Output of paired 2x2 evaluation comparison."""

    n: int
    a_both_pass: int
    b_only_a_pass: int
    c_only_b_pass: int
    d_both_fail: int
    lift: float
    se_paired: float
    se_unpaired: float
    phi_correlation: float
    chi2_cc: float
    exact_p_value: float
    ci_95_paired: tuple[float, float]


class PairedEvaluationTester:
    """Paired contingency testing for binary, multi-category, and multi-model benchmarks."""

    @staticmethod
    def normal_cdf(z: float) -> float:
        """Standard normal cumulative distribution function via math.erf."""
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @classmethod
    def chi2_survival(cls, x: float, df: int) -> float:
        """Upper tail P(Chi^2_df >= x) using regularized gamma / Wilson-Hilferty approximation."""
        if x <= 0.0:
            return 1.0
        if df == 1:
            return 2.0 * (1.0 - cls.normal_cdf(math.sqrt(x)))
        if df == 2:
            return math.exp(-0.5 * x)
        # Wilson-Hilferty cube-root normal transformation for df >= 3
        k = float(df)
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        return 1.0 - cls.normal_cdf(z)

    @staticmethod
    def exact_binomial_two_sided_p(b: int, c: int) -> float:
        """Exact two-sided binomial p-value for b ~ Binomial(b + c, 0.5)."""
        m = b + c
        if m == 0:
            return 1.0
        k_min = min(b, c)
        # Sum 2^{-m} * C(m, i) for i = 0 .. k_min in log-space for numerical stability
        tail_prob = 0.0
        log_denom = m * math.log(2.0)
        for i in range(k_min + 1):
            log_binom = math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1)
            tail_prob += math.exp(log_binom - log_denom)
        return min(1.0, 2.0 * tail_prob)

    @classmethod
    def analyze_paired_2x2(cls, a: int, b: int, c: int, d: int) -> McNemarResult:
        """Analyze a 2x2 paired contingency table (a=11, b=10, c=01, d=00)."""
        n = a + b + c + d
        p_a = (a + b) / n
        p_b = (a + c) / n
        lift = p_a - p_b  # equals (b - c) / n

        var_unpaired = (p_a * (1.0 - p_a) / n) + (p_b * (1.0 - p_b) / n)
        var_paired = (b + c - ((b - c) ** 2) / n) / (n * n)
        se_unpaired = math.sqrt(max(0.0, var_unpaired))
        se_paired = math.sqrt(max(0.0, var_paired))

        denom_phi = math.sqrt(p_a * (1.0 - p_a) * p_b * (1.0 - p_b))
        cov_ab = (a / n) - (p_a * p_b)
        phi = cov_ab / denom_phi if denom_phi > 1e-12 else 0.0

        m = b + c
        chi2_cc = ((abs(b - c) - 1.0) ** 2) / m if m > 0 else 0.0
        exact_p = cls.exact_binomial_two_sided_p(b, c)
        ci_paired = (lift - 1.95996 * se_paired, lift + 1.95996 * se_paired)

        return McNemarResult(
            n=n,
            a_both_pass=a,
            b_only_a_pass=b,
            c_only_b_pass=c,
            d_both_fail=d,
            lift=lift,
            se_paired=se_paired,
            se_unpaired=se_unpaired,
            phi_correlation=phi,
            chi2_cc=chi2_cc,
            exact_p_value=exact_p,
            ci_95_paired=ci_paired,
        )

    @classmethod
    def bowker_symmetry_test(cls, matrix: list[list[int]]) -> tuple[float, int, float]:
        """Bowker's KxK test of symmetry for ordinal rubric transitions."""
        k = len(matrix)
        chi2_stat = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                nij = matrix[i][j]
                nji = matrix[j][i]
                if nij + nji > 0:
                    chi2_stat += ((nij - nji) ** 2) / (nij + nji)
        df = k * (k - 1) // 2
        return chi2_stat, df, cls.chi2_survival(chi2_stat, df)

    @classmethod
    def cochrans_q_test(cls, binary_matrix: list[list[int]]) -> tuple[float, int, float]:
        """Cochran's Q test on N items x M models binary matrix."""
        n_items = len(binary_matrix)
        m_models = len(binary_matrix[0])
        col_totals = [sum(binary_matrix[i][j] for i in range(n_items)) for j in range(m_models)]
        row_totals = [sum(binary_matrix[i]) for i in range(n_items)]

        mean_c = sum(col_totals) / m_models
        num = m_models * (m_models - 1) * sum((c - mean_c) ** 2 for c in col_totals)
        den = m_models * sum(row_totals) - sum(r * r for r in row_totals)
        q_stat = num / den if den > 0 else 0.0
        df = m_models - 1
        return q_stat, df, cls.chi2_survival(q_stat, df)

    @staticmethod
    def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, float, bool]]:
        """Step-down Holm-Bonferroni correction returning (orig_idx, raw_p, adj_p, reject)."""
        m = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        results: list[tuple[int, float, float, bool]] = []
        running_max = 0.0
        still_rejecting = True
        for rank, (orig_idx, raw_p) in enumerate(indexed):
            multiplier = m - rank
            adj_p = min(1.0, max(running_max, raw_p * multiplier))
            running_max = adj_p
            if adj_p > alpha:
                still_rejecting = False
            results.append((orig_idx, raw_p, adj_p, still_rejecting))
        return sorted(results, key=lambda x: x[0])


class MultiFactorVarianceDecomposer:
    """Decomposes evaluation score variance across items, seeds, prompts, and judges."""

    @staticmethod
    def decompose_four_way(
        scores: list[list[list[list[float]]]],
    ) -> dict[str, float]:
        """Compute variance components from a 4D tensor [item][prompt][judge][seed]."""
        n_i = len(scores)
        n_p = len(scores[0])
        n_j = len(scores[0][0])
        n_s = len(scores[0][0][0])
        total_obs = n_i * n_p * n_j * n_s

        grand_mean = (
            sum(
                scores[i][p][j][s]
                for i in range(n_i)
                for p in range(n_p)
                for j in range(n_j)
                for s in range(n_s)
            )
            / total_obs
        )

        item_means = [
            sum(scores[i][p][j][s] for p in range(n_p) for j in range(n_j) for s in range(n_s))
            / (n_p * n_j * n_s)
            for i in range(n_i)
        ]
        prompt_means = [
            sum(scores[i][p][j][s] for i in range(n_i) for j in range(n_j) for s in range(n_s))
            / (n_i * n_j * n_s)
            for p in range(n_p)
        ]
        judge_means = [
            sum(scores[i][p][j][s] for i in range(n_i) for p in range(n_p) for s in range(n_s))
            / (n_i * n_p * n_s)
            for j in range(n_j)
        ]
        cell_means = [
            [
                [
                    sum(scores[i][p][j][s] for s in range(n_s)) / n_s
                    for j in range(n_j)
                ]
                for p in range(n_p)
            ]
            for i in range(n_i)
        ]

        var_item = sum((m - grand_mean) ** 2 for m in item_means) / (n_i - 1)
        var_prompt = sum((m - grand_mean) ** 2 for m in prompt_means) / (n_p - 1)
        var_judge = sum((m - grand_mean) ** 2 for m in judge_means) / (n_j - 1)

        # Within-cell seed variance
        ss_seed = sum(
            (scores[i][p][j][s] - cell_means[i][p][j]) ** 2
            for i in range(n_i)
            for p in range(n_p)
            for j in range(n_j)
            for s in range(n_s)
        )
        var_seed = ss_seed / (n_i * n_p * n_j * (n_s - 1))

        # Interaction residual variance
        ss_inter = sum(
            (
                cell_means[i][p][j]
                - item_means[i]
                - prompt_means[p]
                - judge_means[j]
                + 2.0 * grand_mean
            )
            ** 2
            for i in range(n_i)
            for p in range(n_p)
            for j in range(n_j)
        )
        var_inter = ss_inter / max(1, (n_i * n_p * n_j - n_i - n_p - n_j + 2))
        var_total = var_item + var_prompt + var_judge + var_inter + var_seed
        return {
            "var_item": var_item,
            "var_prompt": var_prompt,
            "var_judge": var_judge,
            "var_interaction": var_inter,
            "var_seed": var_seed,
            "var_total": var_total,
            "pct_item": 100.0 * var_item / var_total,
            "pct_prompt": 100.0 * var_prompt / var_total,
            "pct_judge": 100.0 * var_judge / var_total,
            "pct_interaction": 100.0 * var_inter / var_total,
            "pct_seed": 100.0 * var_seed / var_total,
        }


def run_power_comparison_lab(
    n_items: int = 200,
    p_a: float = 0.85,
    p_b: float = 0.78,
    rho_values: tuple[float, ...] = (0.0, 0.30, 0.50, 0.72, 0.85),
    num_trials: int = 10_000,
    seed: int = 202603,
) -> None:
    """Lab 3.1: Compare Unpaired Z-test vs. Paired McNemar power across item correlations rho."""
    rng = random.Random(seed)
    tester = PairedEvaluationTester()
    for rho in rho_values:
        cov = rho * math.sqrt(p_a * (1.0 - p_a) * p_b * (1.0 - p_b))
        prob_a = max(0.0, min(p_a, p_b, p_a * p_b + cov))
        prob_b = p_a - prob_a
        prob_c = p_b - prob_a
        prob_d = max(0.0, 1.0 - (prob_a + prob_b + prob_c))

        unpaired_hits = 0
        paired_hits = 0
        for _ in range(num_trials):
            a = b = c = d = 0
            for _ in range(n_items):
                u = rng.random()
                if u < prob_a:
                    a += 1
                elif u < prob_a + prob_b:
                    b += 1
                elif u < prob_a + prob_b + prob_c:
                    c += 1
                else:
                    d += 1
            res = tester.analyze_paired_2x2(a, b, c, d)
            z_unpaired = abs(res.lift) / res.se_unpaired if res.se_unpaired > 0 else 0.0
            if z_unpaired >= 1.95996:
                unpaired_hits += 1
            if res.exact_p_value <= 0.05:
                paired_hits += 1

        var_ratio = (res.se_paired / res.se_unpaired) ** 2
        print(
            f"rho={rho:.2f} | E[b]={n_items*prob_b:5.1f}, E[c]={n_items*prob_c:5.1f} | "
            f"Unpaired Power={unpaired_hits/num_trials:.4f} | "
            f"McNemar Paired Power={paired_hits/num_trials:.4f} | "
            f"VarRatio(Paired/Unpaired)={var_ratio:.4f}"
        )


if __name__ == "__main__":
    tester = PairedEvaluationTester()
    print("=== WORKED AUTOPSY: N=200 PAIRED BENCHMARK (a=148, b=22, c=8, d=22) ===")
    autopsy = tester.analyze_paired_2x2(a=148, b=22, c=8, d=22)
    z_unp = autopsy.lift / autopsy.se_unpaired
    p_unp = 2.0 * (1.0 - tester.normal_cdf(abs(z_unp)))
    print(
        f"Lift={autopsy.lift:.4f} (+7.0%), phi={autopsy.phi_correlation:.4f} | "
        f"SE_unpaired={autopsy.se_unpaired:.5f} (Z={z_unp:.4f}, p={p_unp:.4f}) | "
        f"SE_paired={autopsy.se_paired:.5f} (95% CI=[{autopsy.ci_95_paired[0]:.4f}, {autopsy.ci_95_paired[1]:.4f}]) | "
        f"McNemar chi2_cc={autopsy.chi2_cc:.4f}, exact_p={autopsy.exact_p_value:.5f}"
    )

    print("\n=== BOWKER SYMMETRY (3x3) & COCHRAN'S Q + HOLM-BONFERRONI ===")
    bowker_mat = [[85, 24, 6], [8, 42, 18], [2, 5, 10]]
    b_stat, b_df, b_p = tester.bowker_symmetry_test(bowker_mat)
    print(f"Bowker 3x3: chi2={b_stat:.4f}, df={b_df}, p={b_p:.5f}")

    # Synthetic 4-model comparison on N=200 items
    rng_m = random.Random(202633)
    mat_4m = []
    for _ in range(200):
        base = rng_m.random()
        mat_4m.append([
            1 if base < 0.75 else 0,
            1 if base + rng_m.uniform(-0.15, 0.15) < 0.78 else 0,
            1 if base + rng_m.uniform(-0.15, 0.15) < 0.85 else 0,
            1 if base + rng_m.uniform(-0.15, 0.15) < 0.87 else 0,
        ])
    q_stat, q_df, q_p = tester.cochrans_q_test(mat_4m)
    print(f"Cochran's Q (N=200, M=4): Q={q_stat:.4f}, df={q_df}, p={q_p:.6f}")
    raw_ps = [0.0018, 0.0085, 0.0140, 0.0390, 0.1200, 0.4100]
    print("Holm-Bonferroni:", tester.holm_bonferroni(raw_ps, alpha=0.05))

    print("\n=== MULTI-SOURCE VARIANCE DECOMPOSITION (100 items x 4 prompts x 3 judges x 5 seeds) ===")
    rng_v = random.Random(202634)
    item_eff = [rng_v.gauss(0.0, 0.32) for _ in range(100)]
    prompt_eff = [rng_v.gauss(0.0, 0.11) for _ in range(4)]
    judge_eff = [rng_v.gauss(0.0, 0.09) for _ in range(3)]
    tensor = [
        [
            [
                [
                    0.78
                    + item_eff[i]
                    + prompt_eff[p]
                    + judge_eff[j]
                    + rng_v.gauss(0.0, 0.07) * ((i + p + j) % 3 - 1)
                    + rng_v.gauss(0.0, 0.06)
                    for _ in range(5)
                ]
                for j in range(3)
            ]
            for p in range(4)
        ]
        for i in range(100)
    ]
    v_rep = MultiFactorVarianceDecomposer.decompose_four_way(tensor)
    for k_name, val in v_rep.items():
        print(f"  {k_name}: {val:.5f}")

    print("\n=== LAB 3.1: PAIRED VS UNPAIRED POWER (N=200, pA=0.85, pB=0.78, 10,000 trials) ===")
    run_power_comparison_lab()
    print("\n=== LAB 3.1b: SUBTLE LIFT CASE (N=200, pA=0.85, pB=0.79, rho=0.72) ===")
    run_power_comparison_lab(n_items=200, p_a=0.85, p_b=0.79, rho_values=(0.72,), num_trials=10_000)
    print("\n=== CLM-014 REGISTER VERIFICATION: N=200, pA=0.80, pB=0.86, rho=0.72 (10,000 trials) ===")
    run_power_comparison_lab(n_items=200, p_a=0.80, p_b=0.86, rho_values=(0.72,), num_trials=10_000, seed=2026)

    # Self-test assertions
    assert autopsy.exact_p_value < 0.02 and autopsy.se_paired < autopsy.se_unpaired
    assert v_rep["var_total"] > 0.0 and abs(
        v_rep["pct_item"] + v_rep["pct_prompt"] + v_rep["pct_judge"] + v_rep["pct_interaction"] + v_rep["pct_seed"] - 100.0
    ) < 1e-6
