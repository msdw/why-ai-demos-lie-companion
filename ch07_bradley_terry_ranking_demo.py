"""
Chapter 7: Pairwise Ranking Systems — Sequential Elo Volatility vs. Bradley-Terry-Davidson MLE & Cycles.
Why AI Demos Lie — Companion Verification Module (Python 3.12+, Zero External Dependencies).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRecord:
    model_a: int  # Presented in first position (Slot 1)
    model_b: int  # Presented in second position (Slot 2)
    outcome: int  # 1 = A wins, -1 = B wins, 0 = Tie


@dataclass(frozen=True)
class RankingAuditResult:
    model_names: tuple[str, ...]
    bt_elo_ratings: tuple[float, ...]
    bt_ci_lower: tuple[float, ...]
    bt_ci_upper: tuple[float, ...]
    tie_nu: float
    position_gamma: float
    seq_elo_mean: tuple[float, ...]
    seq_elo_min: tuple[float, ...]
    seq_elo_max: tuple[float, ...]
    seq_elo_spread: tuple[float, ...]
    rank_inversion_rate: float
    condorcet_cycles: tuple[tuple[str, str, str], ...]
    max_curl_residual: float


class BradleyTerryDavidsonEngine:
    """Order-independent Bradley-Terry-Davidson MLE with position bias and Condorcet cycle diagnostics."""

    @staticmethod
    def run_sequential_elo(
        matches: list[MatchRecord], n_models: int, k_factor: float = 32.0, base_elo: float = 1000.0
    ) -> list[float]:
        ratings = [base_elo] * n_models
        for m in matches:
            ra, rb = ratings[m.model_a], ratings[m.model_b]
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            sa = 1.0 if m.outcome == 1 else (0.0 if m.outcome == -1 else 0.5)
            ratings[m.model_a] += k_factor * (sa - ea)
            ratings[m.model_b] += k_factor * ((1.0 - sa) - (1.0 - ea))
        # Center mean at base_elo
        offset = base_elo - statistics.fmean(ratings)
        return [r + offset for r in ratings]

    @staticmethod
    def fit_bradley_terry_davidson(
        matches: list[MatchRecord], n_models: int, max_iter: int = 300, tol: float = 1e-9
    ) -> tuple[list[float], float, float]:
        """Fits skills pi_i, Davidson tie parameter nu, and Slot-1 position multiplier gamma via MM iteration."""
        pi = [1.0] * n_models
        nu = 0.35
        gamma = 1.05

        wins = [0.0] * n_models
        ties = [0.0] * n_models
        slot1_wins = 0
        slot1_losses = 0
        total_ties = 0

        for m in matches:
            if m.outcome == 1:
                wins[m.model_a] += 1.0
                slot1_wins += 1
            elif m.outcome == -1:
                wins[m.model_b] += 1.0
                slot1_losses += 1
            else:
                ties[m.model_a] += 1.0
                ties[m.model_b] += 1.0
                total_ties += 1

        for _ in range(max_iter):
            old_pi = pi[:]
            for i in range(n_models):
                eff_wins = wins[i] + 0.5 * ties[i]
                denom_sum = 0.0
                for m in matches:
                    if m.model_a == i or m.model_b == i:
                        pa = gamma * pi[m.model_a]
                        pb = pi[m.model_b]
                        sqrt_ab = math.sqrt(pa * pb)
                        d = pa + pb + nu * sqrt_ab
                        if m.model_a == i:
                            denom_sum += (gamma + 0.5 * nu * math.sqrt(gamma * pb / max(1e-12, pi[i]))) / d
                        else:
                            denom_sum += (1.0 + 0.5 * nu * math.sqrt(pa / max(1e-12, pi[i]))) / d
                if denom_sum > 1e-12:
                    pi[i] = eff_wins / denom_sum

            # Normalize geometric mean of pi to 1.0
            log_mean = sum(math.log(max(1e-12, x)) for x in pi) / n_models
            scale = math.exp(-log_mean)
            pi = [x * scale for x in pi]

            # Update tie parameter nu
            if total_ties > 0:
                nu_denom = 0.0
                for m in matches:
                    pa = gamma * pi[m.model_a]
                    pb = pi[m.model_b]
                    sqrt_ab = math.sqrt(pa * pb)
                    nu_denom += sqrt_ab / (pa + pb + nu * sqrt_ab)
                if nu_denom > 1e-12:
                    nu = total_ties / nu_denom

            # Update Slot-1 position multiplier gamma
            g_num = slot1_wins + 0.5 * total_ties
            g_den = 0.0
            for m in matches:
                pa_raw = pi[m.model_a]
                pb = pi[m.model_b]
                pa = gamma * pa_raw
                sqrt_ab = math.sqrt(pa * pb)
                d = pa + pb + nu * sqrt_ab
                g_den += (pa_raw + 0.5 * nu * math.sqrt(pa_raw * pb / max(1e-12, gamma))) / d
            if g_den > 1e-12:
                gamma = g_num / g_den

            if max(abs(pi[i] - old_pi[i]) for i in range(n_models)) < tol:
                break

        # Convert pi to Elo scale centered at 1000: R_i = 1000 + 400 * log10(pi_i)
        elo_ratings = [1000.0 + 400.0 * math.log10(x) for x in pi]
        return elo_ratings, nu, gamma

    @staticmethod
    def detect_non_transitive_cycles(
        matches: list[MatchRecord], model_names: list[str]
    ) -> tuple[list[tuple[str, str, str]], float]:
        n = len(model_names)
        wins = [[0.0] * n for _ in range(n)]
        totals = [[0.0] * n for _ in range(n)]
        for m in matches:
            totals[m.model_a][m.model_b] += 1.0
            totals[m.model_b][m.model_a] += 1.0
            if m.outcome == 1:
                wins[m.model_a][m.model_b] += 1.0
            elif m.outcome == -1:
                wins[m.model_b][m.model_a] += 1.0
            else:
                wins[m.model_a][m.model_b] += 0.5
                wins[m.model_b][m.model_a] += 0.5

        p_mat = [[0.5] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and totals[i][j] > 0:
                    p_mat[i][j] = wins[i][j] / totals[i][j]

        cycles: list[tuple[str, str, str]] = []
        max_curl = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Logit flow f_uv = ln(p_uv / p_vu)
                    def flow(u: int, v: int) -> float:
                        p = min(0.99, max(0.01, p_mat[u][v]))
                        return math.log(p / (1.0 - p))

                    curl = abs(flow(i, j) + flow(j, k) + flow(k, i))
                    max_curl = max(max_curl, curl)
                    if p_mat[i][j] > 0.55 and p_mat[j][k] > 0.55 and p_mat[k][i] > 0.55:
                        cycles.append((model_names[i], model_names[j], model_names[k]))
                    elif p_mat[j][i] > 0.55 and p_mat[i][k] > 0.55 and p_mat[k][j] > 0.55:
                        cycles.append((model_names[j], model_names[i], model_names[k]))
        return cycles, max_curl


def run_lab_7_1(n_permutations: int = 1000, n_boot: int = 200) -> RankingAuditResult:
    rng = random.Random(202607)
    names = ["Alpha-Code", "Beta-Reason", "Gamma-Safe", "Delta-Fast", "Epsilon-RAG", "Zeta-Base"]
    n_models = len(names)

    # True pairwise preference matrix with an embedded Condorcet cycle across Alpha (0), Beta (1), Gamma (2)
    # Alpha beats Beta (0.68), Beta beats Gamma (0.66), Gamma beats Alpha (0.64) on specialized rubric slice
    true_p = [
        [0.50, 0.68, 0.36, 0.72, 0.75, 0.84],
        [0.32, 0.50, 0.66, 0.70, 0.74, 0.82],
        [0.64, 0.34, 0.50, 0.68, 0.71, 0.80],
        [0.28, 0.30, 0.32, 0.50, 0.58, 0.72],
        [0.25, 0.26, 0.29, 0.42, 0.50, 0.65],
        [0.16, 0.18, 0.20, 0.28, 0.35, 0.50],
    ]

    matches: list[MatchRecord] = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            for rep in range(40):  # 40 games per pair (20 in order IJ, 20 in order JI) -> 600 total matches
                a, b = (i, j) if rep < 20 else (j, i)
                # Apply Slot-1 position advantage (+0.045 win boost) and 12% tie probability
                p_a_win = min(0.92, max(0.05, true_p[a][b] + 0.045))
                u = rng.random()
                if u < 0.12:
                    outcome = 0
                elif u < 0.12 + (1.0 - 0.12) * p_a_win:
                    outcome = 1
                else:
                    outcome = -1
                matches.append(MatchRecord(a, b, outcome))

    bt_elo, nu, gamma = BradleyTerryDavidsonEngine.fit_bradley_terry_davidson(matches, n_models)
    bt_top_idx = max(range(n_models), key=lambda idx: bt_elo[idx])

    # Permutation audit of Sequential Elo
    seq_records: list[list[float]] = [[] for _ in range(n_models)]
    inversions = 0
    for _ in range(n_permutations):
        shuffled = matches[:]
        rng.shuffle(shuffled)
        s_elo = BradleyTerryDavidsonEngine.run_sequential_elo(shuffled, n_models, k_factor=32.0)
        for idx, val in enumerate(s_elo):
            seq_records[idx].append(val)
        seq_top_idx = max(range(n_models), key=lambda idx: s_elo[idx])
        if seq_top_idx != bt_top_idx or s_elo[0] < s_elo[2]:
            inversions += 1

    # Bootstrap 95% CIs for Bradley-Terry-Davidson
    boot_records: list[list[float]] = [[] for _ in range(n_models)]
    for _ in range(n_boot):
        sample = [matches[rng.randrange(len(matches))] for _ in range(len(matches))]
        b_elo, _, _ = BradleyTerryDavidsonEngine.fit_bradley_terry_davidson(sample, n_models, max_iter=120)
        for idx, val in enumerate(b_elo):
            boot_records[idx].append(val)

    ci_low, ci_high = [], []
    for idx in range(n_models):
        s_boot = sorted(boot_records[idx])
        ci_low.append(s_boot[int(0.025 * n_boot)])
        ci_high.append(s_boot[int(0.975 * n_boot)])

    cycles, max_curl = BradleyTerryDavidsonEngine.detect_non_transitive_cycles(matches, names)

    return RankingAuditResult(
        model_names=tuple(names),
        bt_elo_ratings=tuple(bt_elo),
        bt_ci_lower=tuple(ci_low),
        bt_ci_upper=tuple(ci_high),
        tie_nu=nu,
        position_gamma=gamma,
        seq_elo_mean=tuple(statistics.fmean(r) for r in seq_records),
        seq_elo_min=tuple(min(r) for r in seq_records),
        seq_elo_max=tuple(max(r) for r in seq_records),
        seq_elo_spread=tuple(max(r) - min(r) for r in seq_records),
        rank_inversion_rate=inversions / n_permutations,
        condorcet_cycles=tuple(cycles),
        max_curl_residual=max_curl,
    )


if __name__ == "__main__":
    res = run_lab_7_1()
    print("=== CHAPTER 7: BRADLEY-TERRY-DAVIDSON MLE VS. SEQUENTIAL ELO (600 MATCHES, 1000 PERMUTATIONS) ===")
    print(f"Estimated Davidson Tie Parameter (nu)    : {res.tie_nu:.3f}")
    print(f"Estimated Slot-1 Position Multiplier (g) : {res.position_gamma:.3f}")
    print(f"Sequential Elo Top-3 Rank Inversion Rate : {res.rank_inversion_rate:.1%}")
    print(f"Detected Condorcet Non-Transitive Cycles : {res.condorcet_cycles}")
    print(f"Maximum Triadic Logit Curl Residual      : {res.max_curl_residual:.3f}\n")
    print(f"{'Model':<14} | {'BTD MLE Elo':>11} | {'BTD 95% Boot CI':>19} | {'Seq Elo [Min, Max]':>20} | {'Seq Spread':>10}")
    print("-" * 86)
    for i, name in enumerate(res.model_names):
        print(
            f"{name:<14} | {res.bt_elo_ratings[i]:11.1f} | "
            f"[{res.bt_ci_lower[i]:7.1f}, {res.bt_ci_upper[i]:7.1f}] | "
            f"[{res.seq_elo_min[i]:7.1f}, {res.seq_elo_max[i]:7.1f}] | "
            f"{res.seq_elo_spread[i]:10.1f}"
        )

    # Self-test assertions
    assert res.rank_inversion_rate > 0.50
    assert len(res.condorcet_cycles) >= 1 and res.max_curl_residual > 1.0
