"""Chapter 13 Verification & Simulation Module:
Annotator Reliability (Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha, Dawid-Skene EM)
and Human-in-the-Loop Queueing & Cognitive Fatigue (M/M/1, Pollaczek-Khinchine M/G/1, M/M/c).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AgreementReport:
    raw_agreement: float
    expected_agreement: float
    statistic: float
    metric_name: str


@dataclass(frozen=True)
class DawidSkeneResult:
    posterior_truth: list[float]
    predicted_labels: list[int]
    rater_sensitivities: list[float]
    rater_specificities: list[float]
    prevalence: float
    iterations: int


@dataclass(frozen=True)
class QueueFatigueMetrics:
    utilization_rho: float
    mm1_wait_sec: float
    mg1_pk_wait_sec: float
    mmc_wait_sec: float
    effective_error_rate: float
    false_approval_rate: float


class AnnotatorReliabilitySuite:
    """Computes chance-corrected IAA metrics and Dawid-Skene EM latent truth."""

    @staticmethod
    def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int], num_classes: int = 2) -> AgreementReport:
        if len(rater_a) != len(rater_b) or not rater_a:
            raise ValueError("Rater arrays must be non-empty and of equal length.")
        n = len(rater_a)
        agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
        p_o = agree / n
        p_e = 0.0
        for k in range(num_classes):
            p_ak = sum(1 for a in rater_a if a == k) / n
            p_bk = sum(1 for b in rater_b if b == k) / n
            p_e += p_ak * p_bk
        kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
        return AgreementReport(p_o, p_e, kappa, "Cohen's Kappa")

    @staticmethod
    def fleiss_kappa(ratings_matrix: Sequence[Sequence[int]], num_classes: int = 2) -> AgreementReport:
        """ratings_matrix[i][r] is label assigned to item i by rater r."""
        n_items = len(ratings_matrix)
        m_raters = len(ratings_matrix[0])
        counts = [[0] * num_classes for _ in range(n_items)]
        class_totals = [0] * num_classes
        for i, row in enumerate(ratings_matrix):
            for label in row:
                counts[i][label] += 1
                class_totals[label] += 1
        p_i_list = [
            sum(n_ik * (n_ik - 1) for n_ik in counts[i]) / (m_raters * (m_raters - 1))
            for i in range(n_items)
        ]
        p_bar = sum(p_i_list) / n_items
        total_judgments = n_items * m_raters
        p_e = sum((c / total_judgments) ** 2 for c in class_totals)
        kappa = (p_bar - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
        return AgreementReport(p_bar, p_e, kappa, "Fleiss' Kappa")

    @staticmethod
    def krippendorff_alpha(
        ratings_matrix: Sequence[Sequence[int | None]],
        metric: str = "nominal",
    ) -> float:
        """Computes Krippendorff's alpha across items with potentially missing (None) ratings."""
        pairable_values: list[int] = []
        d_o_num = 0.0
        n_total = 0

        def delta_sq(c: int, k: int) -> float:
            if metric == "nominal":
                return 0.0 if c == k else 1.0
            return float((c - k) ** 2)

        for row in ratings_matrix:
            valid = [v for v in row if v is not None]
            m_u = len(valid)
            if m_u < 2:
                continue
            n_total += m_u
            pairable_values.extend(valid)
            pair_diff = 0.0
            for idx_a in range(m_u):
                for idx_b in range(m_u):
                    if idx_a != idx_b:
                        pair_diff += delta_sq(valid[idx_a], valid[idx_b])
            d_o_num += pair_diff / (m_u - 1.0)

        if n_total <= 1:
            return 1.0
        d_o = d_o_num / n_total

        freq: dict[int, int] = {}
        for v in pairable_values:
            freq[v] = freq.get(v, 0) + 1

        d_e_num = 0.0
        classes = sorted(freq.keys())
        for c in classes:
            for k in classes:
                if c != k:
                    d_e_num += freq[c] * freq[k] * delta_sq(c, k)
        d_e = d_e_num / (n_total * (n_total - 1.0))
        return 1.0 - (d_o / d_e) if d_e > 0.0 else 1.0

    @staticmethod
    def dawid_skene_em(
        ratings_matrix: Sequence[Sequence[int | None]],
        max_iter: int = 50,
        tol: float = 1e-6,
        laplace: float = 1e-3,
    ) -> DawidSkeneResult:
        """Binary Dawid-Skene EM estimator for latent truth and rater confusion matrices."""
        n_items = len(ratings_matrix)
        m_raters = len(ratings_matrix[0])
        # Initialize posterior truth t_i via majority vote with tie-breaking
        t_i: list[float] = []
        for row in ratings_matrix:
            valid = [v for v in row if v is not None]
            t_i.append(sum(valid) / len(valid) if valid else 0.5)

        sens = [0.80] * m_raters
        spec = [0.80] * m_raters
        pi = sum(t_i) / n_items
        it_used = 0

        for it in range(1, max_iter + 1):
            it_used = it
            pi = max(min(sum(t_i) / n_items, 1.0 - 1e-6), 1e-6)
            for r in range(m_raters):
                tp = laplace
                fn = laplace
                tn = laplace
                fp = laplace
                for i in range(n_items):
                    y = ratings_matrix[i][r]
                    if y is None:
                        continue
                    if y == 1:
                        tp += t_i[i]
                        fp += 1.0 - t_i[i]
                    else:
                        fn += t_i[i]
                        tn += 1.0 - t_i[i]
                sens[r] = tp / (tp + fn)
                spec[r] = tn / (tn + fp)

            max_diff = 0.0
            for i in range(n_items):
                log_p1 = math.log(pi)
                log_p0 = math.log(1.0 - pi)
                for r in range(m_raters):
                    y = ratings_matrix[i][r]
                    if y is None:
                        continue
                    s_1 = min(max(sens[r], 1e-6), 1.0 - 1e-6)
                    s_0 = min(max(spec[r], 1e-6), 1.0 - 1e-6)
                    if y == 1:
                        log_p1 += math.log(s_1)
                        log_p0 += math.log(1.0 - s_0)
                    else:
                        log_p1 += math.log(1.0 - s_1)
                        log_p0 += math.log(s_0)
                m_log = max(log_p1, log_p0)
                p1 = math.exp(log_p1 - m_log)
                p0 = math.exp(log_p0 - m_log)
                new_t = p1 / (p1 + p0)
                max_diff = max(max_diff, abs(new_t - t_i[i]))
                t_i[i] = new_t

            if max_diff < tol:
                break

        preds = [1 if p >= 0.5 else 0 for p in t_i]
        return DawidSkeneResult(t_i, preds, sens, spec, pi, it_used)


class HITLQueueFatigueSimulator:
    """Analytical & discrete-event simulator for M/M/1, Pollaczek-Khinchine M/G/1, and M/M/c queues."""

    @staticmethod
    def evaluate_station(
        arrival_rate_per_sec: float,
        mean_service_sec: float,
        service_std_sec: float,
        num_servers: int = 1,
        shift_minutes: float = 60.0,
        base_error_rate: float = 0.02,
    ) -> QueueFatigueMetrics:
        mu = 1.0 / mean_service_sec
        rho_single = arrival_rate_per_sec / mu
        rho_pool = arrival_rate_per_sec / (num_servers * mu)
        if rho_single < 1.0:
            mm1_wq = rho_single / (mu * (1.0 - rho_single))
            mg1_wq = (arrival_rate_per_sec * (service_std_sec**2 + mean_service_sec**2)) / (
                2.0 * (1.0 - rho_single)
            )
        else:
            mm1_wq = float("inf")
            mg1_wq = float("inf")

        if rho_pool < 1.0:
            a = arrival_rate_per_sec / mu
            sum_terms = sum((a**k) / math.factorial(k) for k in range(num_servers))
            last_term = ((a**num_servers) / math.factorial(num_servers)) * (1.0 / (1.0 - rho_pool))
            p0 = 1.0 / (sum_terms + last_term)
            erlang_c = last_term * p0
            mmc_wq = erlang_c / (num_servers * mu - arrival_rate_per_sec)
        else:
            mmc_wq = float("inf")

        # Cognitive fatigue & queue-pressure rubber-stamping model
        effective_rho = min(rho_pool if num_servers > 1 else rho_single, 0.99)
        pressure_penalty = 0.18 * math.exp(4.5 * max(0.0, effective_rho - 0.70)) - 0.18
        duration_penalty = 0.0018 * max(0.0, shift_minutes - 45.0)
        eff_err = min(0.48, base_error_rate + pressure_penalty + duration_penalty)
        false_approval = min(0.65, eff_err * 1.35)
        return QueueFatigueMetrics(effective_rho, mm1_wq, mg1_wq, mmc_wq, eff_err, false_approval)


if __name__ == "__main__":
    suite = AnnotatorReliabilitySuite()
    # 1. Worked Example: Prevalence Paradox in Cohen's Kappa (N = 200)
    # Confusion matrix: a(1,1)=172, b(1,0)=10, c(0,1)=10, d(0,0)=8
    rater1 = [1] * 182 + [0] * 18
    rater2 = [1] * 172 + [0] * 10 + [1] * 10 + [0] * 8
    ck = suite.cohens_kappa(rater1, rater2, num_classes=2)
    print(f"Worked Cohen's Kappa: p_o={ck.raw_agreement:.4f}, p_e={ck.expected_agreement:.4f}, kappa={ck.statistic:.4f}")

    # 2. Lab 13.1: 300 evaluation items, 5 noisy raters (including 1 adversarial/spam rater)
    rng = random.Random(202613)
    n_eval = 300
    true_labels = [1 if rng.random() < 0.60 else 0 for _ in range(n_eval)]
    # Rater profiles: (sensitivity q1, specificity q0)
    profiles = [(0.90, 0.88), (0.85, 0.84), (0.82, 0.80), (0.76, 0.75), (0.52, 0.50)]
    matrix: list[list[int]] = []
    for y in true_labels:
        row = []
        for q1, q0 in profiles:
            prob_1 = q1 if y == 1 else (1.0 - q0)
            row.append(1 if rng.random() < prob_1 else 0)
        matrix.append(row)

    fk = suite.fleiss_kappa(matrix, num_classes=2)
    ka = suite.krippendorff_alpha(matrix, metric="nominal")
    ds = suite.dawid_skene_em(matrix)

    r1_err = sum(1 for i in range(n_eval) if matrix[i][3] != true_labels[i]) / n_eval
    r5_err = sum(1 for i in range(n_eval) if matrix[i][4] != true_labels[i]) / n_eval
    maj_preds = [1 if sum(row) >= 3 else 0 for row in matrix]
    maj_err = sum(1 for i in range(n_eval) if maj_preds[i] != true_labels[i]) / n_eval
    ds_err = sum(1 for i in range(n_eval) if ds.predicted_labels[i] != true_labels[i]) / n_eval

    print(f"Lab 13.1 IAA: Fleiss_p_bar={fk.raw_agreement:.4f}, Fleiss_pe={fk.expected_agreement:.4f}, Fleiss_k={fk.statistic:.4f}, Krippendorff_alpha={ka:.4f}")
    print(f"Lab 13.1 Errors: Rater4_err={r1_err:.4f}, SpamRater5_err={r5_err:.4f}, MajorityVote_err={maj_err:.4f}, DawidSkene_err={ds_err:.4f} (iters={ds.iterations})")
    print("DS Estimated Sensitivities:", [round(x, 3) for x in ds.rater_sensitivities])
    print("DS Estimated Specificities:", [round(x, 3) for x in ds.rater_specificities])

    # 3. Queueing & Fatigue Verification across utilization levels
    q_sim = HITLQueueFatigueSimulator()
    for lam, c_srv, mins in [(0.025, 1, 30.0), (0.040, 1, 60.0), (0.045, 1, 90.0), (0.048, 1, 120.0), (0.090, 2, 60.0), (0.135, 3, 45.0)]:
        res = q_sim.evaluate_station(lam, mean_service_sec=20.0, service_std_sec=30.0, num_servers=c_srv, shift_minutes=mins)
        print(
            f"Queue lam={lam:.3f}, c={c_srv}, rho={res.utilization_rho:.2f}, "
            f"MM1_Wq={res.mm1_wait_sec:.2f}s, MG1_PK_Wq={res.mg1_pk_wait_sec:.2f}s, "
            f"MMc_Wq={res.mmc_wait_sec:.2f}s, EffErr={res.effective_error_rate:.4f}, FalseApp={res.false_approval_rate:.4f}"
        )

    # Self-test assertions
    assert ds_err < maj_err < r5_err
