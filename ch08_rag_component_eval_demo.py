"""
Chapter 8: Component-Level Evaluation of RAG, Citations, and Context Compaction.
Why AI Demos Lie — Companion Verification Module (Python 3.12+, Zero External Dependencies).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimVerification:
    claim_id: str
    entailed_by_context: bool
    cited_chunk_ids: tuple[str, ...]
    truly_supporting_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RAGQueryDiagnostic:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    claim_faithfulness: float
    citation_precision: float
    citation_recall: float
    is_unanswerable: bool
    correctly_abstained: bool


class RAGComponentDiagnosticSuite:
    """Computes retrieval ranking metrics, atomic claim faithfulness, citation precision/recall, and NIAH decay."""

    @staticmethod
    def retrieval_metrics(
        retrieved_ids: list[str],
        gold_relevance: dict[str, int],  # chunk_id -> graded relevance in {0, 1, 2, 3}
        k: int,
    ) -> tuple[float, float, float, float]:
        top_k = retrieved_ids[:k]
        relevant_gold = {cid for cid, rel in gold_relevance.items() if rel > 0}
        if not relevant_gold:
            return 1.0, 0.0, 0.0, 1.0

        hits = sum(1 for cid in top_k if cid in relevant_gold)
        recall_k = hits / len(relevant_gold)
        precision_k = hits / k if k > 0 else 0.0

        mrr = 0.0
        for idx, cid in enumerate(top_k):
            if cid in relevant_gold:
                mrr = 1.0 / (idx + 1)
                break

        dcg = 0.0
        for idx, cid in enumerate(top_k):
            rel = gold_relevance.get(cid, 0)
            dcg += (2**rel - 1.0) / math.log2(idx + 2.0)

        ideal_rels = sorted((rel for rel in gold_relevance.values() if rel > 0), reverse=True)[:k]
        idcg = sum((2**rel - 1.0) / math.log2(idx + 2.0) for idx, rel in enumerate(ideal_rels))
        ndcg_k = dcg / idcg if idcg > 1e-12 else 0.0
        return recall_k, precision_k, mrr, ndcg_k

    @staticmethod
    def grounding_and_citation_metrics(
        claims: list[ClaimVerification],
    ) -> tuple[float, float, float]:
        if not claims:
            return 1.0, 1.0, 1.0
        entailed = sum(1 for c in claims if c.entailed_by_context)
        faithfulness = entailed / len(claims)

        total_cited = 0
        valid_cited = 0
        total_needed = 0
        covered_needed = 0

        for c in claims:
            supp_set = set(c.truly_supporting_chunk_ids)
            cited_set = set(c.cited_chunk_ids)
            total_cited += len(cited_set)
            valid_cited += len(cited_set & supp_set)
            total_needed += len(supp_set)
            covered_needed += len(cited_set & supp_set)

        cit_precision = valid_cited / total_cited if total_cited > 0 else 0.0
        cit_recall = covered_needed / total_needed if total_needed > 0 else 1.0
        return faithfulness, cit_precision, cit_recall

    @staticmethod
    def context_retention_probability(
        depth_ratio: float, context_tokens: int, compaction_hops: int
    ) -> float:
        """Models U-shaped Lost-in-the-Middle recall surface and exponential compaction decay."""
        # U-shaped quadratic depth penalty: highest at d=0.0 (primacy) and d=1.0 (recency), minimum at d=0.55
        u_shape = 1.0 - 0.52 * (4.0 * depth_ratio * (1.0 - depth_ratio))
        length_penalty = math.exp(-max(0, context_tokens - 8000) / 180000.0)
        compaction_decay = (0.86) ** compaction_hops
        return min(0.995, max(0.05, 0.98 * u_shape * length_penalty * compaction_decay))


def run_lab_8_1(n_queries: int = 300) -> None:
    rng = random.Random(202608)
    configs = [
        ("C1: Naive Fixed-512 Dense", 0.58, 0.51, 0.89, 0.61, 0.34),
        ("C2: Hybrid BM25 + Dense", 0.78, 0.71, 0.91, 0.68, 0.42),
        ("C3: Hybrid + Cross-Encoder", 0.89, 0.85, 0.94, 0.79, 0.55),
        ("C4: C3 + Citation/Abstain Gate", 0.89, 0.85, 0.98, 0.93, 0.92),
    ]

    print("=== CHAPTER 8: LAB 8.1 RAG COMPONENT DIAGNOSTICS (N = 300 MULTI-HOP QUERIES, K = 5) ===")
    print(
        f"{'Configuration':<30} | {'Rec@5':>6} | {'NDCG@5':>6} | {'Faith':>6} | "
        f"{'CitPrec':>7} | {'NRR':>6} | {'E2E Demo':>8} | {'Verified':>8}"
    )
    print("-" * 100)

    for name, base_rec, base_ndcg, base_faith, base_cit_p, base_nrr in configs:
        rec_list, ndcg_list, faith_list, cit_p_list = [], [], [], []
        unanswerable_total = 60  # 20% of 300 queries lack evidence in corpus
        abstained_correct = 0
        e2e_fluent_pass = 0
        strict_verified_pass = 0

        for q in range(n_queries):
            is_unans = q < unanswerable_total
            if is_unans:
                abstained = rng.random() < base_nrr
                if abstained:
                    abstained_correct += 1
                    strict_verified_pass += 1
                    e2e_fluent_pass += 1
                else:
                    # Hallucinated plausible answer passes vibe check 65% of the time!
                    if rng.random() < 0.65:
                        e2e_fluent_pass += 1
            else:
                r5 = min(1.0, max(0.0, rng.gauss(base_rec, 0.08)))
                n5 = min(1.0, max(0.0, rng.gauss(base_ndcg, 0.07)))
                f_score = min(1.0, max(0.0, rng.gauss(base_faith, 0.05)))
                cp_score = min(1.0, max(0.0, rng.gauss(base_cit_p, 0.07)))
                rec_list.append(r5)
                ndcg_list.append(n5)
                faith_list.append(f_score)
                cit_p_list.append(cp_score)

                # Parametric memory masks retrieval misses in E2E demo
                if r5 > 0.65 or rng.random() < 0.55:
                    e2e_fluent_pass += 1
                if r5 >= 0.75 and f_score >= 0.90 and cp_score >= 0.80:
                    strict_verified_pass += 1

        nrr = abstained_correct / unanswerable_total
        print(
            f"{name:<30} | {statistics.fmean(rec_list):6.3f} | {statistics.fmean(ndcg_list):6.3f} | "
            f"{statistics.fmean(faith_list):6.3f} | {statistics.fmean(cit_p_list):7.3f} | "
            f"{nrr:6.3f} | {e2e_fluent_pass/n_queries:8.1%} | {strict_verified_pass/n_queries:8.1%}"
        )

    print("\n=== CONTEXT DEPTH & COMPACTION RETENTION SURFACE (NIAH CONSTRAINT SURVIVAL) ===")
    print(f"{'Depth d':>8} | {'8k (0 Hops)':>12} | {'32k (0 Hops)':>13} | {'64k (2 Hops)':>13} | {'128k (4 Hops)':>14}")
    print("-" * 70)
    for d in [0.05, 0.25, 0.50, 0.75, 0.95]:
        r_8k = RAGComponentDiagnosticSuite.context_retention_probability(d, 8000, 0)
        r_32k = RAGComponentDiagnosticSuite.context_retention_probability(d, 32000, 0)
        r_64k = RAGComponentDiagnosticSuite.context_retention_probability(d, 64000, 2)
        r_128k = RAGComponentDiagnosticSuite.context_retention_probability(d, 128000, 4)
        print(f"{d:8.2f} | {r_8k:12.1%} | {r_32k:13.1%} | {r_64k:13.1%} | {r_128k:14.1%}")


if __name__ == "__main__":
    run_lab_8_1()

    # Self-test assertions
    assert RAGComponentDiagnosticSuite.context_retention_probability(0.50, 128000, 4) < RAGComponentDiagnosticSuite.context_retention_probability(0.05, 8000, 0)
