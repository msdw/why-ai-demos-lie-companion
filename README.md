# Companion Verification Vault — *Why AI Demos Lie*

**Book Title:** *Why AI Demos Lie: Statistical Evaluation, State Oracles, Fault Injection, and Post-Mortems for Production LLMs and AI Agents*  
**Author:** **Mathurin Aché Arnaud** — *Kaggle Competitions Grandmaster (Peak Rank #6, Datasets Legacy Grandmaster, Notebooks Expert, 2x WiDS Datathon Winner) · Senior Data Scientist, AdvanThink · former VP Machine Learning, Prevision.io · Orange France (2005–2019)*  
**Repository URL:** [https://github.com/msdw/why-ai-demos-lie-companion](https://github.com/msdw/why-ai-demos-lie-companion)  
**License:** MIT License (Source Code) · Copyright © 2026 Mathurin Aché Arnaud (Prose & Taxonomy)

---

## 1. Quick Start: Run the Full 17-Module Verification Suite

Every equation, statistical confidence bound, bias correction estimator, state-mutation oracle, chaos fault injector, and release gate in *Why AI Demos Lie* is verified by a standalone, zero-dependency Python 3.12+ module in this repository.

Run the master verification runner in Python's isolated mode (`-I`):

```bash
python -I run_all_verifications.py
```

---

## 2. Complete Inventory of Executable Reference Modules

| Module | Chapter | Core Primitives & Mathematical Theorems Verified |
| :--- | :--- | :--- |
| `ch01_pass_at_k_demo.py` | Chapter 1 | Combinatorial $\text{pass}@k$ vs. consecutive $\text{pass}^k = (p_{\text{step}}^m)^k$, Beta-Binomial task heterogeneity, and Markov step correlation. |
| `ch02_confidence_and_power_demo.py` | Chapter 2 | Wald failure near boundaries, Wilson score intervals, Clopper-Pearson exact Beta bounds, Rule of Three ($3/n$), and MDE power analysis. |
| `ch03_paired_testing_anova_demo.py` | Chapter 3 | Paired McNemar exact test on discordant pairs $(b, c)$, Bowker's symmetry test, Cochran's $Q$, and multi-factor ANOVA variance decomposition. |
| `ch04_leakage_detector_demo.py` | Chapter 4 | 7 families of evaluation leakage, $n$-gram canary contamination scanners, pre-split vs. post-split pipelines, and multiple-testing optimism bias. |
| `ch05_state_oracles_demo.py` | Chapter 5 | Point-in-Time relational state oracles $\mathcal{O}_t(S_{\text{final}})$, double-entry ledger conservation, and verbal-claim vs. state-diff audits. |
| `ch06_calibrated_llm_judge_demo.py` | Chapter 6 | Position-debiased judge swapping, Rogan-Gladen prevalence correction $\hat{p}$, multivariate delta-method variance, and Youden's $J$ gates. |
| `ch07_bradley_terry_ranking_demo.py` | Chapter 7 | Order-independent Bradley-Terry-Davidson MLE (ties & position bias $\gamma$), bootstrap covariance, and non-transitive Condorcet cycle detection. |
| `ch08_rag_component_eval_demo.py` | Chapter 8 | Retrieval Recall@K, MRR, NDCG@K, atomic claim faithfulness, citation verification, and context compaction survival curves. |
| `ch09_trajectory_auditor_demo.py` | Chapter 9 | $2 \times 2$ Outcome-Trajectory matrix, step optimality ratio, tool-selection precision/recall, and oscillatory loop-cycle entropy. |
| `ch10_latency_and_ecvt_demo.py` | Chapter 10 | TTFT/ITL decomposition, Fenton-Wilkinson Log-Normal tail compounding, quadratic context token growth, and Effective Cost per Verified Task (ECVT). |
| `ch11_chaos_fault_injector_demo.py` | Chapter 11 | Deterministic PRNG chaos proxy injecting HTTP 429/502 storms, ambiguous post-commit timeouts, schema drift, and stale cache reads. |
| `ch12_adversarial_redteam_oracle_demo.py` | Chapter 12 | Cryptographic canary exfiltration oracles (`CANARY-UUID-HMAC`), stratified OWASP ASI01–ASI10 red-teaming, and ASR vs. FRR Pareto frontiers. |
| `ch13_annotator_and_hitl_math_demo.py` | Chapter 13 | Cohen's $\kappa$, Fleiss' $\kappa$, Krippendorff's $\alpha$, Dawid-Skene EM latent truth estimation, and Pollaczek-Khinchine $M/G/1$ reviewer fatigue queues. |
| `ch14_drift_and_cusum_demo.py` | Chapter 14 | Population Stability Index (PSI), exact 1D Wasserstein distance, Kolmogorov-Smirnov $D_{\text{KS}}$, Jensen-Shannon divergence, and Page's CUSUM. |
| `ch15_forensic_replay_and_postmortem_demo.py` | Chapter 15 | Tamper-evident SHA-256 hash-chained forensic telemetry, deterministic cassette replay, and S0–S3 blameless post-mortem generation. |
| `ch16_evaluation_release_gate_demo.py` | Chapter 16 | Cryptographic 6-tuple `ReleaseBundleManifest` and automated 4-Stage `EvaluationReleaseGate` certification engine. |
| `appendix_d_failmetry_engine_demo.py` | Appendix D | Unified `FailMetry` reference engine combining statistical inference, Rogan-Gladen calibration, state oracles, chaos survival, and release gating. |
| `claims.csv` | All | Machine-readable ledger of all 68 quantitative theorems, formulas, sampling parameters, PRNG seeds, and empirical claims verified by the suite. |
| `sources.csv` | All | Machine-readable bibliographic registry of all 70 literature entries across Categories I, II, and III with canonical URLs and DOIs. |

