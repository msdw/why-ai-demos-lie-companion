"""Chapter 11: Chaos Engineering for AI Systems: Seeded Fault Injection for Tools, APIs, and Context.

Deterministic reference implementation of FaultInjectionHarness and Lab 11.1
across 1,200 seeded fault-injection runs (6 fault modes x 200 trials).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

FaultMode = Literal[
    "HTTP_429_RATE_LIMIT",
    "HTTP_503_STORM",
    "AMBIGUOUS_POST_COMMIT_TIMEOUT",
    "TRUNCATED_JSON_STREAM",
    "SCHEMA_KEY_DRIFT",
    "STALE_READ_AFTER_WRITE",
]

OutcomeClass = Literal[
    "SURVIVED_RECOVERED",
    "SAFE_HALT_ESCALATED",
    "UNHANDLED_CRASH_LOOP",
    "SILENT_STATE_CORRUPTION",
]


@dataclass
class LedgerSandboxState:
    balance_cents: int = 100_000
    applied_tx_ids: list[str] = field(default_factory=list)
    idempotency_keys: set[str] = field(default_factory=set)
    write_count: int = 0


@dataclass(frozen=True)
class ChaosMetricSummary:
    architecture: str
    fault_mode: str
    trials: int
    psr: float
    sher: float
    crr: float
    raf: float
    sscr: float


class FaultInjectionHarness:
    """PRNG-seeded chaos injector wrapping tool invocations and auditing state invariants."""

    FAULT_MODES: tuple[FaultMode, ...] = (
        "HTTP_429_RATE_LIMIT",
        "HTTP_503_STORM",
        "AMBIGUOUS_POST_COMMIT_TIMEOUT",
        "TRUNCATED_JSON_STREAM",
        "SCHEMA_KEY_DRIFT",
        "STALE_READ_AFTER_WRITE",
    )

    def __init__(self, master_seed: int = 202611) -> None:
        self.master_seed = master_seed

    def simulate_trial(
        self, fault_mode: FaultMode, trial_idx: int, hardened: bool
    ) -> tuple[OutcomeClass, int, int]:
        """Runs a single seeded workflow under the specified fault mode.

        Returns (outcome_class, actual_tool_calls, nominal_tool_calls).
        Nominal workflow requires 3 calls: [read_account, mutate_ledger, verify_ledger].
        """
        mode_offset = sum(ord(c) * (i + 1) for i, c in enumerate(fault_mode))
        rng = random.Random(self.master_seed + mode_offset + trial_idx * 31)
        state = LedgerSandboxState()
        nominal_calls = 3
        actual_calls = 0
        u = rng.random()

        if fault_mode == "HTTP_429_RATE_LIMIT":
            if not hardened:
                # Hammering without backoff exhausts quota or trips rate ban
                actual_calls = 9 if u < 0.78 else 5
                outcome: OutcomeClass = "UNHANDLED_CRASH_LOOP" if u < 0.78 else "SURVIVED_RECOVERED"
                if outcome == "SURVIVED_RECOVERED":
                    state.write_count = 1
            else:
                # Respects Retry-After with jittered backoff
                actual_calls = 4 if u < 0.94 else 5
                outcome = "SURVIVED_RECOVERED" if u < 0.94 else "SAFE_HALT_ESCALATED"
                if outcome == "SURVIVED_RECOVERED":
                    state.write_count = 1

        elif fault_mode == "HTTP_503_STORM":
            if not hardened:
                actual_calls = 11 if u < 0.88 else 8
                outcome = "UNHANDLED_CRASH_LOOP"
            else:
                # Circuit breaker trips after 2 consecutive 503s and halts safely
                if u < 0.32:
                    actual_calls = 5
                    state.write_count = 1
                    outcome = "SURVIVED_RECOVERED"
                else:
                    actual_calls = 4
                    outcome = "SAFE_HALT_ESCALATED"

        elif fault_mode == "AMBIGUOUS_POST_COMMIT_TIMEOUT":
            # Server commits debit of 15,000 cents, then drops TCP connection before response
            state.balance_cents -= 15_000
            state.write_count += 1
            state.idempotency_keys.add("IDEMP-TX-001")
            if not hardened:
                # Naive agent sees TimeoutError and retries mutation with a fresh request
                actual_calls = 5
                if u < 0.84:
                    state.balance_cents -= 15_000
                    state.write_count += 1
                    outcome = "SILENT_STATE_CORRUPTION"
                else:
                    outcome = "UNHANDLED_CRASH_LOOP"
            else:
                # Hardened harness attaches idempotency key & polls ledger state before retry
                actual_calls = 4
                outcome = "SURVIVED_RECOVERED"

        elif fault_mode == "TRUNCATED_JSON_STREAM":
            if not hardened:
                # Unvalidated json/regex fallback misparses partial amount or crashes
                actual_calls = 7 if u < 0.74 else 4
                if u < 0.19:
                    state.write_count = 2
                    outcome = "SILENT_STATE_CORRUPTION"
                elif u < 0.91:
                    outcome = "UNHANDLED_CRASH_LOOP"
                else:
                    state.write_count = 1
                    outcome = "SURVIVED_RECOVERED"
            else:
                # Strict schema parser catches EOF, retries once or halts cleanly
                actual_calls = 4
                if u < 0.86:
                    state.write_count = 1
                    outcome = "SURVIVED_RECOVERED"
                else:
                    outcome = "SAFE_HALT_ESCALATED"

        elif fault_mode == "SCHEMA_KEY_DRIFT":
            if not hardened:
                # Agent hallucinates default 0 or wrong field when key `amount_cents` is missing
                actual_calls = 6
                if u < 0.34:
                    state.write_count = 2
                    outcome = "SILENT_STATE_CORRUPTION"
                elif u < 0.89:
                    outcome = "UNHANDLED_CRASH_LOOP"
                else:
                    outcome = "SAFE_HALT_ESCALATED"
            else:
                # Pydantic/JSON-Schema contract validator halts immediately on drift
                actual_calls = 2
                outcome = "SAFE_HALT_ESCALATED"

        elif fault_mode == "STALE_READ_AFTER_WRITE":
            # Mutation succeeded, but immediate read replica returns stale pre-mutation balance
            state.balance_cents -= 15_000
            state.write_count = 1
            if not hardened:
                # Agent sees unchanged balance on replica and re-runs the mutation!
                actual_calls = 5
                if u < 0.76:
                    state.balance_cents -= 15_000
                    state.write_count += 1
                    outcome = "SILENT_STATE_CORRUPTION"
                else:
                    outcome = "SURVIVED_RECOVERED"
            else:
                # Hardened oracle uses monotonic write-version token / primary-read pinning
                actual_calls = 4
                outcome = "SURVIVED_RECOVERED"

        # Deterministic state invariant assertion
        if state.write_count > 1:
            outcome = "SILENT_STATE_CORRUPTION"

        return outcome, actual_calls, nominal_calls

    def evaluate_suite(self, trials_per_mode: int = 200) -> list[ChaosMetricSummary]:
        summaries: list[ChaosMetricSummary] = []
        for hardened, arch_name in ((False, "Unhardened ReAct Loop"), (True, "Hardened Chaos-Resilient Harness")):
            total_surv = 0
            total_halt = 0
            total_corrupt = 0
            total_actual_calls = 0
            total_nom_calls = 0
            total_runs = len(self.FAULT_MODES) * trials_per_mode

            for mode in self.FAULT_MODES:
                surv = halt = corrupt = act_c = nom_c = 0
                for idx in range(trials_per_mode):
                    out, ac, nc = self.simulate_trial(mode, idx, hardened)
                    act_c += ac
                    nom_c += nc
                    if out == "SURVIVED_RECOVERED":
                        surv += 1
                    elif out == "SAFE_HALT_ESCALATED":
                        halt += 1
                    elif out == "SILENT_STATE_CORRUPTION":
                        corrupt += 1

                psr = surv / trials_per_mode
                sher = halt / trials_per_mode
                sscr = corrupt / trials_per_mode
                raf = act_c / nom_c
                summaries.append(
                    ChaosMetricSummary(
                        arch_name, mode, trials_per_mode, psr, sher, psr + sher, raf, sscr
                    )
                )
                total_surv += surv
                total_halt += halt
                total_corrupt += corrupt
                total_actual_calls += act_c
                total_nom_calls += nom_c

            summaries.append(
                ChaosMetricSummary(
                    arch_name,
                    "ALL_6_MODES_AGGREGATE",
                    total_runs,
                    total_surv / total_runs,
                    total_halt / total_runs,
                    (total_surv + total_halt) / total_runs,
                    total_actual_calls / total_nom_calls,
                    total_corrupt / total_runs,
                )
            )
        return summaries


def main() -> None:
    harness = FaultInjectionHarness(master_seed=202611)
    results = harness.evaluate_suite(trials_per_mode=200)
    print("=== LAB 11.1: 1,200-RUN SEEDED FAULT INJECTION MATRIX ===")
    for r in results:
        print(
            f"{r.architecture:<33} | {r.fault_mode:<30} | N={r.trials:<4d} | "
            f"PSR={r.psr*100:5.1f}% | SHER={r.sher*100:5.1f}% | CRR={r.crr*100:5.1f}% | "
            f"RAF={r.raf:4.2f}x | SSCR={r.sscr*100:5.2f}%"
        )


if __name__ == "__main__":
    main()

    # Self-test assertions
    harness_check = FaultInjectionHarness(master_seed=202611)
    suite_check = harness_check.evaluate_suite(trials_per_mode=50)
    hardened_sscr = [r.sscr for r in suite_check if 'Hardened' in r.architecture]
    assert all(s == 0.0 for s in hardened_sscr)
