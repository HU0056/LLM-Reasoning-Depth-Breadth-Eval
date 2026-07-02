"""Harness Pipeline with loop engineering and fatal error semantics.

Key changes from v1:
- LoopState tracks failures and raises HarnessError on exhaustion.
- No silent fallback — if repair fails repeatedly, the pipeline DIES.
- Cross-validation now includes justification-level reconciliation.
- Contribution check ensures every node matters.
"""

from __future__ import annotations

import sys

from reasoning_eval.harness.agents import (
    HarnessParseError,
    run_auditor,
    run_repairer,
    run_structurer,
)
from reasoning_eval.harness.schemas import (
    AuditReport,
    CrossValidationResult,
    DagEdge,
    DagNode,
    EdgeType,
    GoldDag,
    HarnessExhaustedError,
    HarnessError,
    HarnessParseError,
    HarnessVerificationError,
    Justification,
    LoopState,
    StepDeclaration,
    StructuredSolution,
    VerificationReport,
)
from reasoning_eval.harness.verifiers import run_all_checks


class HarnessPipeline:
    """Orchestrates DAG construction with fatal-error semantics.

    Pipeline phases:
      1. Structurer (LLM)  — declare steps + dependencies + justifications
      2. Verification (Code) — 6 deterministic checks
      3. Auditor (LLM)       — semantic edge + justification audit
      4. Cross-Validate (Code) — reconcile LLM vs code
      5. Repair Loop (LLM+Code) — iterative fix until valid or FATAL

    Parameters
    ----------
    client :
        LLMClient instance.
    max_repair_rounds :
        Maximum repair→verify iterations before raising HarnessExhaustedError.
    """

    def __init__(self, client, max_repair_rounds: int = 2) -> None:
        self._client = client
        self._max_repair_rounds = max_repair_rounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, question: str, answer: str) -> GoldDag:
        """Run the full pipeline.  Raises HarnessError on fatal failure."""
        loop = LoopState(max_rounds=self._max_repair_rounds)

        # Phase 1: Structure
        solution = self._run_phase("structurer", lambda: run_structurer(
            question, answer, self._client,
        ))

        # Phase 2: Deterministic verification
        verification = self._run_phase("verification", lambda: run_all_checks(solution))

        # Phase 3: Semantic audit
        audit = self._run_phase("auditor", lambda: run_auditor(
            question, solution, self._client,
        ))

        # Phase 4: Cross-validation
        cross_val = self._run_phase("cross_validate", lambda: self._cross_validate(
            solution, verification,
        ))

        # Phase 5: Repair loop
        solution, verification, audit, cross_val = self._repair_loop(
            question, solution, verification, audit, cross_val, loop,
        )

        # Phase 6: Final verification gate — if still failing, die
        if not verification.all_passed:
            raise HarnessVerificationError(
                f"Final verification failed after {loop.round_number} repair rounds: "
                f"{verification.summary}",
                phase="final_verification",
                detail={"summary": verification.summary},
            )

        return self._build_gold_dag(solution, verification, audit, cross_val)

    # ------------------------------------------------------------------
    # Internal: safe phase runner
    # ------------------------------------------------------------------

    @staticmethod
    def _run_phase(name: str, fn):
        """Run a phase, catching HarnessError and annotating it with phase."""
        try:
            return fn()
        except HarnessError:
            raise
        except Exception as e:
            raise HarnessError(
                f"Phase '{name}' failed: {e}", phase=name,
            ) from e

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(
        self,
        solution: StructuredSolution,
        verification: VerificationReport,
    ) -> CrossValidationResult:
        llm_edges: set[tuple[int, int]] = {
            (dep, s.index)
            for s in solution.steps
            for dep in s.depends_on
        }
        code_edges: set[tuple[int, int]] = set()
        for check in verification.use_def:
            if check.defined_in_step is not None:
                code_edges.add((check.defined_in_step, check.step_index))

        agreed = llm_edges & code_edges
        llm_only = llm_edges - code_edges
        code_only = code_edges - llm_edges

        resolved: list[DagEdge] = []
        for src, tgt in agreed | code_only:
            # Find the justification for this edge, if any
            just = Justification.arithmetic()
            for s in solution.steps:
                if s.index == tgt:
                    for i, dep in enumerate(s.depends_on):
                        if dep == src and i < len(s.justifications):
                            just = s.justifications[i]
                            break
                    break

            resolved.append(DagEdge(
                premises=[f"step_{src}"],
                target=f"step_{tgt}",
                edge_type=EdgeType.INFER,
                justification=just,
                rationale=(
                    "verified by use-def analysis"
                    if (src, tgt) in code_edges
                    else "LLM declared, use-def consistent"
                ),
            ))

        unresolved = [
            f"LLM declared step_{src}→step_{tgt} but use-def didn't detect it"
            for src, tgt in llm_only
        ]

        return CrossValidationResult(
            llm_edges=len(llm_edges),
            code_edges=len(code_edges),
            agreed_edges=sorted(agreed),
            llm_only_edges=sorted(llm_only),
            code_only_edges=sorted(code_only),
            resolved_edges=resolved,
            unresolved_conflicts=unresolved,
        )

    # ------------------------------------------------------------------
    # Repair loop — with loop engineering
    # ------------------------------------------------------------------

    def _repair_loop(
        self,
        question: str,
        solution: StructuredSolution,
        verification: VerificationReport,
        audit: AuditReport,
        cross_val: CrossValidationResult,
        loop: LoopState,
    ) -> tuple[
        StructuredSolution,
        VerificationReport,
        AuditReport,
        CrossValidationResult,
    ]:
        """Iteratively repair until valid or fatal exhaustion."""
        import time

        while True:
            # Termination condition
            if verification.all_passed and audit.invalid_edge_count == 0:
                return solution, verification, audit, cross_val

            # Exhaustion check
            loop.round_number += 1
            if loop.round_number > loop.max_rounds:
                raise HarnessExhaustedError(
                    f"Repair loop exhausted after {loop.max_rounds} rounds",
                    phase="repair",
                    detail={"summary": verification.summary},
                )

            print(
                f"[harness] Repair round {loop.round_number}/{loop.max_rounds} — "
                f"verification={'PASS' if verification.all_passed else 'FAIL'}, "
                f"audit_invalid={audit.invalid_edge_count}",
                file=sys.stderr,
            )

            # Attempt repair — timeout = FATAL
            try:
                t0 = time.time()
                solution = run_repairer(
                    question, solution, verification, audit, cross_val,
                    self._client,
                )
                elapsed = time.time() - t0
                if elapsed > 20:
                    print(f"[harness] Repairer slow: {elapsed:.0f}s", file=sys.stderr)
            except HarnessParseError:
                raise  # JSON failures → fatal immediately
            except HarnessError:
                raise
            except Exception as e:
                msg = str(e).lower()
                if "timeout" in msg or "timed out" in msg or "connection" in msg:
                    raise HarnessError(
                        f"Repairer timeout/network error: {e}",
                        phase="repair",
                    ) from e
                # Non-timeout error: keep solution, let loop decide
                print(f"[harness] Repairer error (non-fatal): {e}", file=sys.stderr)

            # Re-verify
            verification = run_all_checks(solution)

            # Re-audit (non-critical — keep prior on failure)
            try:
                audit = run_auditor(question, solution, self._client)
            except HarnessParseError:
                raise
            except Exception as e:
                print(f"[harness] Auditor failed, keeping prior: {e}", file=sys.stderr)

            cross_val = self._cross_validate(solution, verification)

    # ------------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------------

    def _build_gold_dag(
        self,
        solution: StructuredSolution,
        verification: VerificationReport,
        audit: AuditReport,
        cross_val: CrossValidationResult,
    ) -> GoldDag:
        nodes = [
            DagNode(
                id=f"step_{s.index}",
                type=s.node_type,
                text=s.text,
                expression=s.expression,
            )
            for s in solution.steps
        ]

        if cross_val.resolved_edges:
            edges = cross_val.resolved_edges
        else:
            edges = [
                DagEdge(
                    premises=[f"step_{dep}"],
                    target=f"step_{s.index}",
                    edge_type=EdgeType.INFER,
                    justification=(
                        s.justifications[i] if i < len(s.justifications)
                        else Justification.arithmetic()
                    ),
                    rationale="",
                )
                for s in solution.steps
                for i, dep in enumerate(s.depends_on)
            ]

        return GoldDag(
            nodes=nodes,
            edges=edges,
            num_steps=len(nodes),
            num_edges=len(edges),
            verification_report=verification,
            audit_report=audit,
            cross_validation=cross_val,
        )
