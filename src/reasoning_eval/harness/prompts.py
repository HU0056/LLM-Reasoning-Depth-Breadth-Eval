"""Prompt templates for LLM agents — with few-shot examples and justification.

Each agent's task is narrow and specific.  Few-shot examples teach the model
the EXACT output format AND the annotation standards simultaneously.
"""

# ═══════════════════════════════════════════════════════
# Structurer — declare steps + dependencies + justifications
# ═══════════════════════════════════════════════════════

STRUCTURER_SYSTEM = """You are a mathematical reasoning analyst. Your task:
decompose a math problem's reference solution into ATOMIC reasoning steps.
For each step, declare what prior steps it depends on, AND provide the
MATHEMATICAL JUSTIFICATION for each dependency.

CRITICAL RULES:
1. Each step must be ONE atomic unit of reasoning — ONE calculation OR ONE deduction.
2. "depends_on" lists ALL previous-step indices this step LOGICALLY requires.
3. For EACH element in "depends_on", there MUST be a matching entry in
   "justifications" (same order) specifying the MATHEMATICAL BASIS.
4. Every calculation step MUST include an "expression" field.
5. The justification type must be EXACTLY one of:
   - "arithmetic": basic + - × ÷
   - "algebra": substitution, equation solving, factoring, algebraic manipulation
   - "theorem": application of a named theorem
   - "axiom": fundamental logical/mathematical axiom
   - "definition": unfolding a definition
   - "simplification": reducing to canonical form
   - "substitution": replacing equals with equals
   - "equivalence": logical equivalence
   - "induction": mathematical induction
6. For "theorem" type, always provide the theorem name in "reference".
7. Step types: given / operation / fact / conclusion / verification.
8. Every step MUST contribute to reaching the final answer. No irrelevant steps."""

STRUCTURER_USER = """Decompose this math problem and its reference solution
into atomic reasoning steps with explicit dependency declarations and
mathematical justifications.

## PROBLEM
{question}

## REFERENCE SOLUTION
{answer}

## FORMAT
Output JSON:
{{
  "steps": [
    {{
      "index": 0,
      "text": "...",
      "depends_on": [],
      "node_type": "given",
      "justifications": []
    }},
    {{
      "index": 1,
      "text": "...",
      "depends_on": [0],
      "node_type": "operation",
      "expression": "48/2=24",
      "justifications": [
        {{"type": "arithmetic", "reference": "division", "is_atomic": true}}
      ]
    }},
    ...
  ],
  "final_answer": "72"
}}

## FEW-SHOT EXAMPLES

### Example 1: Simple Arithmetic
PROBLEM: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
REFERENCE: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

OUTPUT:
{{
  "steps": [
    {{
      "index": 0,
      "text": "Natalia sold 48 clips in April.",
      "depends_on": [],
      "node_type": "given",
      "justifications": []
    }},
    {{
      "index": 1,
      "text": "She sold half as many in May, so May sales = 48/2 = 24 clips.",
      "depends_on": [0],
      "node_type": "operation",
      "expression": "48/2=24",
      "justifications": [
        {{"type": "arithmetic", "reference": "division", "is_atomic": true}}
      ]
    }},
    {{
      "index": 2,
      "text": "Total clips = 48 + 24 = 72.",
      "depends_on": [0, 1],
      "node_type": "conclusion",
      "expression": "48+24=72",
      "justifications": [
        {{"type": "arithmetic", "reference": "addition", "is_atomic": true}},
        {{"type": "arithmetic", "reference": "addition", "is_atomic": true}}
      ]
    }}
  ],
  "final_answer": "72"
}}

### Example 2: Multi-step with External Knowledge
PROBLEM: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
REFERENCE: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

OUTPUT:
{{
  "steps": [
    {{
      "index": 0,
      "text": "Weng earns $12 per hour.",
      "depends_on": [],
      "node_type": "given",
      "justifications": []
    }},
    {{
      "index": 1,
      "text": "She worked 50 minutes.",
      "depends_on": [],
      "node_type": "given",
      "justifications": []
    }},
    {{
      "index": 2,
      "text": "Earnings per minute = 12/60 = $0.20.",
      "depends_on": [0],
      "node_type": "operation",
      "expression": "12/60=0.2",
      "justifications": [
        {{"type": "arithmetic", "reference": "division", "is_atomic": true}}
      ]
    }},
    {{
      "index": 3,
      "text": "Total earnings = 0.2 * 50 = $10.",
      "depends_on": [1, 2],
      "node_type": "conclusion",
      "expression": "0.2*50=10",
      "justifications": [
        {{"type": "arithmetic", "reference": "multiplication", "is_atomic": true}},
        {{"type": "arithmetic", "reference": "multiplication", "is_atomic": true}}
      ]
    }}
  ],
  "final_answer": "10"
}}"""


# ═══════════════════════════════════════════════════════
# Auditor — verify edges and justifications
# ═══════════════════════════════════════════════════════

AUDITOR_SYSTEM = """You are a mathematical logic auditor. You verify two things
about each declared dependency edge in a reasoning DAG:

1. LOGICAL VALIDITY: Does the target step genuinely REQUIRE the source step?
   Could the target be derived WITHOUT the source?  If yes, the edge is spurious.

2. JUSTIFICATION CORRECTNESS: Is the declared mathematical justification
   appropriate for this inference?  Could a different justification be needed?

For each edge, output a verdict.  Also scan for MISSING dependencies — steps
that SHOULD depend on a prior step but don't declare it.

ERROR CATEGORIES:
- "none": edge is valid and justified
- "wrong_premise": target doesn't actually depend on this source
- "missing_step": an intermediate step is missing between source and target
- "irrelevant": the source is topically similar but logically unnecessary
- "circular": the dependency creates or contributes to a cycle
- "wrong_justification": the declared justification type is incorrect
- "insufficient_justification": the justification is too weak for this inference"""

AUDITOR_USER = """Audit every declared dependency edge and justification in this
reasoning DAG.  Also scan for MISSING dependencies.

## PROBLEM
{question}

## REASONING STEPS
{steps_text}

## DECLARED EDGES
{edges_text}

Output JSON:
{{
  "verdicts": [
    {{
      "edge_source": "step_0",
      "edge_target": "step_2",
      "valid": true,
      "confidence": 0.95,
      "error_category": "none",
      "justification_ok": true,
      "suggestion": ""
    }},
    ...
  ],
  "missing_edges": [[source_idx, target_idx], ...],
  "overall_quality": 0.85
}}"""


# ═══════════════════════════════════════════════════════
# Repairer — fix all detected issues
# ═══════════════════════════════════════════════════════

REPAIRER_SYSTEM = """You are a reasoning graph repair specialist. Given a DAG
with detected errors (from code verification and audit), produce a CORRECTED
version.  Apply ALL of these fixes:

1. Remove spurious edges (declared but not logically needed).
2. Add missing edges (detected by use-def, audit, or topology checks).
3. Fix wrong justifications — match the justification type to the actual operation.
4. Remove non-contributing steps — any step not on a path from a given to the conclusion.
5. Ensure all justifications are appropriate for their operations.
6. Ensure the DAG is acyclic and every node connects to the conclusion.

Output the FULL corrected structured solution.  Do NOT output a partial fix."""

REPAIRER_USER = """Repair the following reasoning DAG based on all detected issues.

## PROBLEM
{question}

## CURRENT SOLUTION
```json
{current_solution}
```

## VERIFICATION REPORT
{verification_report}

## AUDIT REPORT
{audit_report}

## CROSS-VALIDATION
{cross_validation}

Output the FULL corrected structured solution as JSON with schema:
{{"steps": [...], "final_answer": "..."}}"""


# ═══════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES (loaded at module level for reuse)
# ═══════════════════════════════════════════════════════

STRUCTURER_FEWSHOT = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
        "output": {
            "steps": [
                {"index": 0, "text": "Natalia sold 48 clips in April.", "depends_on": [], "node_type": "given", "justifications": []},
                {"index": 1, "text": "She sold half as many in May: 48/2 = 24 clips.", "depends_on": [0], "node_type": "operation", "expression": "48/2=24", "justifications": [{"type": "arithmetic", "reference": "division", "is_atomic": True}]},
                {"index": 2, "text": "Total clips = 48 + 24 = 72.", "depends_on": [0, 1], "node_type": "conclusion", "expression": "48+24=72", "justifications": [{"type": "arithmetic", "reference": "addition", "is_atomic": True}, {"type": "arithmetic", "reference": "addition", "is_atomic": True}]},
            ],
            "final_answer": "72",
        },
    },
]
