# Human-in-the-Loop DAG Audit

## Sample: gsm8k_train_00000

**Question:** Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

**Gold Answer:** 72

### Gold DAG

```
  [0] Natalia sold clips to 48 of her friends in April (given)
  [1] How many clips did Natalia sell altogether? (question)
  [2] 48/2 = 24 clips in May (operation/div)
  [3] 48+24 = 72 clips total (conclusion/add)
  [4] #### 72 (final answer)

Edges: 0→2, 0→3, 2→3, 3→4
Start: [0], Goal: 4
```

### Model Output (deepseek-chat, temperature=0.3)

```
Step 1: 48 clips sold in April
Step 2: Half of 48 = 48÷2 = 24 clips sold in May
Step 3: Total clips = 48 + 24 = 72
Final Answer: 72
```

### Manual Step-to-Node Mapping

| Model Step | Gold Node | Confidence | Rationale |
|---|---|---|---|
| "48 clips sold in April" | **0** | 1.0 | Both state the given fact — April has 48 clips. Gold node 0 is the only node with "48 clips in April". |
| "Half of 48 = 48÷2 = 24" | **2** | 1.0 | Both compute 48/2=24. This is the unique defining operation for the intermediary "24". |
| "Total = 48+24 = 72" | **3** | 1.0 | Both sum 48+24 to get 72. This is the conclusion step. |
| "Final Answer: 72" | **4** | 1.0 | Both state the final answer 72. |

### Verification

- All 4 steps correctly mapped to unique gold nodes ✅
- Logical flow preserved: 0(given) → 2(div) → 3(add) → 4(answer) ✅
- No false positives: each mapping is one-to-one and logically correct ✅
- Depth should be 100 (all nodes on path lit) ✅
- Consistency should be high (no contradictions, no jumps, no redundancy) ✅

### Expected Scores

```
correct=True, depth=100, consistency≈90-100, lit=4/5 (node 1 is the question, not a reasoning step)
```

### Conclusion

This is a **perfect match**. The LLM mapper should handle this trivially — numbers 48,24,72 are uniquely defined by nodes 0,2,3 respectively. Node 1 (the question restatement) may remain unvisited, which is correct behavior — the question is not a reasoning step.

---

## Summary of Audit Findings

| Sample | Correct | Depth | Lit/Total | Issue |
|---|---|---|---|---|
| gsm8k_train_00000 | Yes | 100 | 4/5 | Node 1 (question) unvisited — correct |
| omni_math_test_00001 | Yes | 60 | 2/17 | Sentence-level granularity mismatch |

### Design Decision

Node 1 ("How many clips...") should not be a gold DAG node — it's the question, not a reasoning step. Omni-MATH pre-built graphs have the same issue: question sentences are mixed with solution sentences. This can be fixed in graph construction (marking question nodes as "context" type) but is not critical for scoring accuracy since unvisited question nodes don't affect depth calculations.
