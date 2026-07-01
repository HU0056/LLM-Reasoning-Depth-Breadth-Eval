"""Step-to-DAG-node mapper with anti-fabrication protection.

The core risk: a model step might have high-enough Jaccard similarity to
a gold DAG node purely by accident (shared numbers, common words), leading
to a FALSE mapping.  This is "fabrication" — the model didn't actually
reason about that gold step, but the mapper credits it.

Mitigations (each independently tunable):
1. RAISED confidence threshold (0.35, up from 0.20)
2. BI-DIRECTIONAL check: gold node text must also contain key tokens from model step
3. STRUCTURAL gate: the mapped node must be reachable from the current
   verification state (enforced at the verifier level, but mapper can
   pre-filter candidates)
"""

from __future__ import annotations

from reasoning_eval.common.schema import MappingResult
from reasoning_eval.common.text_utils import jaccard, normalize_text


# ── Tunable thresholds ──

FABRICATION_MIN_CONFIDENCE = 0.35  # was 0.20
BI_DIRECTIONAL_BOOST = 0.15        # added when bidirectional match confirmed
SUSPICIOUS_SCORE_THRESHOLD = 0.25  # below this, treat as NO match
STRUCTURAL_CANDIDATE_BOOST = 0.05  # added for structurally reachable nodes


def _mentions_rule(step_norm: str, source: str, target: str) -> bool:
    return f"{source.lower()}->{target.lower()}" in step_norm.replace(" ", "")


def _token_set(text: str) -> set[str]:
    """Extract numeric and alphabetic tokens for bidirectional matching."""
    import re
    norm = normalize_text(text)
    return set(re.findall(r"[a-z0-9]+|[一-鿿]+", norm))


def _bidirectional_overlap(model_step: str, gold_prop: str) -> float:
    """Check if gold proposition text also matches the model step.

    Returns 1.0 if gold tokens are a subset of model tokens (gold IS in the step),
    0.0 if there's no overlap, and a Jaccard otherwise.
    """
    step_tokens = _token_set(model_step)
    gold_tokens = _token_set(gold_prop)
    if not step_tokens or not gold_tokens:
        return 0.0
    overlap = step_tokens & gold_tokens
    # What fraction of gold tokens appear in the step?
    gold_coverage = len(overlap) / len(gold_tokens) if gold_tokens else 0.0
    return gold_coverage


def map_step_to_node(step_text: str, graph: dict) -> MappingResult:
    """Map a model step to the best-matching gold DAG node.

    Fabrication checks:
    - Confidence < FABRICATION_MIN_CONFIDENCE → no match
    - Bidirectional overlap bonus for genuine matches
    - Suspicious matches (below SUSPICIOUS_SCORE_THRESHOLD) → no match
    """
    step_norm = normalize_text(step_text)
    nodes = graph["nodes"]
    edges = graph.get("edges", [])

    # ── Check 1: exact rule-text match (highest confidence) ──
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        rule_text = normalize_text(edge.get("rule_text", ""))
        if rule_text and (
            _mentions_rule(step_norm, source, target)
            or rule_text in step_norm
        ):
            return MappingResult(
                step_text, target, 0.95,
                f"matched rule text {edge.get('rule_text', '')} → target {target}",
            )

    # ── Check 2: node-by-node scoring with fabrication guard ──
    best_node: str | None = None
    best_conf: float = 0.0
    best_reason = "no proposition or rule matched"
    best_bi_conf: float = 0.0

    for node in nodes:
        prop = node.get("proposition", "")
        node_id = node.get("id", "")
        prop_norm = normalize_text(prop)

        score = 0.0
        reasons: list[str] = []

        # Substring match (strong signal)
        if prop_norm and prop_norm in step_norm:
            score = 0.75
            reasons.append(f"step contains proposition '{prop[:60]}'")
            if "不成立" in step_norm.replace(" ", ""):
                score = 0.60  # negation weakens the match

        # Jaccard similarity (fallback signal)
        jacc = jaccard(step_text, prop)
        if jacc > score:
            score = jacc
            reasons.append(f"Jaccard={jacc:.3f} with '{prop[:60]}'")

        # Bidirectional check: does gold text also match the step?
        bi_overlap = _bidirectional_overlap(step_text, prop)
        if bi_overlap >= 0.5:
            score += BI_DIRECTIONAL_BOOST
            reasons.append(f"bidirectional overlap={bi_overlap:.2f}")

        # Structural boost: is this node known to be reachable?
        # (We can't fully check here without verifier state, but we can check
        #  if the node is a known start node)
        if node.get("type") == "given":
            score += STRUCTURAL_CANDIDATE_BOOST
            reasons.append("start node")

        if score > best_conf:
            best_node = node_id
            best_conf = score
            best_bi_conf = bi_overlap
            best_reason = "; ".join(reasons)

    # ── Fabrication gates ──
    if best_conf < SUSPICIOUS_SCORE_THRESHOLD:
        return MappingResult(
            step_text, None, round(best_conf, 3),
            f"fabrication gate: best score {best_conf:.3f} < "
            f"suspicious threshold {SUSPICIOUS_SCORE_THRESHOLD}",
        )

    if best_conf < FABRICATION_MIN_CONFIDENCE:
        return MappingResult(
            step_text, None, round(best_conf, 3),
            f"fabrication gate: best score {best_conf:.3f} < "
            f"minimum {FABRICATION_MIN_CONFIDENCE}",
        )

    # Additional check: if bidirectional overlap is very low, the match is
    # likely coincidental (shared numbers, not shared reasoning)
    if best_conf < 0.45 and best_bi_conf < 0.3:
        return MappingResult(
            step_text, None, round(best_conf, 3),
            f"fabrication gate: score {best_conf:.3f} with bidirectional "
            f"overlap {best_bi_conf:.2f} — likely coincidental token overlap",
        )

    return MappingResult(
        step_text, best_node, round(best_conf, 3), best_reason,
    )
