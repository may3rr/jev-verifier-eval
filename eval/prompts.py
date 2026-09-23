"""Prompt and state construction for every system under comparison.

All systems receive the same information: the claim, the evidence records, and
the decision space. Only the serialisation and the reasoning budget differ,
which is what makes the compact/expanded x direct/reasoning comparison
interpretable.

Note: the executed pilot rendered evidence with per-condition character caps
(2500 for direct-compact, 3000 for reason-compact, 6000 for the expanded
conditions and JEV) because ``max_chars`` was not threaded through. The
functions below take an explicit ``max_chars`` so future runs can hold the
evidence identical across conditions.
"""
from __future__ import annotations

import json

LABELS = ["NEI", "SUPPORTS", "REFUTES"]
DIGIT_TO_LABEL = {"0": "NEI", "1": "SUPPORTS", "2": "REFUTES"}
LABEL_TO_DIGIT = {v: k for k, v in DIGIT_TO_LABEL.items()}

DEFINITIONS = (
    "0 = NOT ENOUGH INFO: the evidence is on topic but does not settle the claim.\n"
    "1 = SUPPORTS: the evidence, taken together, entails the claim.\n"
    "2 = REFUTES: the evidence, taken together, contradicts the claim."
)

RULES = (
    "Rules:\n"
    "- Judge only from the supplied evidence. Outside knowledge is not evidence.\n"
    "- NEI is evidence insufficiency, never a synonym for contradiction.\n"
    "- A claim that overstates scope, modality, time, population, or quantity is not supported.\n"
    "- Numeric, unit, negation, entity, and temporal mismatches are contradictions only when the "
    "evidence actually addresses the same slot.\n"
    "- Several evidence records may need to be combined; a combination counts only if the "
    "combination itself is stated or logically entailed."
)

FEW_SHOT = (
    "Example 1\n"
    "Evidence: [E1] The bylaw applies to practitioners registered in this province.\n"
    "Claim: Every province requires practitioners to be at least 60 years old.\n"
    "Answer: 2\n\n"
    "Example 2\n"
    "Evidence: [E1] The trial enrolled 120 adults and reported a 12% response rate.\n"
    "Claim: The treatment improves response rates in adults.\n"
    "Answer: 0\n"
)


def _fewshot(allowed: tuple[str, ...]) -> str:
    """Few-shot block filtered to examples whose answer digit is admissible.

    The shipped prompt always embedded both examples, so on binary datasets the
    worked examples demonstrated the out-of-space digit ``0`` -- the label-safe
    variant keeps only examples whose answer is inside ``allowed``.
    """
    digits = {LABEL_TO_DIGIT[label] for label in LABELS if label in allowed}
    return "".join(
        block + ("\n" if not block.endswith("\n\n") else "")
        for block in FEW_SHOT.split("\n\n")
        if any(f"Answer: {d}" in block for d in digits)
    )


def render_evidence(evidence: list[str], max_chars: int = 6000) -> str:
    lines: list[str] = []
    budget = max_chars
    for index, sentence in enumerate(evidence, start=1):
        text = sentence.strip()
        if not text:
            continue
        if len(text) > budget:
            text = text[:budget] + " ..."
        budget -= len(text)
        lines.append(f"[E{index}] {text}")
        if budget <= 0:
            break
    return "\n".join(lines)


def _task_block(claim: str, evidence: str) -> str:
    return f"Evidence:\n{evidence}\n\nClaim:\n{claim}"


def _definition_block(allowed: tuple[str, ...]) -> str:
    lines = [
        f"{LABEL_TO_DIGIT[label]} = {LABEL_LONG[label]}: {LABEL_HELP[label]}"
        for label in LABELS
        if label in allowed
    ]
    return "\n".join(lines)


LABEL_LONG = {"NEI": "NOT ENOUGH INFO", "SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"}
LABEL_HELP = {
    "NEI": "the evidence is on topic but does not settle the claim",
    "SUPPORTS": "the evidence, taken together, entails the claim",
    "REFUTES": "the evidence, taken together, contradicts the claim",
}


def _digits(allowed: tuple[str, ...]) -> str:
    return " or ".join(LABEL_TO_DIGIT[label] for label in LABELS if label in allowed)


def direct_messages(claim: str, evidence: str, expanded: bool, allowed: tuple[str, ...] = tuple(LABELS),
                    max_chars: int = 6000) -> list[dict]:
    definitions = _definition_block(allowed)
    digits = _digits(allowed)
    if expanded:
        user = (
            "You are a claim verification service. Decide whether the evidence entails the claim.\n"
            f"{definitions}\n\n{RULES}\n\n{FEW_SHOT}"
            f"{_task_block(claim, render_evidence(evidence, max_chars))}\n\n"
            f"Reply with exactly one character: {digits}."
        )
    else:
        user = (
            f"Decide one digit. {definitions}\n"
            f"{_task_block(claim, render_evidence(evidence, max_chars))}\n"
            f"Answer with one character only ({digits}):"
        )
    return [{"role": "user", "content": user}]


def cot_messages(claim: str, evidence: str, expanded: bool, allowed: tuple[str, ...] = tuple(LABELS),
                 max_chars: int = 6000, label_safe: bool = False) -> list[dict]:
    definitions = _definition_block(allowed)
    digits = _digits(allowed)
    fewshot = _fewshot(allowed) if label_safe else FEW_SHOT
    if expanded:
        user = (
            "You are a careful claim verification service. Work through the evidence before deciding.\n"
            f"{definitions}\n\n{RULES}\n\n{fewshot}"
            "Procedure:\n"
            "1. Decompose the claim into atomic assertions (object, predicate, quantity, scope, time).\n"
            "2. For each atom, state whether the evidence supports, contradicts, or is silent, and quote "
            "the shortest decisive span.\n"
            "3. Combine the atom verdicts into one decision, checking scope and modality explicitly.\n"
            "4. Write the reasoning first, then the decision. Keep the reasoning under 120 words.\n\n"
            f"{_task_block(claim, render_evidence(evidence, max_chars))}\n\n"
            f"End with a final line exactly of the form 'FINAL: <{digits}>'."
        )
    else:
        user = (
            "Think step by step, then decide.\n"
            f"{definitions}\n"
            f"{_task_block(claim, render_evidence(evidence, max_chars))}\n\n"
            f"Give a short reason, then a final line exactly of the form 'FINAL: <{digits}>'."
        )
    return [{"role": "user", "content": user}]


def jev_payload(claim: str, evidence: list[str], max_chars: int = 6000, allowed: tuple[str, ...] = tuple(LABELS)) -> tuple[dict, dict]:
    """JEV takes structured state plus typed questions and returns probabilities."""
    state = {
        "claim": claim,
        "evidence": render_evidence(evidence, max_chars),
        "label_space": {k: v for k, v in DIGIT_TO_LABEL.items() if v in allowed},
    }
    questions = {
        "supports": {
            "type": "noul",
            "instructions": (
                "Given only the supplied evidence, does the evidence entail the claim as written? "
                "Ignore any fact that is not in the evidence."
            ),
            "criteria": {
                "true": "The evidence, possibly combining several records, entails the claim.",
                "false": "The evidence does not entail the claim, or contradicts it.",
            },
        },
    }
    if "REFUTES" in allowed:
        questions["refutes"] = {
            "type": "noul",
            "instructions": (
                "Given only the supplied evidence, does the evidence contradict the claim as written? "
                "Only answer true when the evidence addresses the same object and slot."
            ),
            "criteria": {
                "true": "The evidence states something incompatible with the claim.",
                "false": "The evidence is compatible with the claim, silent, or merely insufficient.",
            },
        }
    return state, questions


def parse_direct(text: str) -> str | None:
    for char in text or "":
        if char in DIGIT_TO_LABEL:
            return DIGIT_TO_LABEL[char]
    return None


def parse_cot(text: str) -> str | None:
    """The reasoning protocol requires a FINAL line; anything else is a protocol
    failure. We deliberately do NOT fall back to scanning the reasoning text for
    digits: a bare '1' or '2' in the prose is not a decision."""
    lowered = (text or "").lower()
    marker = lowered.rfind("final")
    if marker == -1:
        return None
    tail = text[marker:]
    for char in tail:
        if char in DIGIT_TO_LABEL:
            return DIGIT_TO_LABEL[char]
    return None
