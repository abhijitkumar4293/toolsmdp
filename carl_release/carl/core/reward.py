"""EM normalization + binary outcome reward (paper Section 3.1 Reward).

Standard normalization: lowercase, article removal, punctuation strip, whitespace collapse.
For multi-word entity answers, substring containment is also accepted (paper Section 4.1 Evaluation).
"""
import math
import re
import string


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_number(text: str) -> str | None:
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        v = float(text)
        if not math.isfinite(v):
            return None
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return None


def extract_answer(text: str, dataset: str) -> str | None:
    if not text:
        return None
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    boxes = list(re.finditer(r"\\boxed\{([^}]*)\}", text))
    if boxes:
        return boxes[-1].group(1).strip()
    if dataset == "gsm8k":
        m = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if m:
            return m.group(1).strip()
    if dataset in ("gsm8k", "finqa"):
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else None
    # QA fallback
    m = re.search(r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].rstrip(".") if lines else None


def exact_match(pred: str | None, gold) -> bool:
    if pred is None:
        return False
    if isinstance(gold, list):
        return any(exact_match(pred, g) for g in gold)
    pn, gn = normalize_number(pred), normalize_number(str(gold))
    if pn is not None and gn is not None:
        return pn == gn
    np, ng = normalize_answer(pred), normalize_answer(str(gold))
    if np == ng:
        return True
    # Substring acceptance for multi-word entities (paper Section 4.1)
    if ng and ng in np:
        return True
    return False


def compute_reward(generated_text: str, gold_answer, dataset: str) -> float:
    pred = extract_answer(generated_text, dataset)
    if pred is None:
        return 0.0
    return 1.0 if exact_match(pred, gold_answer) else 0.0
