import math
import re
import string


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, strip whitespace/punctuation/articles."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_number(text: str) -> str | None:
    """Try to extract a numeric value from text and normalize it."""
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        val = float(text)
        if not math.isfinite(val):
            return None
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return None


def extract_answer(text: str, dataset: str) -> str | None:
    """Extract the final answer from generated text, dataset-aware.

    Checks for <answer>...</answer> tag first (universal), then falls back
    to dataset-specific patterns.
    """
    if not text:
        return None

    # Universal: <answer>...</answer> tag
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to dataset-specific patterns
    if dataset in ("gsm8k",):
        return _extract_gsm8k(text)
    elif dataset in ("math",):
        return _extract_math(text)
    elif dataset in ("finqa",):
        return _extract_finqa(text)

    # For QA datasets (hotpotqa, nq, musique, 2wiki, triviaqa):
    return _extract_qa_answer(text)


def _extract_gsm8k(text: str) -> str | None:
    """GSM8K: look for #### <number> pattern, or final number."""
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return _extract_last_number(text)


def _extract_math(text: str) -> str | None:
    r"""MATH: look for \boxed{...} pattern, or final expression."""
    # Find the last \boxed{...} (may be nested)
    matches = list(re.finditer(r"\\boxed\{", text))
    if matches:
        start = matches[-1].end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
    return _extract_last_number(text)


def _extract_finqa(text: str) -> str | None:
    """FinQA: look for a final numeric answer, handle percentages."""
    return _extract_last_number(text)


def _extract_qa_answer(text: str) -> str | None:
    """For QA datasets: extract the final answer from the text.

    Looks for patterns like:
    - "The answer is <answer>"
    - "Answer: <answer>"
    - Last sentence/line as fallback
    """
    # "the answer is ..."
    match = re.search(r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")

    # "Answer: ..."
    match = re.search(r"answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".")

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1].rstrip(".")

    return None


def _extract_last_number(text: str) -> str | None:
    """Extract the last number from text."""
    matches = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def exact_match(pred: str, gold) -> bool:
    """Check if prediction matches gold answer(s).

    Gold can be a string or a list of acceptable answers (NQ, TriviaQA).
    """
    if pred is None:
        return False

    if isinstance(gold, list):
        return any(exact_match(pred, g) for g in gold)

    # Try numeric comparison first
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    # Normalized string comparison
    return normalize_answer(pred) == normalize_answer(gold)


def compute_reward(generated_text: str, gold_answer, dataset: str) -> float:
    """Compute binary reward: 1.0 if correct, 0.0 if wrong."""
    pred = extract_answer(generated_text, dataset)
    if pred is None:
        return 0.0
    return 1.0 if exact_match(pred, gold_answer) else 0.0
