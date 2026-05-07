from carl.core.reward import normalize_answer, exact_match, compute_reward, extract_answer


def test_normalize():
    assert normalize_answer("The Apple Pie") == "apple pie"


def test_exact_match_substring_for_entity():
    assert exact_match("The Love Route (1960)", "The Love Route") is True


def test_numeric_match():
    assert exact_match("42.0", "42") is True


def test_compute_reward_extracts_boxed():
    text = "calc... \\boxed{8}"
    assert compute_reward(text, "8", "gsm8k") == 1.0


def test_extract_answer_tag():
    assert extract_answer("...<answer>Edinburgh</answer>...", "musique") == "Edinburgh"
