from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from primal_dual_qg.metrics import exact_match_score, token_f1_score, lexical_novelty


def test_exact_match_uses_squad_normalization():
    assert exact_match_score("The Eiffel Tower!", "eiffel tower") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1_score("New York City", "New York")
    assert 0.79 < score < 0.81


def test_lexical_novelty():
    score = lexical_novelty(
        "Which scientist discovered penicillin?",
        "Alexander Fleming discovered penicillin in 1928.",
        "Alexander Fleming",
    )
    assert 0.0 < score < 1.0
