import math

from lastrag.chunking import MAX_WORDS, split_sections
from lastrag.metrics import bootstrap_ci, paired_bootstrap, per_query


def test_single_gold_found_at_rank_3():
    m = per_query(["x", "y", "g", "z"], ("g",))
    assert m["recall_any@1"] == 0 and m["recall_any@5"] == 1
    assert m["recall_all@5"] == 1
    assert m["mrr"] == 1 / 3
    assert math.isclose(m["ndcg@10"], 1 / math.log2(4))


def test_two_golds_any_vs_all():
    m = per_query(["g1", "a", "b", "c", "d", "e", "g2"], ("g1", "g2"))
    assert m["recall_any@5"] == 1 and m["recall_all@5"] == 0
    assert m["recall_all@10"] == 1
    assert m["mrr"] == 1.0


def test_gold_missing():
    m = per_query(["a", "b"], ("g",))
    assert m["mrr"] == 0 and m["ndcg@10"] == 0 and m["recall_any@20"] == 0
    assert m["first_gold_rank"] == math.inf


def test_bootstrap_brackets_the_mean():
    lo, hi = bootstrap_ci([0, 1] * 100)
    assert lo < 0.5 < hi and hi - lo < 0.2


def test_paired_bootstrap_sees_a_clear_win():
    d = paired_bootstrap([0] * 100, [1] * 60 + [0] * 40)
    assert math.isclose(d["diff"], 0.6) and d["lo"] > 0 and d["p"] < 0.01


def test_sections_merge_small_and_cap_large():
    body = (
        "Intro " * 10
        + "\n## A\n"
        + "a " * 50
        + "\n## B\n"
        + "b " * 5
        + "\n## C\n"
        + "line\n" * 900
    )
    chunks = split_sections(body)
    assert (
        chunks[0].startswith("Intro") and "## A" in chunks[0]
    )  # 10-word intro merged forward
    assert "## B" in chunks[1] and "## C" in chunks[1]  # 5-word section merged forward
    assert all(len(c.split()) <= MAX_WORDS + 3 for c in chunks)
    assert all(
        c.startswith("## ") for c in chunks[1:]
    )  # split pieces repeat the heading
