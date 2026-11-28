"""Agent tables: the selected alpha, peer rows from FL-M3's scoring."""

import numpy as np
import pytest

from filinglens.agents.tool_tables import find_alpha, peer_rows


def test_find_alpha_wherever_the_record_nests_it():
    assert find_alpha({"alpha": 0.9}) == 0.9
    assert find_alpha({"selection": {"hit_rate_by_alpha": {"0.9": 0.87}, "alpha": 0.9}}) == 0.9
    with pytest.raises(KeyError):
        find_alpha({"selection": {"alpha": "0.9"}})
    with pytest.raises(KeyError):
        find_alpha({"flag": True})


def test_peer_rows_rank_others_by_the_mixed_score():
    order = [10, 20, 30]
    z_text = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 0.5], [1.0, 0.5, 0.0]])
    z_ratio = np.array([[0.0, -1.0, 3.0], [-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rows = peer_rows(order, z_text, z_ratio, alpha=0.5, top=2)
    assert len(rows) == 6 and all(cik != peer for cik, _, peer, *_ in rows)
    first = [r for r in rows if r[0] == 10]
    assert [r[2] for r in first] == [30, 20]  # 0.5*1 + 0.5*3 = 2.0 beats 0.5*2 - 0.5 = 0.5
    assert first[0][3] == pytest.approx(2.0) and first[0][1] == 1 and first[0][6] == 0.5
