import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "further_analysis"))

import exclusion  # noqa: E402
import pairs  # noqa: E402


def test_left_out_quenchers_are_restored_after_loading():
    saved = dict(pairs.LEFT_OUT_QUENCHERS)
    with pytest.raises(RuntimeError):
        with exclusion.left_out_quenchers_included():
            assert pairs.LEFT_OUT_QUENCHERS == {}
            raise RuntimeError
    assert pairs.LEFT_OUT_QUENCHERS == saved


def test_emission_ratio_uses_the_highest_concentration():
    t = pd.DataFrame({"pc_old_id": 1, "quencher_id": 2, "quencher_conc_M": [0, 0, 0.002, 0.004, 0.004],
                      "intensity_au": [100.0, 110.0, 90.0, 50.0, 60.0]})
    r = exclusion.emission_ratios(t).iloc[0]
    assert r["conc_max / mM"] == pytest.approx(4.0)
    assert r["I0_over_I_at_max"] == pytest.approx(105.0 / 55.0)
