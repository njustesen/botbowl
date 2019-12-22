import pytest
from ffai.core.model import D3, D6, D8, BBDie
from ffai.core.table import BBDieResult
import numpy as np


@pytest.mark.parametrize("die", [D3, D6, D8, BBDie])
def test_d_die(die):
    results = []
    n = 6
    if die == D3:
        n = 3
    elif die == D8:
        n = 8
    elif die == BBDieResult:
        n = 5  # Two push results
    for i in range(100):
        rnd = np.random.RandomState(0)
        result = die(rnd).value
        if die == D3:
            assert result in [1, 2, 3]
        elif die == D6:
            assert result in [1, 2, 3, 4, 5, 6]
        elif die == D8:
            assert result in [1, 2, 3, 4, 5, 6, 7, 8]
        elif die == BBDie:
            assert result in [BBDieResult.ATTACKER_DOWN, BBDieResult.BOTH_DOWN, BBDieResult.PUSH, BBDieResult.DEFENDER_STUMBLES, BBDieResult.DEFENDER_DOWN]
        results.append(result)
        if len(results) == n:
            break
    assert len(results) == n


def test_d3_fixation():
    for seed in range(10):
        rnd = np.random.RandomState(seed)
        D3.fix_result(1)
        D3.fix_result(2)
        D3.fix_result(3)
        assert D3(rnd).value == 1
        assert D3(rnd).value == 2
        assert D3(rnd).value == 3
    with pytest.raises(ValueError):
        D3.fix_result(0)
    with pytest.raises(ValueError):
        D3.fix_result(4)


def test_d6_fixation():
    for seed in range(10):
        rnd = np.random.RandomState(seed)
        D6.fix_result(1)
        D6.fix_result(2)
        D6.fix_result(3)
        D6.fix_result(4)
        D6.fix_result(5)
        D6.fix_result(6)
        assert D6(rnd).value == 1
        assert D6(rnd).value == 2
        assert D6(rnd).value == 3
        assert D6(rnd).value == 4
        assert D6(rnd).value == 5
        assert D6(rnd).value == 6
    with pytest.raises(ValueError):
        D6.fix_result(0)
    with pytest.raises(ValueError):
        D6.fix_result(7)


def test_d8_fixation():
    for seed in range(10):
        rnd = np.random.RandomState(seed)
        D8.fix_result(1)
        D8.fix_result(2)
        D8.fix_result(3)
        D8.fix_result(4)
        D8.fix_result(5)
        D8.fix_result(6)
        D8.fix_result(7)
        D8.fix_result(8)
        assert D8(rnd).value == 1
        assert D8(rnd).value == 2
        assert D8(rnd).value == 3
        assert D8(rnd).value == 4
        assert D8(rnd).value == 5
        assert D8(rnd).value == 6
        assert D8(rnd).value == 7
        assert D8(rnd).value == 8
    with pytest.raises(ValueError):
        D8.fix_result(0)
    with pytest.raises(ValueError):
        D8.fix_result(9)


def test_bb_fixation():
    BBDie.clear_fixes()
    for seed in range(10):
        rnd = np.random.RandomState(seed)
        BBDie.fix_result(BBDieResult.ATTACKER_DOWN)
        BBDie.fix_result(BBDieResult.BOTH_DOWN)
        BBDie.fix_result(BBDieResult.PUSH)
        BBDie.fix_result(BBDieResult.DEFENDER_STUMBLES)
        BBDie.fix_result(BBDieResult.DEFENDER_DOWN)
        assert BBDie(rnd).value == BBDieResult.ATTACKER_DOWN
        assert BBDie(rnd).value == BBDieResult.BOTH_DOWN
        assert BBDie(rnd).value == BBDieResult.PUSH
        assert BBDie(rnd).value == BBDieResult.DEFENDER_STUMBLES
        assert BBDie(rnd).value == BBDieResult.DEFENDER_DOWN
    with pytest.raises(ValueError):
        BBDie.fix_result(1)
