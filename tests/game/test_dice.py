import pytest
from ffai.core.model import D3, D6, D8, BBDie
from ffai.core.table import BBDieResult
import numpy as np

n = 10000


@pytest.mark.parametrize("die", [D3, D6, D8])
def test_d_die(die):
    results = {}
    for seed in range(n):
        rnd = np.random.RandomState(seed)
        result = die(rnd).value
        if result in results.keys():
            results[result] += 1
        else:
            results[result] = 1
    l = len(results.keys())
    assert l == 3 if die == D3 else True
    assert l == 6 if die == D6 else True
    assert l == 8 if die == D8 else True
    for key in results.keys():
        assert 0 < key <= l
        assert (n / (l+1)) < results[key] < (n / (l-1))


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


def test_bb_die():
    results = {}
    for seed in range(n):
        rnd = np.random.RandomState(seed)
        result = BBDie(rnd).value
        if result in results.keys():
            results[result] += 1
        else:
            results[result] = 1
    l = len(results.keys())
    assert l == 5
    for key in results.keys():
        assert key in [BBDieResult.ATTACKER_DOWN,
                       BBDieResult.BOTH_DOWN,
                       BBDieResult.DEFENDER_DOWN,
                       BBDieResult.DEFENDER_STUMBLES,
                       BBDieResult.PUSH]
        if key == BBDieResult.PUSH:
            assert (n / (6+1))*2 < results[key] < (n / (6-1))*2
        else:
            assert (n / (6 + 1)) < results[key] < (n / (6 - 1))
