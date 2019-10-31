import pytest
from ffai.core.model import D3, D6, D8, BBDie
from ffai.core.table import BBDieResult
import numpy as np

n = 10000


@pytest.mark.parametrize("die", [D3, D6, D8])
def test(die):
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
