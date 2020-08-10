import pytest

ak = pytest.importorskip("awkward1")

import numba
from vector.numba.awkward.lorentz.xyzt import behavior
from vector.single.lorentz.xyzt import LorentzXYZTFree


@numba.njit
def fill_it(testit, output):
    output.append(testit)
    output.append(testit)


def test_fillable():
    testit = LorentzXYZTFree(1, 2, 3, 4)
    output = ak.ArrayBuilder(behavior=behavior)
    fill_it(testit, output)

    assert str(output.snapshot()) == "[Lxyz(1 2 3 4), Lxyz(1 2 3 4)]"


@numba.njit
def adding_muons(input):
    for muons in input:
        for i in range(len(muons)):
            for j in range(i + 1, len(muons)):
                return muons[i] + muons[j]


def test_addition(ak_HZZ_example):
    example = ak.Array(ak_HZZ_example, behavior=behavior)
    assert str(adding_muons(example)) == "Lxyz(-15.2 -11 -19.5 94.2)"


@numba.njit
def do_cool_stuff(input, output):
    for muons in input:
        output.begin_list()

        for i in range(len(muons)):
            output.begin_list()

            for j in range(i + 1, len(muons)):
                zboson = muons[i] + muons[j]

                output.begin_tuple(2)
                output.index(0)
                output.append(zboson)
                output.index(1)
                output.append(zboson.mag)
                output.end_tuple()

            output.end_list()

        output.end_list()


def test_cool_stuff(ak_HZZ_example):
    example = ak.Array(ak_HZZ_example, behavior=behavior)
    output = ak.ArrayBuilder(behavior=behavior)
    do_cool_stuff(example, output)

    assert (
        str(output.snapshot())
        == "[[[(Lxyz(-15.2 -11 -19.5 94.2), 90.2)], []], [[]], [[(, ... [[]], [[]], [[]], [[]]]"
    )
