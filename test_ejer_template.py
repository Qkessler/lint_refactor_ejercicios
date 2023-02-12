from ejer_template import (
    es_primo,
    suma_n_primeros,
    tabla_del_n,
    primos_inferiores,
    esperanza_de_vida,
)


def test_suma_n_primeros():
    expected = [0, 4.0, 2.666666666666667, 3.466666666666667]
    found = [suma_n_primeros(num) for num in range(4)]

    assert expected == found


def test_tabla_del_n():
    tests = [
        (1, 1),
        (1, 2),
        (2, 6),
        (-1, 3),
        (-3, 4),
        (0, 6),
    ]
    expected = [
        [0],
        [0, 1],
        [0, 2, 4, 6, 8, 10],
        [0, -1, -2],
        [0, -3, -6, -9],
        [0, 0, 0, 0, 0, 0],
    ]
    found = [tabla_del_n(numero, top) for numero, top in tests]

    assert expected == found


def test_es_primo():
    tests = [0, 1, 2, 3, 5, 6, 9, 13]
    expected = [False, False, True, True, True, False, False, True]
    found = [es_primo(num) for num in tests]

    assert expected == found


def test_primos_inferiores():
    tests = [-1, 0, 1, 2, 5, 9]
    expected = [[], [], [], [], [2, 3], [2, 3, 5, 7]]
    found = [primos_inferiores(num) for num in tests]

    assert expected == found


def test_esperanza_de_vida():
    tests = [
        ({"a": 1, "b": 2, "c": 3}, {1: 50, 2: 65, 3: 75}),
        ({"a": 1, "b": 1, "c": 3}, {1: 50, 2: 65, 3: 75}),
        ({}, {}),
    ]
    expected = [
        {"a": 51, "b": 67, "c": 78},
        {"a": 51, "b": 51, "c": 78},
        {},
    ]
    found = [esperanza_de_vida(p, e) for p, e in tests]

    assert expected == found
