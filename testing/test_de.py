import pytest
import numpy as np
import random

from differential_evolution import DifferentialEvolution

SEED = 7
random.seed(SEED)
np.random.seed(SEED)


# CONSTANTS

def rastrigin(array, A=10):
    return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (
            array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))


BOUNDS = np.array([[-20, 20], [-20, 20]])
FOBJ = rastrigin

"""
Ваша задача добиться 100% покрытия тестами DifferentialEvolution
Различные этапы тестирования логики разделяйте на различные функции
Запуск команды тестирования:
pytest -s test_de.py --cov-report=json --cov
"""


def test_initialization():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    assert de.fobj == FOBJ
    assert np.all(de.bounds) == np.all(BOUNDS)
    assert de.mutation_coefficient == 0.8
    assert de.crossover_coefficient == 0.7
    assert de.population_size == 20


def test_population_initialization():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    de._init_population()
    assert de.population.shape == (de.population_size, de.dimensions)
    assert np.all(de.population >= 0) and np.all(de.population <= 1)
    assert np.all(de.population_denorm >= de.min_bound) and np.all(de.population_denorm <= de.max_bound)


def test_mutation():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    de.population = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    de.idxs = [0, 1, 2]

    a, b, c = de.population[np.random.choice(de.idxs, 3, replace=False)]
    de.mutant = de._mutation()

    assert np.all(de.a) == np.all(a)
    assert np.all(de.b) == np.all(b)
    assert np.all(de.c) == np.all(c)
    assert np.all(de.mutant) == np.all([0., 0., 0.06])


def test_crossover():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    de.dimensions = 3
    de.crossover_coefficient = 0.5
    de.cross_points = de._crossover()
    assert de.cross_points.shape == (de.dimensions,)
    assert np.any(de.cross_points)


def test_recombination():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    de.dimensions = 3
    de.cross_points = np.array([True, False, True])
    de.mutant = np.array([0.1, 0.2, 0.3])
    de.population = np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    de.min_bound, de.max_bound = np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])
    de.diff = np.fabs(de.min_bound - de.max_bound)
    de.trial, de.trial_denorm = de._recombination(1)

    assert de.trial.shape == (de.dimensions,)
    assert np.all(de.trial >= 0) and np.all(de.trial <= 1)
    assert de.trial_denorm.shape == (de.dimensions,)
    assert np.all(de.trial_denorm >= de.min_bound) and np.all(de.trial_denorm <= de.max_bound)


def test_evaluation():
    de = DifferentialEvolution(FOBJ, BOUNDS)
    de.population = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    de.fitness = np.array([1.0, 2.0, 3.0])
    de.trial = np.array([0.4, 0.5, 0.6])
    de.trial_denorm = np.array([0.4, 0.5, 0.6])
    de.best_idx = np.argmin(de.fitness)
    de._evaluate(0.5, 1)

    assert de.fitness[1] == 0.5
    assert np.all(de.population[1] == de.trial)
    assert de.best_idx == 1
    assert np.all(de.best == de.trial_denorm)


def test_iteration():
    de = DifferentialEvolution(FOBJ, BOUNDS)

    de._init_population()
    de.iterate()

    assert len(de.population) == de.population_size
    assert len(de.fitness) == de.population_size
    assert de.best_idx is not None
    assert de.best is not None


pytest.main()
