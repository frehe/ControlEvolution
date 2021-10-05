import numpy as np
from abc import ABC
from OptimizationProblems import OptimizationProblem

class EvolutionaryAlgorithm(ABC):
    def __init__(self, population_size: int, optimization_problem: OptimizationProblem) -> None:
        self.population_size = population_size
        self.population = np.random.randn(self.population_size, optimization_problem.n)
        self.parameters = np.ndarray(1)

    def step(self, input: np.ndarray):
        pass