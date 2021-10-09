import numpy as np

from EvolutionaryAlgorithms import EvolutionaryAlgorithm
from OptimizationProblems import OptimizationProblem


class Mutation(EvolutionaryAlgorithm):
    def __init__(
        self, population_size: int, optimization_problem: OptimizationProblem
    ) -> None:
        super().__init__(population_size, optimization_problem)
        # TODO: Try with more complicated EA dynamics
        self.A = 100 * np.eye(population_size * optimization_problem.n)
        self.B = 0.1 * np.eye(population_size * optimization_problem.n)

    def step(self, input: np.ndarray):
        # EA is a Linear system: x(k+1) = Ax(k) + Bu(k)
        # State x(k) is of shape (pop_size x n, 1)
        # Input u(k) is of shape (pop_size x n, 1)

        # Mutate
        self.population = np.matmul(self.A, self.population) + np.matmul(self.B, input)
