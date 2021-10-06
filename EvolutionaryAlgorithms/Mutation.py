import numpy as np

from EvolutionaryAlgorithms import EvolutionaryAlgorithm
from OptimizationProblems import OptimizationProblem

class Mutation(EvolutionaryAlgorithm):
    def __init__(self, population_size: int, optimization_problem: OptimizationProblem) -> None:
        super().__init__(population_size, optimization_problem)
        self.A = np.eye(population_size)
        self.B = 0.1 * np.eye(population_size)
    
    def step(self, input: np.ndarray):
        # Linear system: x(k+1) = Ax(k) + Bu(k) where A=eye(), B=diag(0.1)
        # State x(k) is of shape (pop_size, n)
        # shape(A) = (pop_size, pop_size)
        # Input u(k) is of shape (pop_size, n)
        # shape(B) = (pop_size, pop_size)

        # Mutate
        self.population = (
            np.matmul(self.A, self.population)
            + np.matmul(self.B, input)
        )