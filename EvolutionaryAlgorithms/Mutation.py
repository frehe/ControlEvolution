import numpy as np

from EvolutionaryAlgorithms import EvolutionaryAlgorithm
from OptimizationProblems import OptimizationProblem

class Mutation(EvolutionaryAlgorithm):
    def __init__(self, population_size: int, optimization_problem: OptimizationProblem) -> None:
        super().__init__(population_size, optimization_problem)
        self.A = np.eye(optimization_problem.n)
        self.B = 0.1 * np.eye(optimization_problem.n)
    
    def step(self, input: np.ndarray):
        # Linear system: x(k+1) = Ax(k) + Bu(k) where A=eye(), B=diag(0.1)

        # Mutate
        self.population = (
            np.matmul(
                np.repeat(self.A[np.newaxis, :, :], 10, 0),
                self.population[:, :, np.newaxis]
            )
            + np.matmul(
                np.repeat(self.B[np.newaxis, :, :], 10, 0),
                input[:, :, np.newaxis]
            )
        ).squeeze()