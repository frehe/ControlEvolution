import numpy as np

from EvolutionaryAlgorithms import EvolutionaryAlgorithm
from OptimizationProblems import OptimizationProblem

class Mutation(EvolutionaryAlgorithm):
    def __init__(self, population_size: int, optimization_problem: OptimizationProblem) -> None:
        super().__init__(population_size, optimization_problem)
    
    def step(self, input: np.ndarray):
        # Linear system: x(k+1) = Ax(k) + Bu(k) where A=eye(), B=diag(0.1)

        # Mutate
        self.population += 0.1 * np.matmul(np.eye(self.population_size), input)