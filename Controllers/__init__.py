from abc import ABC
import numpy as np

from scipy.linalg import solve_discrete_are
from EvolutionaryAlgorithms import EvolutionaryAlgorithm

from OptimizationProblems import OptimizationProblem

class Controller(ABC):
    def __init__(self, optimization_problem: OptimizationProblem, evolutionary_algorithm: EvolutionaryAlgorithm) -> None:
        super().__init__()
        self.optimization_problem = optimization_problem
        self.evolutionary_algorithm = evolutionary_algorithm
    
    def step(self, state: np.ndarray) -> np.ndarray:
        # LQR controller
        pop_size, n = state.shape
        A = self.evolutionary_algorithm.A
        B = self.evolutionary_algorithm.B
        Q = self.optimization_problem.Q
        R = np.zeros((pop_size, pop_size))

        P = solve_discrete_are(A, B, Q, R)
        F = np.linalg.solve(R + B.T * P * B, B.T * P * A)

        return -1.0 * np.matmul(F, state)