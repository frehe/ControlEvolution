from abc import ABC
import numpy as np

from scipy.linalg import solve_discrete_are

from OptimizationProblems import OptimizationProblem

class Controller(ABC):
    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        super().__init__()
        self.optimization_problem = optimization_problem
    
    def step(self, state: np.ndarray) -> np.ndarray:
        # LQR controller
        pop_size, n = state.shape
        A = np.eye(n)
        B = 0.1 * np.eye(n)
        Q = self.optimization_problem.Q
        R = np.zeros((n, n))

        P = solve_discrete_are(A, B, Q, R)
        F = np.linalg.solve(R + B.T * P * B, B.T * P * A)

        return -1.0 * np.squeeze(np.matmul(
            np.repeat(F[np.newaxis,:,:], pop_size, axis=0),
            state[:,:,np.newaxis]
        ))