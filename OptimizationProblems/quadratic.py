import numpy as np
from OptimizationProblems import OptimizationProblem

class Quadratic(OptimizationProblem):
    def __init__(self, n: int, pop_size: int) -> None:
        super().__init__(n, pop_size)
        self.Q = np.eye(pop_size)
        
    def function(self, x: np.ndarray) -> np.float:
        return np.trace(np.matmul(x, np.matmul(self.Q, x.T)))