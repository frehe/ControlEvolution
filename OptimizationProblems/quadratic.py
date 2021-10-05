import numpy as np
from OptimizationProblems import OptimizationProblem

class Quadratic(OptimizationProblem):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        
    def function(self, x: np.ndarray) -> np.float:
        return x.transpose() * x