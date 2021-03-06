import numpy as np
from OptimizationProblems import OptimizationProblem

class Quadratic(OptimizationProblem):
    def __init__(self, n: int, pop_size: int) -> None:
        super().__init__(n, pop_size)
        self.Q = np.eye(pop_size * n)
        
    def function(self, x: np.ndarray) -> np.float:
        """Compute cost function of a whole population

        :param x: array of shape (pop_size, n)
        :type x: np.ndarray
        :return: Cost value x'*Q*x summed over individuals
        :rtype: np.float
        """
        return 0.5 * np.trace(np.matmul(x, np.matmul(self.Q, x.T)))