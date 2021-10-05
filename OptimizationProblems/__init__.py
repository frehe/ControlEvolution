import numpy as np
from abc import ABC


class OptimizationProblem(ABC):
    def __init__(self, n: int) -> None:
        self.n = n
    
    def function(self, x: np.ndarray) -> np.float:
        pass 