import numpy as np
from abc import ABC


class OptimizationProblem(ABC):
    def __init__(self, n: int, pop_size: int) -> None:
        self.n = n
        self.pop_size = pop_size
    
    def function(self, x: np.ndarray) -> np.float:
        pass 