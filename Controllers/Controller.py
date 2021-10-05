from abc import ABC
import numpy as np
from numpy.linalg.linalg import solve
from scipy.linalg import solve_discrete_are

class Controller(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def step(self, state: np.ndarray) -> np.ndarray:
        # LQR controller
        pop_size, n = state.shape
        A = np.eye(n)
        B = 0.1 * np.eye(n)
        Q = np.eye(n)
        R = np.zeros((n, n))

        P = solve_discrete_are(A, B, Q, R)
        F = np.linalg.solve(R + B.T * P * B, B.T * P * A)

        return -1.0 * np.squeeze(np.matmul(
            np.repeat(F[np.newaxis,:,:], pop_size, axis=0),
            state[:,:,np.newaxis]
        ))