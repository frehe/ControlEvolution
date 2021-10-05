import numpy as np

from EvolutionaryAlgorithms import EvolutionaryAlgorithm
from OptimizationProblems import OptimizationProblem

class Crossover(EvolutionaryAlgorithm):
    def __init__(self, population_size: int, optimization_problem: OptimizationProblem) -> None:
        super().__init__(population_size, optimization_problem)
    
    def step(self, input: np.ndarray):
        # Average random pairs of individuals
        population_permuted = np.random.shuffle(self.population)
        self.population = (self.population + population_permuted) / 2

        # Mutate
        self.population += 0.1 * np.random.standard_normal(np.size(self.population))