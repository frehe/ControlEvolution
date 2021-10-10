import numpy as np
from EvolutionaryAlgorithms.Mutation import Mutation

from OptimizationProblems.quadratic import Quadratic
from Controllers import Controller
from EvolutionaryAlgorithms.crossover import Crossover


num_iterations = 10
pop_size = 3
n = 2

def run_optimization():
    optimization_problem = Quadratic(n, pop_size)
    evolutionary_algorithm = Mutation(pop_size, optimization_problem)
    controller = Controller(optimization_problem, evolutionary_algorithm)

    for k in range(num_iterations):
        # u(k) = K * x(k)
        input = controller.step(evolutionary_algorithm.population)

        # x(k+1) = Ax(k) + Bu(k)
        evolutionary_algorithm.step(input)
    
    print(np.reshape(evolutionary_algorithm.population, (pop_size, n)))


run_optimization()