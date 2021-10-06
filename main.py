import numpy as np
from EvolutionaryAlgorithms.Mutation import Mutation

from OptimizationProblems.quadratic import Quadratic
from Controllers.Controller import Controller
from EvolutionaryAlgorithms.crossover import Crossover


num_iterations = 10

def run_optimization():
    optimization_problem = Quadratic(2)
    evolutionary_algorithm = Mutation(10, optimization_problem)
    controller = Controller(optimization_problem, evolutionary_algorithm)

    for k in range(num_iterations):
        # u(k) = K * x(k)
        input = controller.step(evolutionary_algorithm.population)

        # x(k+1) = Ax(k) + Bu(k)
        evolutionary_algorithm.step(input)
    
    print(evolutionary_algorithm.population)


run_optimization()