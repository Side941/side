import numpy as np
import matplotlib.pyplot as plt

averageBestFitness =[]
for j in range(10):
    # Objective function (minimization problem)
    
    def objective_function(x):
        utility = 0
        for i in range(1, len(x)):
            utility += i * (2 * x[i] ** 2 - x[i - 1]) ** 2
        utility += (x[0] - 1) ** 2
        return utility
    """
    def objective_function(x):
        utility1 = 0
        utility2 = 0
        for i in range(len(x)):
            utility1 += (x[i]**2)
            utility2 += (0.5 * (i+1) * x[i])
        return (utility1 + (utility2**2) + (utility2**4))
    """
    # Ant Colony Optimization algorithm
    def ant_colony_optimization(num_ants, num_iterations, alpha, beta, rho, Q, search_space):
        best_solution = None
        best_fitness = float('inf')

        pheromone_matrix = np.ones_like(search_space) / len(search_space)

        for iteration in range(num_iterations):
            solutions = []

            # Ants build solutions
            for ant in range(num_ants):
                position = np.random.uniform(search_space[0], search_space[1], size=len(search_space))
                solutions.append(position)

            # Evaluate solutions and update pheromone
            for ant, solution in enumerate(solutions):
                fitness = objective_function(solution)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution

                # Update pheromone
                pheromone_matrix = (1 - rho) * pheromone_matrix
                pheromone_matrix += Q / fitness

            # Evaporation of pheromone
            pheromone_matrix = (1 - rho) * pheromone_matrix

        return best_solution, best_fitness

    # Parameters
    num_ants = 100
    num_iterations = 100
    alpha = 1.0  # Pheromone importance
    beta = 2.0  # Heuristic information importance
    rho = 0.1  # Evaporation rate
    Q = 1.0  # Pheromone deposit
    search_space = (-5, 10)

    # Run ACO algorithm
    best_solution, best_fitness = ant_colony_optimization(num_ants, num_iterations, alpha, beta, rho, Q, search_space)

    # Print results
    #print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
    averageBestFitness.append(best_fitness)
average = sum(averageBestFitness)/ len(averageBestFitness)
print("average: ", average)

plt.plot(range(1, 11), averageBestFitness, label="Best Fitness", marker='o', linestyle='-')
plt.xlabel('Run')
plt.ylabel('Best Fitness')
plt.legend()
plt.grid(True)
plt.title('Best Fitness in Each Run of ACO')
plt.show()