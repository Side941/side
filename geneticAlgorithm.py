import random
import matplotlib.pyplot as plt
import copy
import math

averageBestFitness = [] # Array to hold the average best fitness over 10 runs

#The for loop below allow the code to be run 10 times
for j in range(10):
    N = 20  # number of genes
    P = 100  # size of population
    MUTRATE = 0.12 #mutation rate
    MUTSTEP = 0.9#mutation step
    MIN = -10
    MAX = 10
    #MIN = -5 #max for the second function
    #MAX = 32.768 #max for the third function
    #MIN = -32.768 #min for the third function
    maxGeneration = 100 #Number of generations
    eliteSize = 1  # Number of elites to be retained in each generation
    tournament_size = 40  # Tournament size
    averageFitness, bestFitness, worstFitness = [], [], [] # arrays to hold the types of fitness

    class individual:
        def __init__(self):
            self.gene = [0] * N
            self.fitness = 0

        def __str__(self):
            return f"gene = {self.gene}\n fitness = {self.fitness}"

    population = [] #empty array to hold the population

    # Initialize the population
    for x in range(0, P):
        tempgene = []
        for y in range(0, N):
            tempgene.append(random.uniform(MIN, MAX))  # Generate genes within the given range
        newind = individual()
        newind.gene = tempgene.copy()
        population.append(newind)
        
    
    
    # the function below is the test function for the first function    
    def test_function(ind):
        utility = 0
            
        for i in range(1,N):
            utility += i*(2*ind.gene[i]**2 - ind.gene[i-1])**2
        utility = utility + ((ind.gene[0] - 1)**2)
        return utility   
    
    """
    #the function below is the test function for the second function

    def test_function(ind):
        utility1 = 0
        utility2 = 0
        for i in range(N):
            utility1 += (ind.gene[i]**2)
            utility2 += (0.5 * (i+1) * ind.gene[i])
        return (utility1 + (utility2**2) + (utility2**4))
        
    #The function below is the test function for the third function
        def test_function(ind):
        term1 = 0
        term2 = 0
        for i in range(N):
            term1 += ind.gene[i] ** 2
            term2 += math.cos(2.0 * math.pi * ind.gene[i])
        utility1 = -20.0 * math.exp(-0.2 * math.sqrt(0.05 * term1))
        utility2 = -math.exp(0.05 * term2)
        return utility1 + utility2 + 20.0 + math.exp(1)
"""
    for i in range(P):
        population[i].fitness = test_function(population[i])

    # function for the total fitness
    # this function is returning the avarage.
    def test_population(population):
        fitness = 0
        for i in range(P):
            fitness = fitness + population[i].fitness

        return fitness / len(population)


    def selection(population, tournament_size):
    # Tournament selection
        pop2 = []
        for i in range(0, P):
            tournament = random.sample(range(0, P), tournament_size)
            winner = min(tournament, key=lambda x: population[x].fitness)
            pop2.append(copy.deepcopy(population[winner]))
        return pop2
    
    
    #The function below is performing elitism
    def elitism(population, eliteSize):
        elites = sorted(population, key=lambda x: x.fitness)
        best_individual = copy.deepcopy(elites[0])
        worst_individual = copy.deepcopy(elites[-1])

        # This line find the index of the best individual in the original population
        best_index = population.index(elites[0])

        return best_individual, worst_individual, best_index

    #The function below is performing crossover
    def crossover(offspring):

        toff1 = individual() 
        toff2 = individual() 
        temp = individual()
        for i in range( 0, P, 2 ):
            toff1 = copy.deepcopy(offspring[i]) 
            toff2 = copy.deepcopy(offspring[i+1]) 
            temp = copy.deepcopy(offspring[i]) 
            #The line below  makes 4 randon cross points
            crosspoints = sorted(random.sample(range(1, N), 4))

            for k in range(0, 4, 2):
                for j in range(crosspoints[k], crosspoints[k + 1]):
                    toff1.gene[j] = toff2.gene[j]
                    toff2.gene[j] = temp.gene[j] 

            offspring[i] = copy.deepcopy(toff1) 
            offspring[i+1] = copy.deepcopy(toff2)
        return offspring


    #The function below is performing mutation
    def mutation(offspring):
        pop = []
        for i in range(0, P):
            newind = individual()
            newind.gene = []
            for j in range(0, N):
                gene = offspring[i].gene[j]
                mutprob = random.random()
                if mutprob < MUTRATE:
                    alter = random.uniform(-MUTSTEP, MUTSTEP)
                    gene = gene + alter
                    if (gene > MAX):
                        gene = MAX
                    if (gene < MIN):
                        gene = MIN
                newind.gene.append(gene)
            pop.append(newind)
        return pop

    #This for loop iterate over the max generation by performing, selection, elitism, crossover and mutation
    for i in range(maxGeneration):
        offspring = selection(population, tournament_size)
        elites = elitism(population, eliteSize)
        offspring.extend(elites)  # Adding elites to the selected individuals

        offspring = crossover(offspring)
        offspring = mutation(offspring)

        population = copy.deepcopy(offspring)

        # Apply elitism by replacing the best individual with the worst individual
        best_individual, worst_individual, best_index = elitism(population, eliteSize)
        population[best_index] = copy.deepcopy(worst_individual)

        for ind in population:
            ind.fitness = test_function(ind)
        mean_fitness = test_population(population)
        worst_fitness = max(ind.fitness for ind in population)
        best_fitness = min(ind.fitness for ind in population)
        averageFitness.append(mean_fitness)
        bestFitness.append(best_fitness)
        worstFitness.append(worst_fitness)

    print("Best Fitness:", bestFitness[-1])

    # Store the best fitness value for each run
    averageBestFitness.append(bestFitness[-1])

# Calculate and print the average of the best fitness values
average_of_best_fitness = sum(averageBestFitness) / len(averageBestFitness)
print("Average Best Fitness:", average_of_best_fitness)

#Ploting the results
plt.plot(averageFitness, label="Average Fitness")
plt.plot(bestFitness, label="Best Fitness")
plt.plot(worstFitness, label="Worst Fitness")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()
