import random
import matplotlib.pyplot as plt

averageBestUtility =[]
for j in range(10):
    N = 20
    LOOPS = 10000

    class solution:
        
        def __init__(self):
            self.variable = [0]*N
            self.utility = 0 
        


    iteration = []
    utility_v =[]


    individual = solution()

    for j in range (N):
        individual.variable[j] = random.randint(-10, 10)
    individual.utility = float('inf')

    def test_function(ind):
        utility = 0
        
        for i in range(1,N):
            utility += i*(2*ind.variable[i]**2 - ind.variable[i-1])**2
        utility = utility + ((ind.variable[0] - 1)**2)
        return utility
    """
    #Second function
    def test_function(ind):
        utility1 = 0
        utility2 = 0
        for i in range(N):
            utility1 += (ind.variable[i]**2)
            utility2 += (0.5 * (i+1) * ind.variable[i])
        return (utility1 + (utility2**2) + (utility2**4))
"""
    newind = solution()
    for x in range (LOOPS):
        for i in range(N):
            newind.variable[i] = individual.variable[i]
        change_point = random.randint(0, N-1) 
        newind.variable[change_point] = random.randint(0,100)
        newind.utility = test_function( newind )
        if newind.utility <= individual.utility:
            individual.variable[change_point] = newind.variable[change_point] 
            individual.utility = newind.utility
        
        utility_v.append(individual.utility)
        iteration.append(x)
        #print(utility_v) 
        
    #print(utility_v[-1])
    averageBestUtility.append(utility_v[-1])

# Calculate and print the average of the best fitness values
average_of_best_utility = sum(averageBestUtility) / len(averageBestUtility)
print("Average Best Utility:", average_of_best_utility)
    
    #fig, ax = plt.subplots()

      
plt.plot(utility_v)
plt.xlabel('iteration')
plt.ylabel('utility')
plt.grid(True)
plt.show()

