import random as r
import pandas as pd

# Function to evaluate the fitness of an individual in the population
def fitness(individual):
    # Example function: -(x-3)^2 + 9
    fit = -1 * (individual - 3) ** 2 + 9
    return fit

# Two random parents produce an offspring
def mate(parents, numParents):
    parentOne = parents[r.randrange(0, numParents)]
    parentTwo = parents[r.randrange(0, numParents)]

    if len(parentOne) < len(parentTwo):
        point = r.randrange(1, len(parentOne))
    else:
        point = r.randrange(1, len(parentTwo))

    offspring = parentOne[:point] + parentTwo[point:]
    return offspring

# Randomly initialize the first individuals within a population
def initialization(popSize):
    # Initial population of random numbers between 0 - 10
    initPopulation = [r.randrange(0, 11) for i in range(popSize)]
    return initPopulation

# Evaluate the fitness of all individuals in the population
def evaluation(population):

    fittest = 0
    popSize = len(population)
    evaluations = [0] * popSize
    total = 0

    for i in range(0, popSize):
        evaluations[i] = fitness(population[i])
        total += evaluations[i]

        if fittest < evaluations[i]:
            fittestScore = evaluations[i]
            fittestIndividual = population[i]

    averageFitness = total/len(evaluations)
    
    return evaluations, fittestScore, fittestIndividual, averageFitness

# Select individuals of an evaluated population to generate the next Generation
def selection(population, evaluations, fittest):
    
    selectedPop = []
    for i in range(len(population)):
        selectionChance = evaluations[i]/fittest
        if r.random() < selectionChance:
            selectedPop.append(population[i])

    return selectedPop

# Creation of the second generation
def reproduction(population, mutationRate, generationSize):
    popSize = len(population)

    # Convert to binary
    for i in range(popSize):
        population[i] = bin(population[i]).replace('0b', '00') # String

     # Fill the second Generation
    while len(population) < generationSize:
        parents = population
        population.append(mate(parents, popSize))

    # Perform mutation
    for i in range(0, generationSize):
        flip = r.randrange(1, len(population[i]))
        mutation = r.random() * 100
        if mutation > mutationRate:
            population[i] = population[i][:flip] + str(int(not int(population[i][flip]))) + population[i][flip+1:]

    # Convert back to decimal
    for i in range(0, generationSize):
        population[i] = int(population[i], base = 2) # Integer

    return population

# Genetic Algorithm
def geneticAlgorithm(numGenerations, mutationRate, generationSize, chartOutput = True):
    
    t = 0
    generations = [initialization(generationSize)]
    evaluations = []
    fitness = [[],[],[]]
    selections = []
    chart = None

    while t < numGenerations:

        # Evaluation
        evaluations, fittestScore, fittestIndividual, averageFitness = evaluation(generations[t])
        evaluations.append(evaluations)

        # Store values for plotting
        fitness[0].append(fittestIndividual)
        fitness[1].append(fittestScore)
        fitness[2].append(averageFitness)

        # Selection
        selections.append(selection(generations[t], evaluations, fittestScore))

        # Crossover, Mutation, Reproduction
        generations.append(reproduction(selections[t], mutationRate, generationSize))
        t += 1

    # Determine the fittest individual overall
    for i in range(len(fitness[0])):
        if fittestScore < fitness[1][i]:
            fittestIndividual = fitness[0][i]
            fittestScore = fitness[1][i]

    # Chart the generations if desired
    if chartOutput is True:
        data = {
            "Fittest": fitness[0],
            "Fitness Score": fitness[1],
            "Average": fitness[2]
        }
        chart = pd.DataFrame(data)
        chart.index.name = "Generations"

    return fittestScore, fittestIndividual, chart


# Run the Algorithm to maximize the function: -(x-3)^2 + 9
# Expected fittest Individual: 3
# Expected maximum Fitness: 9

fittestScore, fittestIndividual, chart = geneticAlgorithm(numGenerations=10, mutationRate=50, generationSize=10)

print(fittestScore, fittestIndividual)
print(chart)