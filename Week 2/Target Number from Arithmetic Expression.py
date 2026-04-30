import random

DIGITS = [str(i) for i in range(10)]
OPERATORS = ['+', '-', '*']


def initialize_population(pop_size, num_genes):
    population = []

    for _ in range(pop_size):
        individual = []

        for i in range(num_genes):
            if i % 2 == 0:
                individual.append(random.choice(DIGITS))
            else:
                individual.append(random.choice(OPERATORS))
        population.append(individual)
        print(individual)
    return population


def evaluate_expression(ind):
    result = int(ind[0])

    i = 1
    while i < len(ind):
        op = ind[i]
        num = int(ind[i + 1])

        if op == '+':
            result += num
        elif op == '-':
            result -= num
        elif op == '*':
            result *= num

        i += 2
    return result


# Fitness
def fitness(ind, target):
    value = evaluate_expression(ind)
    return 1 / (1 + abs(target - value))


# Selection 
def select(population, fitnesses):
    candidates = random.sample(range(len(population)), 3)
    best = max(candidates, key=lambda i: fitnesses[i])
    return population[best][:]


# Crossover
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]


# Mutation
def mutate(ind, mutation_rate):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            if i % 2 == 0:
                ind[i] = random.choice(DIGITS)
            else:
                ind[i] = random.choice(OPERATORS)
    return ind


# Genetic Algorithm
def run_ga(target, pop_size, num_genes, generations, mutation_rate):

    population = initialize_population(pop_size, num_genes)

    for gen in range(generations):

        fitnesses = [fitness(ind, target) for ind in population]

        best_index = max(range(len(population)), key=lambda i: fitnesses[i])
        best_ind = population[best_index]
        best_val = evaluate_expression(best_ind)

        print(f"Generation {gen+1}: {' '.join(best_ind)} = {best_val}")

        if best_val == target:
            print(" solution found!")
            break

        new_population = []

        while len(new_population) < pop_size:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)

            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population


target1 = 10
genes1 = 5

target2 = 42
genes2 = 9

target3 = 350
genes3 = 9

target4 = 999
genes4 = 13

target5 = 500
genes5 = 13

pop_size = 10
mutation_rate = 0.05
generations = 300

print("Test 1")
run_ga(target1, pop_size, genes1, generations, mutation_rate)

print("\nTest 2")
run_ga(target2, pop_size, genes2, generations, mutation_rate)

print("\nTest 3")
run_ga(target3, pop_size, genes3, generations, mutation_rate)

print("\nTest 4")
run_ga(target4, pop_size, genes4, generations, mutation_rate)

print("\nTest 5")
run_ga(target5, pop_size, genes5, generations, mutation_rate)
