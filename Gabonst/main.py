import random
import matplotlib.pyplot as plt

# Parameters
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.1

# functionf(x) = x^2
def fitness(x):
    return x**2

# Random
def create_population(size):
    return [random.randint(0, 31) for _ in range(size)]

#roulette wheel yöntemi)

def select(population):
    fitness_values = [fitness(x) for x in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    return random.choices(population, weights=probabilities, k=2)


def crossover(parent1, parent2):
    point = random.randint(1, 4)
    mask = (1 << point) - 1
    child1 = (parent1 & mask) | (parent2 & ~mask)
    child2 = (parent2 & mask) | (parent1 & ~mask)
    return child1, child2


def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_bit = 1 << random.randint(0, 4)
        individual ^= mutation_bit
    return individual


def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    max_fitness_history = []
    avg_fitness_history = []

    for generation in range(GENERATIONS):
        fitness_values = [fitness(x) for x in population]
        max_fitness_history.append(max(fitness_values))
        avg_fitness_history.append(sum(fitness_values) / len(fitness_values))

        population = sorted(population, key=fitness, reverse=True)
        next_generation = population[:2]  # Elitizm
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        population = next_generation[:POPULATION_SIZE]

    return max_fitness_history, avg_fitness_history

# GABONST
def gabonst_algorithm():
    population = create_population(POPULATION_SIZE)
    max_fitness_history = []
    avg_fitness_history = []

    for generation in range(GENERATIONS):
        fitness_values = [fitness(x) for x in population]
        max_fitness_history.append(max(fitness_values))
        avg_fitness_history.append(sum(fitness_values) / len(fitness_values))

        fitness_mean = sum(fitness_values) / len(population)
        population = sorted(population, key=fitness, reverse=True)
        next_generation = population[:2]  # Elitizm
        
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            if fitness(child1) < fitness_mean:
                child1 = random.randint(0, 31)
            if fitness(child2) < fitness_mean:
                child2 = random.randint(0, 31)

            next_generation.append(child1)
            next_generation.append(child2)
        population = next_generation[:POPULATION_SIZE]

    return max_fitness_history, avg_fitness_history

ga_max, ga_avg = genetic_algorithm()
gabonst_max, gabonst_avg = gabonst_algorithm()

plt.figure(figsize=(12, 6))

# GA 
plt.plot(ga_max, label="GA Max Fitness", linestyle='-', marker='o')
plt.plot(ga_avg, label="GA Avg Fitness", linestyle='--', marker='x')

# GABONST 
plt.plot(gabonst_max, label="GABONST Max Fitness", linestyle='-', marker='s')
plt.plot(gabonst_avg, label="GABONST Avg Fitness", linestyle='--', marker='^')

plt.title("Genetic Algorithm vs. GABONST Fitness ")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.show()
