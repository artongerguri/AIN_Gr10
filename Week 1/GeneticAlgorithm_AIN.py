import random
import string
import matplotlib.pyplot as plt



def initialize_population(pop_size: int, length: int, charset: str):
    return [
        "".join(random.choice(charset) for _ in range(length))
        for _ in range(pop_size)
    ]



def fitness(individual: str, target: str) -> float:
    matches = sum(1 for c, t in zip(individual, target) if c == t)
    return matches / len(target)



def select(population, fitnesses, tournament_k: int = 3) -> str:
    indices = random.sample(range(len(population)), tournament_k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def crossover(parent1: str, parent2: str) -> str:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    n = len(parent1)
    if n < 2:
        return parent1  

    point = random.randint(1, n - 1)  
    return parent1[:point] + parent2[point:]


def mutate(individual: str, mutation_rate: float, charset: str) -> str:
    
    chars = list(individual)
    for i in range(len(chars)):
        if random.random() < mutation_rate:
            chars[i] = random.choice(charset)
    return "".join(chars)


def run_ga(target: str,
           pop_size: int = 100,
           generations: int = 200,
           mutation_rate: float = 0.01,
           charset: str = None,
           tournament_k: int = 3,
           elitism: bool = True,
           verbose: bool = True):

    if charset is None:
        charset = string.ascii_uppercase + " "

    length = len(target)
    population = initialize_population(pop_size, length, charset)

    best_history = []
    best_individual = None
    best_fit = -1.0

    for gen in range(generations):
        fits = [fitness(ind, target) for ind in population]

        gen_best_idx = max(range(pop_size), key=lambda i: fits[i])
        gen_best_fit = fits[gen_best_idx]
        gen_best_ind = population[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_individual = gen_best_ind

        best_history.append(best_fit)

        if verbose:
            print(f"Gen {gen:03d} | best_fitness={best_fit:.3f} | best='{best_individual}'")

        if best_fit >= 1.0:
            break

        new_population = []

        if elitism:
            new_population.append(gen_best_ind)

        while len(new_population) < pop_size:
            p1 = select(population, fits, tournament_k=tournament_k)
            p2 = select(population, fits, tournament_k=tournament_k)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, charset)
            new_population.append(child)

        population = new_population

    return best_history, best_individual, best_fit


def plot_histories(histories, labels, title):
    plt.figure()
    for h, lab in zip(histories, labels):
        plt.plot(h, label=lab)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness so far")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    tests = [
        ("01", "HI", string.ascii_uppercase),                 
        ("02", "EVOLUTION", string.ascii_uppercase),          
        ("03", "NATURE INSPIRED", string.ascii_uppercase + " "),  
        ("04", "AAAAAAAAAA", string.ascii_uppercase),         
        ("05", "GA IS COOL", string.ascii_uppercase + " "),  
    ]

    pop_size = 100
    mutation_rate = 0.01
    generations = 200

    all_hist = []
    all_labels = []

    results = {}

    for code, target, charset in tests:
        print("\n" + "=" * 60)
        print(f"Running test {code}: '{target}'")
        hist, best_ind, best_fit = run_ga(
            target=target,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            charset=charset,
            tournament_k=3,
            elitism=True,
            verbose=False  
        )
        results[code] = (hist, best_ind, best_fit)
        all_hist.append(hist)
        all_labels.append(f"{code}:{target}")

        print(f"Result {code}: best_fit={best_fit:.3f}, best='{best_ind}', gens={len(hist)-1}")

    plot_histories(all_hist, all_labels, "GA Convergence (Best fitness per generation)")

    h2 = results["02"][0]
    h3 = results["03"][0]
    plot_histories([h2, h3], ["02:EVOLUTION", "03:NATURE INSPIRED"], "Compare convergence: String 02 vs 03")
