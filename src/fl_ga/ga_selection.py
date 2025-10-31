import numpy as np
import copy


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Tournament selection for GA.
    """
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = max(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best_idx])
    return selected


def crossover(parent1, parent2, p_c):
    """
    Single-point crossover.
    """
    if np.random.rand() < p_c:
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2


def mutation(chromosome, p_m, num_clients):
    """
    Mutation: change one gene to a new unique value.
    """
    chrom = copy.deepcopy(chromosome)
    for i in range(len(chrom)):
        if np.random.rand() < p_m:
            new_val = np.random.randint(num_clients)
            while new_val in chrom:
                new_val = np.random.randint(num_clients)
            chrom[i] = new_val
    return chrom


def ga_client_selection(
    num_clients, k=5, pop_size=90, generations=10, local_accs=None, adaptive=True
):
    """
    GA for client selection using FedCSGA-inspired method.
    Fitness: average local accuracy of selected clients.
    """
    # Initialize population
    population = []
    for _ in range(pop_size):
        chrom = np.random.choice(num_clients, k, replace=False).tolist()
        population.append(chrom)

    best_chrom = None
    best_fitness = -np.inf

    for gen in range(generations):
        # Adaptive probabilities
        if adaptive:
            p_c = 0.5 + 0.4 * (gen / generations)  # from 0.5 to 0.9
            p_m = 0.02 + 0.03 * (gen / generations)  # from 0.02 to 0.05
        else:
            p_c = 0.8
            p_m = 0.1

        # Fitness
        fitnesses = []
        for chrom in population:
            fitness = np.mean([local_accs[i] for i in chrom])
            fitnesses.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_chrom = chrom

        # Selection
        selected = tournament_selection(population, fitnesses)

        # Crossover
        new_population = []
        for i in range(0, len(selected), 2):
            p1 = selected[i]
            p2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            c1, c2 = crossover(p1, p2, p_c)
            new_population.extend([c1, c2])

        # Mutation
        population = [
            mutation(chrom, p_m, num_clients) for chrom in new_population[:pop_size]
        ]

    return best_chrom


if __name__ == "__main__":
    # Test
    num_clients = 10
    local_accs = np.random.rand(num_clients)
    selected = ga_client_selection(num_clients, k=5, local_accs=local_accs)
    print(f"Selected clients: {selected}")
