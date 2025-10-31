# Mathematical Comparison of FedCSGA and GenFed Algorithms in Federated Learning

## Introduction

This report provides a detailed mathematical comparison between two genetic algorithm (GA)-based approaches for optimizing Federated Learning (FL): **FedCSGA** (Wu et al., 2025) and **GenFed** (Zheng et al., 2025). Both papers address challenges in FL, such as non-IID data distributions and system heterogeneity, but differ in focus—FedCSGA optimizes client selection, while GenFed optimizes model aggregation. The analysis draws from the papers' methodologies, highlighting similarities, differences, and performance implications.

## Problem Setup

Federated Learning (FL) aims to train a global model \( \theta \) across \( M \) distributed clients without exchanging raw data, due to privacy constraints. Each client \( i \) has local dataset \( D_i \) of size \( n_i \), with total data \( N = \sum n_i \).

### Challenges
- **Non-IID Data**: Client data distributions differ, e.g., \( p(x, y | i) \neq p(x, y | j) \), leading to biased local updates and slow convergence.
- **System Heterogeneity**: Varying client compute/communication capacities cause stragglers.
- **Client Selection**: Random selection ignores quality, resulting in suboptimal aggregation.

### Optimization Goal
Minimize global loss \( \mathcal{L}(\theta) = \sum_{i=1}^M \frac{n_i}{N} \mathcal{L}_i(\theta) \), where \( \mathcal{L}_i(\theta) = \mathbb{E}_{(x,y) \sim D_i} \ell(\theta; x, y) \), under constraints like communication rounds and client participation.

Traditional FedAvg: Select \( k \) clients randomly, train locally for \( E \) epochs with SGD, aggregate via weighted average.

## Dataset and Model Details

### Dataset: MNIST
- **Description**: 70,000 grayscale images (60,000 train, 10,000 test) of handwritten digits (0-9), 28x28 pixels.
- **Partitioning**: Non-IID via Dirichlet distribution \( \text{Dir}(\alpha) \) with \( \alpha = 0.5 \) (moderate heterogeneity). For each class \( c \in \{0,\dots,9\} \), sample proportions \( p_c = (p_{c,1}, \dots, p_{c,M}) \sim \text{Dir}(\alpha \cdot \mathbf{1}_M) \), assign samples accordingly.
- **Result**: Clients have skewed class distributions, simulating real-world non-IID.

### Model: Multi-Layer Perceptron (MLP)
- **Architecture**: 3 layers – Input (784), Hidden1 (128), Hidden2 (64), Output (10).
- **Forward Pass**:
  \[
  h_1 = \sigma(W_1 x + b_1), \quad h_2 = \sigma(W_2 h_1 + b_2), \quad \hat{y} = \text{softmax}(W_3 h_2 + b_3)
  \]
  where \( \sigma \) is ReLU, \( x \in \mathbb{R}^{784} \), \( \hat{y} \in \mathbb{R}^{10} \).
- **Parameters**: ~109,000 (784*128 + 128 + 128*64 + 64 + 64*10 + 10).
- **Training**: Cross-entropy loss \( \ell = -\sum y \log \hat{y} \), SGD with lr=0.01, batch_size=32, epochs=5 per round.

## First Principles: Deriving GA for FL Optimization

FL optimization is combinatorial: selecting subsets of clients is NP-hard. GA provides a heuristic via evolutionary search.

### Why GA?
- **Evolutionary Analogy**: Clients as "individuals"; model updates as "traits"; aggregation as "reproduction."
- **Fitness**: Measures subset quality (e.g., convergence speed, accuracy).
- **Operators**: Mimic natural selection to explore solution space.

### Derivation for Client Selection (FedCSGA)
- **Formulation**: Maximize \( f(S) \) for \( S \subseteq [M], |S|=k \), where \( f(S) \) proxies global performance.
- **From First Principles**: In non-IID, optimal \( S \) balances diversity (participation) and quality (local fit). GA evolves \( S \) via fitness, avoiding exhaustive search.

### Derivation for Aggregation (GenFed)
- **Formulation**: Post-selection, choose \( \rho_t \leq k \) models to aggregate.
- **From First Principles**: Aggregation as "crossover" – combine best "genes" (parameters). Fitness via validation accuracy ensures robustness.

## FedCSGA: GA-Based Client Selection

## FedCSGA: GA-Based Client Selection

FedCSGA uses a full GA to select optimal client subsets, modeling selection as an optimization problem under time constraints.

### Mathematical Formulation
- **Problem**: Select subset \( S \subseteq \{1, \dots, M\} \) of size \( k \) to maximize fitness, where \( M \) is total clients.
- **Chromosome**: Permutation of \( k \) client indices, e.g., \( [c_1, c_2, \dots, c_k] \).
- **Fitness Function** (for non-IID settings):
  \[
  f(S) = \alpha \cdot \frac{|S|}{M} + (1 - \alpha) \cdot \frac{A(S)}{A_{\max}}
  \]
  where \( \alpha = 0.7 \), \( A(S) \) is global model accuracy after training/aggregating \( S \), and \( A_{\max} \) is maximum possible accuracy. For IID, \( f(S) = \frac{|S|}{M} \).

- **GA Parameters**:
  - Population size: 90
  - Generations: 10
  - Selection: Tournament (size 3)
  - Crossover: Single-point, adaptive probability \( p_c = k_1 + k_2 \cdot \frac{g}{G} \), where \( k_1 = 0.5 \), \( k_2 = 0.9 \), \( g \) is current generation, \( G = 10 \).
  - Mutation: Gene replacement, adaptive probability \( p_m = k_3 + k_4 \cdot \frac{g}{G} \), where \( k_3 = 0.02 \), \( k_4 = 0.05 \).

### Algorithm Steps
1. Initialize population with random permutations.
2. For each generation:
   - Evaluate fitness (simulate FL round for each chromosome).
   - Select parents via tournament.
   - Apply crossover and mutation.
3. Return best chromosome's clients for FL aggregation.

### Key Mathematical Insights
- **Optimization**: Balances participation (numerator) and quality (denominator) via \( \alpha \).
- **Adaptivity**: \( p_c \) and \( p_m \) increase to favor exploitation over exploration.
- **Complexity**: Fitness evaluation requires \( O(k \cdot E \cdot B) \) per chromosome, where \( E \) is local epochs, \( B \) batch size—expensive but effective for small \( M \).

### Derivation from First Principles
FL client selection is a subset selection problem: find \( S^* = \arg\max_S f(S) \), where \( f(S) \) measures contribution to global convergence. Since \( 2^M \) subsets are intractable, GA approximates via evolutionary search. Fitness \( f(S) \) combines diversity (\( |S|/M \)) and quality (\( A(S)/A_{\max} \)), derived from FL's bias-variance trade-off in non-IID settings. Adaptive operators ensure early exploration (low \( p_c, p_m \)) and late exploitation (high \( p_c, p_m \)), mimicking natural evolution.

## GenFed: GA-Inspired Model Aggregation

GenFed reinterprets FL as GA: local training as mutation, aggregation as crossover, model evaluation as fitness. It selects top models post-training for aggregation.

### Mathematical Formulation
- **Problem**: From \( K \) participating clients, select top \( \rho_t \) models for aggregation.
- **Fitness**: Validation accuracy \( A_i \) of client \( i \)'s model on global validation set.
- **Selection**: Sort models by \( A_i \), select top \( \rho_t \):
  \[
  S = \arg\max_{|S|=\rho_t} \sum_{i \in S} A_i
  \]
  (Equivalent to greedy top-\( \rho_t \).)
- **Aggregation**: FedAvg on selected models:
  \[
  \theta_{t+1} = \sum_{i \in S} w_i \theta_{i,t}, \quad w_i = \frac{n_i}{\sum_{j \in S} n_j}
  \]
- **Dynamic \( \rho_t \)** (strategies):
  - Constant: \( \rho_t = \rho_{\max} \)
  - Linear: \( \rho_t = \min(\rho_{\max}, \lfloor \rho_{\max} \cdot \frac{t}{T} + 1 \rfloor) \)
  - Power: \( \rho_t = \min(\rho_{\max}, \lfloor \rho_{\max} \cdot (1 - b^t) + 1 \rfloor) \), \( b \in (0,1) \)
  - Sinusoidal: Periodic variation, e.g., \( \rho_t = \min(\rho_{\max}, \lfloor \rho_{\max} \cdot \sin(\frac{t}{2c} \pi) + 1 \rfloor) \)

### Algorithm Steps
1. Clients train locally, upload models.
2. Server evaluates \( A_i \) on validation set.
3. Select top \( \rho_t \) models.
4. Aggregate selected models.

### Key Mathematical Insights
- **GA Analogy**: Selection mimics fitness-based survival; aggregation as crossover.
- **Dynamic Selection**: \( \rho_t \) balances exploration (high \( \rho_t \)) and exploitation (low \( \rho_t \)).
- **Complexity**: \( O(K \log K) \) for sorting, low overhead compared to FedCSGA.

### Derivation from First Principles
In FL, aggregation amplifies poor updates in non-IID settings. GenFed treats aggregation as "crossover" of high-fitness "chromosomes" (models). Fitness \( A_i \) measures model quality on neutral validation data, ensuring robustness. Dynamic \( \rho_t \) adapts to round \( t \), starting inclusive (high \( \rho_t \)) for diversity, narrowing for precision, derived from exploration-exploitation in optimization.

## Mathematical Comparisons

### Similarities
- **GA Inspiration**: Both draw from evolutionary principles—FedCSGA uses full GA operators; GenFed uses fitness-based selection.
- **Fitness-Based Selection**: FedCSGA's fitness optimizes subsets; GenFed selects by model accuracy.
- **Heterogeneity Handling**: Address non-IID (data) and system (delays) issues via quality metrics.
- **Aggregation**: Both use FedAvg (weighted or unweighted variants).

### Differences
- **Scope**:
  - FedCSGA: Pre-selection (which clients participate).
  - GenFed: Post-selection (which models to aggregate).
- **Fitness**:
  - FedCSGA: Composite (participation + accuracy), evaluated via simulation.
  - GenFed: Direct (validation accuracy), no simulation needed.
- **GA Implementation**:
  - FedCSGA: Full GA (population, generations, crossover, mutation).
  - GenFed: Simplified (sort-and-select, no evolution).
- **Parameters**:
  - FedCSGA: Population 90, gen 10, adaptive ops.
  - GenFed: No fixed GA params; dynamic \( \rho_t \).
- **Computational Cost**:
  - FedCSGA: High (fitness simulation per chromosome).
  - GenFed: Low (single sort per round).
- **Optimality**:
  - FedCSGA: Approximates NP-hard selection via GA.
  - GenFed: Exact top-selection, but assumes all clients participate.

### Theoretical Advantages/Disadvantages
- **FedCSGA**: Better for constrained selection (e.g., time limits), handles complex trade-offs. Risk of local optima; higher overhead.
- **GenFed**: Simpler, scalable; focuses on aggregation quality. Assumes sufficient participation; less adaptive to selection constraints.

### Detailed Mathematical Trade-Offs
- **Convergence Analysis**: FedCSGA minimizes selection regret \( R(S) = \mathbb{E}[ \mathcal{L}(\theta_S) - \mathcal{L}(\theta^*) ] \), where \( \theta_S \) is aggregated from \( S \). GA approximates optimal \( S^* \). GenFed minimizes aggregation error \( E(\rho_t) = \mathbb{E}[ \|\theta_{t+1} - \theta^*\|^2 ] \), with \( \rho_t \) controlling variance-bias.
- **Scalability**: FedCSGA's \( O(P \cdot G \cdot C) \) (P=population, G=generations, C=cost per fitness) vs. GenFed's \( O(K \log K) \), favoring GenFed for large K.
- **Robustness**: FedCSGA's simulation-based fitness handles non-stationary client quality; GenFed's validation-based fitness assumes neutral data.

## Performance Analysis

Based on papers' results (simulated on CIFAR-10/MNIST, non-IID):

- **Accuracy Gains**:
  - FedCSGA: 2.4%-7.7% better than baselines (FedAvg, FedCS).
  - GenFed: 5% better on CIFAR-10, robust to attacks.
- **Convergence**:
  - FedCSGA: 45.6%-54.3% more clients selected, faster convergence.
  - GenFed: 10-35x fewer rounds to target accuracy.
- **Efficiency**:
  - FedCSGA: Reduces communication by optimizing selection.
  - GenFed: Low overhead, stable in varying client counts.
- **Comparison**: FedCSGA excels in selection optimization; GenFed in aggregation robustness. Combined, they could complement (e.g., GA selection + top aggregation).

## Conclusion

FedCSGA and GenFed both leverage GA concepts to enhance FL but target different stages: selection vs. aggregation. Mathematically, FedCSGA's composite fitness and evolutionary operators provide deeper optimization at higher cost, while GenFed's direct selection offers efficiency and simplicity. In practice, FedCSGA suits resource-constrained selection, GenFed suits large-scale aggregation. Future work could integrate both for hybrid FL optimization.

## References
- Wu, J., et al. (2025). Optimizing Client Selection in Federated Learning Base on Genetic Algorithm. *Cluster Computing*.
- Zheng, H., et al. (2025). Accelerating Federated Learning with genetic algorithm enhancements. *Expert Systems with Applications*.