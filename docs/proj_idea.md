
##set page(margin: (x: 1in, y: 1in))
#set text(font: "Libertinus Serif", size: 12pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

// Title page styling
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Study of the Feasibility of Using Genetic Algorithms in Client Selection in Federated Learning
  ]
  
  #v(1em)
  
  #text(size: 14pt)[
    Team 16
  ]
  
  #v(0.5em)
  
  #text(size: 12pt)[
    Sarthak Mishra — 22b0432 \
    Abhimanyu Singh Rathore — 22b1806
  ]
]

#v(2em)

= Problem Statement 
Federated Learning (FL) allows collaborative model training across distributed clients without sharing raw data, but challenges like data heterogeneity (non-IID distributions) and inefficient client selection can lead to slow convergence and suboptimal performance. This project explores using genetic algorithms (GAs) to optimize client selection in FL, aiming to select participants that maximize model quality while minimizing communication costs, thereby improving overall efficiency in resource-constrained scenarios. 
= Abstract 
Federated Learning (FL) has become a cornerstone for privacy-preserving machine learning, enabling multiple clients—such as mobile devices or edge nodes—to train a shared model by exchanging only model updates, not raw data. However, in real-world deployments, clients often exhibit heterogeneous data distributions (non-IID), varying computational capabilities, and unreliable participation, which can degrade the global model's convergence speed and accuracy. Traditional FL algorithms like FedAvg randomly or uniformly select clients, often overlooking factors like data quality or resource availability, leading to inefficient aggregation and potential bias toward dominant clients. This project proposes a simple implementation of a GA-optimized client selection mechanism in FL, inspired by recent frameworks like FedCSGA, to address these issues. We will simulate an FL environment with 10-20 clients using the MNIST dataset for image classification, partitioned non-IID via a Dirichlet distribution (alpha=0.5 for moderate heterogeneity). The base model will be a lightweight multilayer perceptron (MLP) with two hidden layers (128 and 64 units) to ensure quick training. In each communication round, a GA will evolve a population of client subsets (e.g., population size 50, generations 10) using fitness functions based on estimated client contributions, such as local accuracy or gradient norms, to select the top-k clients (e.g., k=5) for updates. The GA implementation will use basic evolutionary operations: selection via tournament, crossover with probability 0.8, and mutation at 0.1, implemented manually with NumPy for simplicity 
The workflow includes: 
+ Client data preparation and local training with SGD (learning rate 0.01, batch size 32, 5 local epochs). 
+ GA-based selection on the server. 
+ FedAvg aggregation of selected updates. 
+ Evaluation over 100 rounds, tracking global accuracy, loss, convergence rate, and communication efficiency (e.g., bytes per round). 
For comparison, we will benchmark against standard FedAvg with random selection. Expected outcomes indicate GA-optimized selection could improve accuracy by 5-10% and reduce rounds to convergence by 20-30% on non-IID MNIST, based on similar studies, while maintaining low overhead. This demonstrates GA's value in practical FL for applications like IoT or healthcare. If time allows, ablate GA parameters or test on Fashion-MNIST. The setup uses PyTorch for ease, runnable on a standard laptop in days, providing hands-on insights into bio-inspired optimizations in distributed learning.


= References

Wang, Y., et al. (2025). Optimizing Client Selection in Federated Learning Based on Genetic Algorithm with Adaptive Operators. _Cluster Computing_. https://doi.org/10.1007/s10586-025-05106-5

Kang, H., & Ahn, J. (2025). Accelerating Federated Learning with Genetic Algorithm Enhancements. _Expert Systems with Applications_. https://doi.org/10.1016/j.eswa.2025.127636
s#set page(margin: (x: 1in, y: 1in))
#set text(font: "Libertinus Serif", size: 12pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

// Title page styling
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Study of the Feasibility of Using Genetic Algorithms in Client Selection in Federated Learning
  ]
  
  #v(1em)
  
  #text(size: 14pt)[
    Team 16
  ]
  
  #v(0.5em)
  
  #text(size: 12pt)[
    Sarthak Mishra — 22b0432 \
    Abhimanyu Singh Rathore — 22b1806
  ]
]

#v(2em)

= Problem Statement 
Federated Learning (FL) allows collaborative model training across distributed clients without sharing raw data, but challenges like data heterogeneity (non-IID distributions) and inefficient client selection can lead to slow convergence and suboptimal performance. This project explores using genetic algorithms (GAs) to optimize client selection in FL, aiming to select participants that maximize model quality while minimizing communication costs, thereby improving overall efficiency in resource-constrained scenarios. 
= Abstract 
Federated Learning (FL) has become a cornerstone for privacy-preserving machine learning, enabling multiple clients—such as mobile devices or edge nodes—to train a shared model by exchanging only model updates, not raw data. However, in real-world deployments, clients often exhibit heterogeneous data distributions (non-IID), varying computational capabilities, and unreliable participation, which can degrade the global model's convergence speed and accuracy. Traditional FL algorithms like FedAvg randomly or uniformly select clients, often overlooking factors like data quality or resource availability, leading to inefficient aggregation and potential bias toward dominant clients. This project proposes a simple implementation of a GA-optimized client selection mechanism in FL, inspired by recent frameworks like FedCSGA, to address these issues. We will simulate an FL environment with 10-20 clients using the MNIST dataset for image classification, partitioned non-IID via a Dirichlet distribution (alpha=0.5 for moderate heterogeneity). The base model will be a lightweight multilayer perceptron (MLP) with two hidden layers (128 and 64 units) to ensure quick training. In each communication round, a GA will evolve a population of client subsets (e.g., population size 50, generations 10) using fitness functions based on estimated client contributions, such as local accuracy or gradient norms, to select the top-k clients (e.g., k=5) for updates. The GA implementation will use basic evolutionary operations: selection via tournament, crossover with probability 0.8, and mutation at 0.1, implemented manually with NumPy for simplicity 
The workflow includes: 
+ Client data preparation and local training with SGD (learning rate 0.01, batch size 32, 5 local epochs). 
+ GA-based selection on the server. 
+ FedAvg aggregation of selected updates. 
+ Evaluation over 100 rounds, tracking global accuracy, loss, convergence rate, and communication efficiency (e.g., bytes per round). 
For comparison, we will benchmark against standard FedAvg with random selection. Expected outcomes indicate GA-optimized selection could improve accuracy by 5-10% and reduce rounds to convergence by 20-30% on non-IID MNIST, based on similar studies, while maintaining low overhead. This demonstrates GA's value in practical FL for applications like IoT or healthcare. If time allows, ablate GA parameters or test on Fashion-MNIST. The setup uses PyTorch for ease, runnable on a standard laptop in days, providing hands-on insights into bio-inspired optimizations in distributed learning.


= References

Wang, Y., et al. (2025). Optimizing Client Selection in Federated Learning Based on Genetic Algorithm with Adaptive Operators. _Cluster Computing_. https://doi.org/10.1007/s10586-025-05106-5

Kang, H., & Ahn, J. (2025). Accelerating Federated Learning with Genetic Algorithm Enhancements. _Expert Systems with Applications_. https://doi.org/10.1016/j.eswa.2025.127636
et page(margin: (x: 1in, y: 1in))
#set text(font: "Libertinus Serif", size: 12pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

// Title page styling
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Study of the Feasibility of Using Genetic Algorithms in Client Selection in Federated Learning
  ]
  
  #v(1em)
  
  #text(size: 14pt)[
    Team 16
  ]
  
  #v(0.5em)
  
  #text(size: 12pt)[
    Sarthak Mishra — 22b0432 \
    Abhimanyu Singh Rathore — 22b1806
  ]
]

#v(2em)

= Problem Statement 
Federated Learning (FL) allows collaborative model training across distributed clients without sharing raw data, but challenges like data heterogeneity (non-IID distributions) and inefficient client selection can lead to slow convergence and suboptimal performance. This project explores using genetic algorithms (GAs) to optimize client selection in FL, aiming to select participants that maximize model quality while minimizing communication costs, thereby improving overall efficiency in resource-constrained scenarios. 
= Abstract 
Federated Learning (FL) has become a cornerstone for privacy-preserving machine learning, enabling multiple clients—such as mobile devices or edge nodes—to train a shared model by exchanging only model updates, not raw data. However, in real-world deployments, clients often exhibit heterogeneous data distributions (non-IID), varying computational capabilities, and unreliable participation, which can degrade the global model's convergence speed and accuracy. Traditional FL algorithms like FedAvg randomly or uniformly select clients, often overlooking factors like data quality or resource availability, leading to inefficient aggregation and potential bias toward dominant clients. This project proposes a simple implementation of a GA-optimized client selection mechanism in FL, inspired by recent frameworks like FedCSGA, to address these issues. We will simulate an FL environment with 10-20 clients using the MNIST dataset for image classification, partitioned non-IID via a Dirichlet distribution (alpha=0.5 for moderate heterogeneity). The base model will be a lightweight multilayer perceptron (MLP) with two hidden layers (128 and 64 units) to ensure quick training. In each communication round, a GA will evolve a population of client subsets (e.g., population size 50, generations 10) using fitness functions based on estimated client contributions, such as local accuracy or gradient norms, to select the top-k clients (e.g., k=5) for updates. The GA implementation will use basic evolutionary operations: selection via tournament, crossover with probability 0.8, and mutation at 0.1, implemented manually with NumPy for simplicity 
The workflow includes: 
+ Client data preparation and local training with SGD (learning rate 0.01, batch size 32, 5 local epochs). 
+ GA-based selection on the server. 
+ FedAvg aggregation of selected updates. 
+ Evaluation over 100 rounds, tracking global accuracy, loss, convergence rate, and communication efficiency (e.g., bytes per round). 
For comparison, we will benchmark against standard FedAvg with random selection. Expected outcomes indicate GA-optimized selection could improve accuracy by 5-10% and reduce rounds to convergence by 20-30% on non-IID MNIST, based on similar studies, while maintaining low overhead. This demonstrates GA's value in practical FL for applications like IoT or healthcare. If time allows, ablate GA parameters or test on Fashion-MNIST. The setup uses PyTorch for ease, runnable on a standard laptop in days, providing hands-on insights into bio-inspired optimizations in distributed learning.


= References

Wang, Y., et al. (2025). Optimizing Client Selection in Federated Learning Based on Genetic Algorithm with Adaptive Operators. _Cluster Computing_. https://doi.org/10.1007/s10586-025-05106-5

Kang, H., & Ahn, J. (2025). Accelerating Federated Learning with Genetic Algorithm Enhancements. _Expert Systems with Applications_. https://doi.org/10.1016/j.eswa.2025.127636
