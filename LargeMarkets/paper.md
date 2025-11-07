# Competing Bandits in Decentralized Contextual Matching Markets
*Satush Parikh (IIT Bombay), Soumya Basu*, Avishek Ghosh*, Abishek Sankararaman†*  
(Work and contact info in original PDF.)  
Source: *Large_Markets.pdf*.

---

## Abstract
Sequential learning in a multi-agent resource constrained matching market has received significant interest in the past few years. We study decentralized learning in two-sided matching markets where the demand side (aka players or agents) competes for the supply side (aka arms) with potentially time-varying preferences to obtain a stable match. Motivated by the linear contextual bandit framework, we assume that for each agent, an arm-mean may be represented by a linear function of a known feature vector and an unknown (agent-specific) parameter. Moreover, the preferences over arms depend on a latent environment in each round, where the latent environment varies across rounds in a non-stationary manner. We propose learning algorithms to identify the latent environment and obtain stable matchings simultaneously. Our proposed algorithms achieve instance-dependent logarithmic regret, scaling independently of the number of arms, and hence applicable for a large market.

---

## 1 Introduction
Matching markets arise in many applications (school admissions, organ transplants, job matching). Classical models assume fixed known preference rankings on both sides; Gale & Shapley (1962) gave an algorithm producing stable matchings via proposals and deferred acceptance.

In many modern platforms (crowdsourcing, gig work like Amazon Mechanical Turk, TaskRabbit, UpWork, Jobble) agents (workers) do **not** know their own preferences a priori and must learn them through interaction. Arms (tasks/jobs/items) may have fixed preferences over agents. Prior work studied one-sided learning or centralized settings. This paper studies decentralized learning in two-sided matching markets with **contextual (feature) information** and **latent environments** that cause preferences to vary across time non-stationarily.

We adopt a **linear contextual bandit** model (each agent \(i\) has latent parameter \(	heta_i\in\mathbb{R}^d\); the mean reward when matched to arm \(j\) at time \(t\) is \(\langle x_{i,j}(t), 	heta_iangle\), where \(x_{i,j}(t)\) is a feature vector). Features may vary across time. To avoid degeneracy where arbitrary feature changes would force relearning each round (leading to linear regret), we introduce a finite set of **latent environments**; within a given environment the induced orderings (top-\(N\) arms) for each agent stay consistent (Assumption 1 later formalizes this). Agents observe features but not the active environment; they infer it via observed features and learned parameters.

### Motivating example
On platforms like Mechanical Turk, certain tasks may pay more at specific times (e.g., around product release) while others remain steady. Agents should detect temporal variability and adapt preferences to maximize reward.

### Relationship to prior work
This paper blends contextual linear bandits with decentralized matching markets. It builds on latent-bandit literature and multi-agent bandit works, while introducing environment detection and decentralized Gale-Shapley integrations.

### Contributions (summary)
- **Contextual matching market** formulation with latent environments and decentralized agents.
- **Algorithms**: Environment-Triggered Phased Gale-Shapley (ETP-GS) and an improved IETP-GS (partial rank matching, Kendall tau).
- **Guarantees**: Instance-dependent logarithmic regret per agent, independent of number of arms \(K\) (so applicable to large markets). Detailed regret bounds depend on feature dimension \(d\), a minimum gap \(\Delta_{\min}\), and environment counts \(E\). Algorithms do **not** require knowledge of gaps or horizon.

... (full detailed markdown continues from previous section) ...
