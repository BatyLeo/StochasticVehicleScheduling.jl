# Mathematical formulation

Here we define the Stochastic vehicle scheduling problem and how to build an InferOpt end-to-end learning pipeline which can predict solutions of this difficult problem.

## Deterministic Vehicle Scheduling

Vehicle Scheduling involves assigning vehicles to cover a set of scheduled tasks, while minimising the number of vehicles used and other costs.
### Instance
An instance of the problem is composed of:
- A set of tasks ``v\in\bar V``:
    - Scheduled task begin time: ``t_v^b``
    - Scheduled task end time: ``t_v^e (> t_v^b)``
    - Scheduled travel time from task ``u`` to task ``v``: ``t_{(u, v)}^{tr}``
- Task ``v`` can be scheduled after task ``u`` on a vehicle path only if: ``t_v^b \geq t_u^e + t_{(u, v)}^{tr}``
- Vehicle cost: ``c_{\text{vehicle}}``
- Objective to minimize: ``c_{\text{vehicle}}\times [\text{number of vehicle used}]``

### Mathematical model
We model an instance by the following acyclic Digraph ``D = (V, A)``:
- ``V = \bar V\cup \{o, d\}``, with ``o`` and ``d`` dummy origin and destination nodes connected to all tasks:
    - ``(o, v)`` arc for all task ``v\in \bar V``
    - ``(v, d)`` arc for all task ``v \in \bar V``
- There is an arc between tasks ``u`` and ``v`` only if ``t_v^b \geq t_u^e + t_{(u, v)}^{tr}``

A feasible vehicle tour is an ``o-d`` path ``P\in\mathcal P``. A feasible solution is a set of disjoint feasible vehicle tours fulfilling all tasks exctly once. This can be formulated as the following flow Linear Program:
```math
\boxed{\begin{aligned}
\min & \, c_{\text{vehicle}}\sum_{a\in \delta^+(o)} y_a &\\
s.t. & \sum_{a\in \delta^-(v)} y_a = \sum_{a\in \delta^+(v)} y_a, & \forall v \in V\backslash \{o, d\}\\
& \sum_{a\in \delta^-(v)} y_a = 1, & \forall v \in V\backslash \{o, d\}\\
& y_a \in \{0, 1\}, &\forall a\in A
\end{aligned}}
```

``\implies`` easy to solve

---

## Stochastic Vehicle Scheduling

Stochastic vehoicle scheduling is a variation of the deterministic version presented above, but scheduled task times can be perturbed by random events after the scheduling is fixed. The goal is now to also minimize the expectation of the detotal delay.

We consider the same framework as baove, to which we add the following:
- Delay cost: ``c_\text{delay}``
- Set of scenarios ``\Omega``.
- We indroduce three sets of random variables, wich can take different values depending on the scenario ``\omega\in\Omega``
    - Perturbed beginning time: ``\xi_v^b \geq t_v^b``
    - Perturbed end time: ``\xi_v^e \geq t_v^e``
    - Perturbed travel time: ``\xi_{(u,v)}^{tr} \geq t_{(u,v)}^{tr}``
- We fix ``\xi_o^e = 0`` and ``\xi_d^b = +\infty``
- Given an ``o-uv`` path ``P``, we define recursively the end time ``\tau_v``, and the total delay ``\Delta_v`` along ``P``:
```math
\boxed{\left\{\begin{aligned}
\tau_v &= \xi_v^e + \max(\tau_u +\xi_{(u,v)}^{tr} - \xi_v^b, 0)\\
\Delta_v &= \Delta_u + \max(\tau_u +\xi_{(u,v)}^{tr} - \xi_v^b, 0)
\end{aligned}\right.}
```

- Objective to minimize: ``c_{\text{vehicle}} \times [\text{number of vehicles used}] + c_\text{delay} \times \mathbb E[\text{total delay}]``

### MIP formulation
```math
\begin{aligned}
\min & \,c_{\text{vehicle}}\sum_{a\in \delta^+(o)} y_a + \frac{c_{\text{delay}}}{|\Omega|}\sum_{\omega\in\Omega}\sum_{v\in V\backslash\{o, d\}} \Delta_v^\omega\\
\text{s.t.} & \sum_{a\in\delta^-(v)}y_a = \sum_{a\in\delta^+(v)}y_a &\forall v\in V\backslash\{o, d\}\\
& \sum_{a\in\delta^-(v)}y_a = 1 &\forall v\in V\backslash\{o, d\}\\
& \Delta_v^\omega \geq \sum_{a\in\delta^-(v), a=(u, v)} (\Delta_u^\omega + \tau_u^\omega + \xi_{(u,v)}^{tr}(\omega) - \xi_v^b(\omega)) y_a & \forall v\in V\backslash\{o, d\}, \forall \omega\in\Omega\\
& \tau_v^\omega \geq \xi_v^e(\omega) + \sum_{a\in\delta^-(v), a=(u, v)} (\tau_u^\omega + \xi_{(u,v)}^{tr}(\omega) - \xi_v^b(\omega)) y_a & \forall v\in V\backslash\{o, d\}, \forall \omega\in\Omega\\
& \Delta_v^\omega\geq 0 & \forall v\in V\backslash\{o, d\}, \forall \omega\in\Omega\\
& \tau_v^\omega\geq 0 & \forall v\in V\backslash\{o, d\}, \forall \omega\in\Omega\\
& y_a\in\{0,1\} & \forall a\in A
\end{aligned}
```
- Quadratic delay constraints can be linearized using [Mc Cormick linearization](https://optimization.mccormick.northwestern.edu/index.php/McCormick_envelopes).

``\implies`` does not scale well with tasks and scenarios number

### Column generation formulation

Cost of a path ``P``: ``c_P^\omega = c_{\text{vehicle}} + c_\text{delay}\times \Delta_P^\omega``

```math
\begin{aligned}
\min & \frac{1}{|\Omega|}\sum_{\omega\in\Omega}\sum_{p\in\mathcal{P}}c_p^\omega y_p &\\
\text{s.t.} & \sum_{p\ni v} y_p = 1 &\forall v\in V\backslash\{o, d\} \quad(\lambda_v\in\mathbb R)\\
& y_p\in\{0,1\} & \forall p\in \mathcal{P}
\end{aligned}
```

This formulation can be solved using a column generation algorithm. The associated subproblem is a constrained shortest path problem of the form :
```math
\min_{P\in\mathcal P} (c_P  - \sum_{v\in P}\lambda_v)
```

This kind of problem can be solved using generalized ``A^\star`` algorithms (cf. [Parmentier 2017](https://arxiv.org/abs/1504.07880) and [ConstrainedShortestPath.jl](https://github.com/BatyLeo/ConstrainedShortestPaths.jl)).

``\implies`` better, but still does not scale well
