# Problem statement

## Deterministic Vehicle Scheduling

The Vehicle Scheduling Problem (VSP) consists in assigning vehicles to cover a set of scheduled tasks, while minimizing total costs. Let ``\bar V`` be the set of tasks. Each task ``v\in \bar V`` has a scheduled beginning time ``t_v^b``, and a scheduled end time ``t_v^e``, such that ``t_v^e > t_v^b``. We denote ``t^{tr}_{(u, v)}`` the travel time from task ``u`` to task ``v``. A task ``v`` can be scheduled after another task ``u`` only if we can reach it in time, before it starts, i.e. if
```math
t_v^b \geq t_u^e + t^{tr}_{(u, v)}
```

An instance of VSP can be modeled with a directed graph ``D = (V, A)``, with ``V = \bar V\cup\{o, d\}``, and ``o``, ``d`` origin and destination dummy nodes. All tasks are connected to ``o``, and ``d`` is connected to all tasks. There is an arc between two tasks ``u`` and ``v`` only if the equation above is satisfied. The resulting graph ``D`` is acyclic.

A solution of the VSP problem is a list of disjoint ``o-d`` paths such that all tasks are fulfilled exactly once. The objective is to minimize the number of vehicles used. This can be formulated as the following Linear Program:

```math
\begin{aligned}
\min & \, \sum_{a\in \delta^+(o)} y_a &\\
s.t. & \sum_{a\in \delta^-(v)} y_a = \sum_{a\in \delta^+(v)} y_a, & \forall v \in \bar V\\
& \sum_{a\in \delta^-(v)} y_a = 1, & \forall v \in \bar V\\
& (y_a \in \{0, 1\}), &\forall a\in A
\end{aligned}
```

This can be solved either using a flow algorithm, or using a linear program solver (constraints form a flow polytope, the binary constraint can be relaxed, and we obtain an easy to solve Linear Program).

---

## Stochastic Vehicle Scheduling

In the Stochastic Vehicle Scheduling Problem (StoVSP), we consider the same setting as the deterministic version, to which we add the following. Once the scheduling decision is set, we observe random delays, which propagate along vehicle tours. The objective is to minimize the sum of vehicle costs and expected total delay costs.

We consider a finite set of scenarios ``s\in S``. For each task ``v\in \bar V``, we denote ``\gamma_v^s`` the intrinsic delay of ``v`` in scenario ``s``, and ``d_v^s`` its total delay. We also denote ``\delta_{u, v}^s`` the slack between tasks ``u`` and ``v``. These quantities follow the delay propagation equation when ``u`` and ``v`` are consecutively operated by the same vehicle:
```math
d_v^s = \gamma_v^s + \max(d_u^s - \delta_{u, v}^s, 0)
```

This leads to a much more difficult problem to solve. If the instance isn't too big, we can solve it using an integer program with quadratic constraints or a column generation algorithm.
