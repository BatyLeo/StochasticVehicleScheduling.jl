```@meta
EditURL = "<unknown>/docs/src/literate/algorithms.jl"
```

# Optimization algorithms

In this section, we describe the various optimization algorithms implemented in this
package, and illustrate them on the following small instance, with 25 tasks and 10 scenarios:

````@example algorithms
using StochasticVehicleScheduling
instance = create_random_instance(nb_tasks=25, nb_scenarios=10)
plot_instance_on_map(instance)
````

## Heuristic algorithms

The first category of algorithms implemented are heuristics, which given good but not
necessarily optimal solutions in a short amount of time.

### Deterministic heuristic

The first heuristic consists in solving a deterministic version of the instance, which
minimizes vehicle costs plus travel costs of vehicles:

```math
\begin{aligned}
\min & \, c_{\text{vehuicle}}\sum_{a\in \delta^+(o)} y_a + c_{\text{delay}}\sum_{a\in a}w_a y_a &\\
s.t. & \sum_{a\in \delta^-(v)} y_a = \sum_{a\in \delta^+(v)} y_a, & \forall v \in\bar V\\
& \sum_{a\in \delta^-(v)} y_a = 1, & \forall v \in\bar V\\
& y_a \in \{0, 1\}, &\forall a\in A
\end{aligned}
```

This is a linear flow program very easy to solve:

````@example algorithms
_, h_solution = solve_deterministic_VSP(instance)
h_value = evaluate_solution(h_solution, instance)
println("Heuristic solution value: $h_value")
````

### Local search

A more advanced heuristic is the local search, which uses the deterministic solution as an
initialization point, and try to improve it by applying elementary modifications to it.

````@example algorithms
ls_solution = heuristic_solution(instance; nb_it=1_000)
ls_value = evaluate_solution(ls_solution, instance)
println("Local search solution value: $ls_value")
````

We obtain a better solution than the previous heuristic!

## Exact algorithms

There are also two exact optimization algorithms implemented in this package.

### MIP formulation

One way to solve the stochastic vehicle scheduling problem is to model it as the following linear program with quadratic constraints:
```math
\begin{aligned}
\min_{d, y} & \,c_{\text{delay}}\dfrac{1}{|S|}\sum\limits_{s\in S}\sum\limits_{v\in V\backslash\{o,d\}} d_v^s + c_{\text{vehicle}} \sum\limits_{a\in\delta^+(o)}y_a\\
\text{s.t.} & \sum_{a\in\delta^-(v)}y_a = \sum_{a\in\delta^+(v)}y_a &\forall v\in \bar V\\
& \sum_{a\in\delta^-(v)}y_a = 1 &\forall v\in \bar V\\
& d_v^s \geq \gamma_v^s + \sum_{\substack{a\in\delta^-(v) \\ a=(u, v)}} (d_u^s - \delta_{u, v}^s) y_a & \forall v\in \bar V, \forall s\in S\\
& d_v^s\geq \gamma_v^s & \forall v\in \bar V, \forall s\in S\\
& y_a\in\{0,1\} & \forall a\in A
\end{aligned}
```
Quadratic delay constraints can be linearized using [Mc Cormick linearization](https://optimization.mccormick.northwestern.edu/index.php/McCormick_envelopes).

``\implies`` does not scale well with tasks and scenarios number

````@example algorithms
mip_value, mip_solution = solve_scenarios(instance)
println("MIP optimal value: $mip_value")
````

The solution value is better than both heuristic values, as expected.

### Column generation formulation

Another option is to use a column generation approach with variables $y_P$ which equals one if route $P\in \mathycal P$ is selected.
Cost of a path ``P``: ``c_P^s = c_{\text{vehicle}} + c_\text{delay}\times \sum_{v\in P} d_v^s``

```math
\begin{aligned}
\min & \frac{1}{|S|}\sum_{s\in S}\sum_{P\in\mathcal{P}}c_P^s y_P &\\
\text{s.t.} & \sum_{p\ni v} y_P = 1 & \forall v\in \bar V & \quad(\lambda_v\in\mathbb R)\\
& y_P\in\{0,1\} & \forall p\in \mathcal{P} &
\end{aligned}
```

The associated sub-problem of the column generation formulation is a constrained shortest path problem of the form :
```math
\min_{P\in\mathcal P} (c_P  - \sum_{v\in P}\lambda_v)
```
It can be solved using generalized ``A^\star`` algorithms (cf. [Parmentier 2017](https://arxiv.org/abs/1504.07880) and [ConstrainedShortestPath.jl](https://github.com/BatyLeo/ConstrainedShortestPaths.jl)).

``\implies`` still does not scale well when the number of tasks and scenarios increases

````@example algorithms
col_solution = column_generation_algorithm(instance)
col_value = evaluate_solution(col_solution, instance)
println("Column generation optimal value: $col_value")
````

The column generation solution has the same value as the MIP one, as expected (both are optimal).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
