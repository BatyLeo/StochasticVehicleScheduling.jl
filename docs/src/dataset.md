# Dataset generation
Let ``(x_1, \dots, x_N)`` a set of instances of the stochastic scheduling problem. For this example, we will place ourselves in the [learning by imitation setting](https://axelparmentier.github.io/InferOpt.jl/dev/math/#Learning-by-imitation), with precomputed *good* solutions ``\bar y_i`` for each instance ``x_i``. So we need to generate a set of instances ``(x_1, \dots, x_N)`` and associated solutions ``(\bar y_1, \dots, \bar y_N)``.

The function [`generate_dataset`](@ref StochasticVehicleScheduling.generate_dataset) does all of this. It calls other functions, that are described below.

---
## Instances

Instances generation and features computing are the same are similar as in [Parmentier](https://pubsonline.informs.org/doi/abs/10.1287/opre.2020.2094). See the article for more details.
### Generation

The function [`create_random_city`](@ref StochasticVehicleScheduling.create_random_city) creates a random instance with given number of tasks and number of scenarios.

By default, a [`City`](@ref StochasticVehicleScheduling.City) is a square of 50 minutes width, divided in 25 squared (10 minutes width) districts. Each task ``v`` has (uniformly drawn) start point, start time ``t_v^b``, end point, and end time ``t_v^e``.

For each scenario ``\omega``, we roll the following random variables:
- For each district ``d`` and hour ``h`` of the day, ``\varepsilon_{d, h}`` are independent log-normal variables with randomly (uniform) drawn parameters ``(\mu, \sigma)\in [1, 3]\times [0.4, 0.6]``
- For each district ``d`` and hour ``h`` of the day, ``\zeta_{d,h}^{dis}`` models the congestion in the district at this time.
    - ``\forall d,\, \zeta^{dis}_{d, 0} = \varepsilon_{d,0}``
    - ``\forall d,h,\, \zeta^{dis}_{d, h+1} = \frac{1}{2}\zeta^{dis}_{d, h} + \varepsilon_{d,h}``
- For each hour ``h`` of the day, ``\zeta^{inter}_h`` models the congestion on roads between districts, and is computed similarly:
    - ``I\sim \log\mathcal{N}(\mu=0.02, \sigma=0.05)``
    - ``\zeta^{inter}_0 = I``
    - ``\zeta^{inter}_{h+1} = (\zeta^{inter}_{h} + 0.1)I  ``

Let ``v`` be a task corresponding to a trip between district ``d_1`` and ``d_2``. We compute the perturbed start, end, and travel times are computed like this:
- ``\xi_v^b = t_v^b § \varepsilon_v``
- ``\xi_v^e = \xi_v^b + t_v^e - t_v^b + \zeta^{dis}_{d_1,h(\xi_1)} + \zeta^{inter}_{h(\xi_2)} + \zeta^{dis}_{d_2, h(\xi_3)}``
    - ``\xi_1 = \xi_v^b``
    - ``\xi_2 = \xi_1 + \zeta^{dis}_{d_1,h(\xi_1)}``
    - ``\xi_3 = \xi_2 + t_v^e - t_v^b + \xi^{inter}_{h(\xi_2)}``
- ``a=(u,v)\rightarrow`` ``\xi_a^{tr} = \xi_v^b + t_a^{tr} + \zeta^{dis}_{d_1,h(\xi_1)} + \zeta^{inter}_{h(\xi_2)} + \zeta^{dis}_{d_2, h(\xi_3)}``
    - ``\xi_1 = \xi_u^e``
    - ``\xi_2 = \xi_1 + \zeta^{dis}_{d_1,h(\xi_1)}``
    - ``\xi_3 = \xi_2 + t_a^{tr} + \xi^{inter}_{h(\xi_2)}``
### Encoding

For an instance, the [`compute_features`](@ref StochasticVehicleScheduling.compute_features) function computes a matrix of 20 features for every arc of the corresponding graph :
- Length of the arc in minutes (deterministic travel time)
- Cost of a vehicle if the arc if connected to the source
- The 9 deciles of the slack ``\xi_v^b - (\xi_u^e + \xi_a^{tr})``
- The cumulative probability distribution of the slack, evaluated in ``[-100, -50,-20,-10,0,50,200,500]``

---
## Solutions

We label each instance with [`heuristic_solution`](@ref StochasticVehicleScheduling.heuristic_solution), wich is a local search initialized with the solution of the deterministic problem (see [`solve_deterministic_VSP`](@ref StochasticVehicleScheduling.solve_deterministic_VSP))
